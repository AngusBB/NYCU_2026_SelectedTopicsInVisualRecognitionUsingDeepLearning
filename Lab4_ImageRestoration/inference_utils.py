"""Shared checkpoint and inference helpers."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.nn import functional as F

from models import PromptIR


def load_promptir_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[PromptIR, dict]:
    """Load a PromptIR checkpoint saved by train.py."""

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = checkpoint.get("model_args", {"decoder": True})
    model = PromptIR(**model_args).to(device)
    state_dict = checkpoint["state_dict"]
    cleaned_state = {
        key.removeprefix("module."): value for key, value in state_dict.items()
    }
    model.load_state_dict(cleaned_state, strict=True)
    model.eval()
    return model, checkpoint


def pad_to_multiple(
    tensor: torch.Tensor,
    multiple: int = 8,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad BCHW tensor on bottom/right so H and W are divisible by multiple."""

    height, width = tensor.shape[-2:]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return tensor, (height, width)
    padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return padded, (height, width)


def _tta_transform(tensor: torch.Tensor, mode: int) -> torch.Tensor:
    """Apply one of 8 dihedral test-time augmentations."""

    if mode < 4:
        return torch.rot90(tensor, k=mode, dims=(-2, -1))
    rotated = torch.rot90(tensor, k=mode - 4, dims=(-2, -1))
    return torch.flip(rotated, dims=(-1,))


def _tta_inverse(tensor: torch.Tensor, mode: int) -> torch.Tensor:
    """Invert one of 8 dihedral test-time augmentations."""

    if mode < 4:
        return torch.rot90(tensor, k=-mode, dims=(-2, -1))
    unflipped = torch.flip(tensor, dims=(-1,))
    return torch.rot90(unflipped, k=-(mode - 4), dims=(-2, -1))


@torch.no_grad()
def restore_batch(
    model: torch.nn.Module,
    degraded: torch.Tensor,
    amp: str = "bf16",
    tta: bool = False,
) -> torch.Tensor:
    """Run restoration and crop away any inference padding."""

    device = next(model.parameters()).device
    original_size = degraded.shape[-2:]
    degraded = degraded.to(device, non_blocking=True)
    degraded, _ = pad_to_multiple(degraded)

    if device.type == "cuda" and amp != "off":
        dtype = torch.float16 if amp == "fp16" else torch.bfloat16
        context = torch.autocast(device_type="cuda", dtype=dtype)
    else:
        context = torch.no_grad()

    with context:
        if tta:
            restored_sum = None
            for mode in range(8):
                augmented = _tta_transform(degraded, mode)
                restored_aug = model(augmented).clamp(0.0, 1.0)
                restored_mode = _tta_inverse(restored_aug, mode)
                restored_sum = (
                    restored_mode
                    if restored_sum is None
                    else restored_sum + restored_mode
                )
            restored = restored_sum / 8.0
        else:
            restored = model(degraded).clamp(0.0, 1.0)
    height, width = original_size
    return restored[..., :height, :width]
