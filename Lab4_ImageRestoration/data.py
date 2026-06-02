"""Dataset and image helpers for the HW4 PromptIR baseline."""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ImagePair:
    """One degraded/clean training pair."""

    name: str
    degraded_path: Path
    clean_path: Path
    kind: str
    index: int


def natural_key(path_or_name: str | Path) -> list[int | str]:
    """Sort filenames as humans expect, e.g. 2.png before 10.png."""

    name = Path(path_or_name).name
    return [
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", name)
    ]


def list_test_images(data_root: str | Path) -> list[Path]:
    """Return sorted unlabeled test images."""

    test_dir = Path(data_root) / "test" / "degraded"
    return sorted(test_dir.glob("*.png"), key=natural_key)


def collect_pairs(data_root: str | Path) -> list[ImagePair]:
    """Collect paired rain/snow train images and verify completeness."""

    root = Path(data_root)
    degraded_dir = root / "train" / "degraded"
    clean_dir = root / "train" / "clean"
    pairs: list[ImagePair] = []
    missing: list[str] = []

    for kind in ("rain", "snow"):
        for idx in range(1, 1601):
            degraded_path = degraded_dir / f"{kind}-{idx}.png"
            clean_path = clean_dir / f"{kind}_clean-{idx}.png"
            if not degraded_path.exists() or not clean_path.exists():
                missing.append(f"{kind}-{idx}")
                continue
            pairs.append(
                ImagePair(
                    name=degraded_path.name,
                    degraded_path=degraded_path,
                    clean_path=clean_path,
                    kind=kind,
                    index=idx,
                )
            )

    if missing:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(
            f"Missing {len(missing)} train pairs. First missing: {preview}"
        )
    return pairs


def split_pairs(
    pairs: Iterable[ImagePair],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[ImagePair], list[ImagePair]]:
    """Make a deterministic stratified train/validation split."""

    rng = random.Random(seed)
    by_kind: dict[str, list[ImagePair]] = {"rain": [], "snow": []}
    for pair in pairs:
        by_kind[pair.kind].append(pair)

    train_pairs: list[ImagePair] = []
    val_pairs: list[ImagePair] = []
    for kind in ("rain", "snow"):
        items = sorted(by_kind[kind], key=lambda item: item.index)
        if not items:
            continue
        rng.shuffle(items)
        val_count = max(1, int(round(len(items) * val_ratio)))
        val_pairs.extend(items[:val_count])
        train_pairs.extend(items[val_count:])

    rng.shuffle(train_pairs)
    val_pairs.sort(key=lambda item: (item.kind, item.index))
    return train_pairs, val_pairs


def load_rgb(path: str | Path) -> np.ndarray:
    """Load an RGB image as uint8 HWC."""

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert uint8 HWC image to float tensor CHW in [0, 1]."""

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape {image.shape}.")
    array = np.ascontiguousarray(image.transpose(2, 0, 1))
    return torch.from_numpy(array).float().div(255.0)


def tensor_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """Convert CHW or BCHW tensor in [0, 1] to uint8 numpy."""

    tensor = tensor.detach().clamp(0.0, 1.0).mul(255.0).round().byte().cpu()
    return tensor.numpy()


def save_chw_image(array: np.ndarray, path: str | Path) -> None:
    """Save a CHW uint8 array to disk."""

    if array.ndim != 3 or array.shape[0] != 3:
        raise ValueError(f"Expected CHW RGB array, got shape {array.shape}.")
    image = Image.fromarray(array.transpose(1, 2, 0), mode="RGB")
    image.save(path)


def paired_random_crop(
    degraded: np.ndarray,
    clean: np.ndarray,
    patch_size: int,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop matching patches from degraded and clean images."""

    height, width = degraded.shape[:2]
    if clean.shape[:2] != (height, width):
        raise ValueError("Degraded and clean images must have matching sizes.")
    if patch_size <= 0:
        return degraded, clean
    if patch_size > height or patch_size > width:
        raise ValueError(
            f"Patch size {patch_size} is larger than image size {height}x{width}."
        )
    top = rng.randint(0, height - patch_size)
    left = rng.randint(0, width - patch_size)
    bottom = top + patch_size
    right = left + patch_size
    return degraded[top:bottom, left:right], clean[top:bottom, left:right]


def paired_augment(
    degraded: np.ndarray,
    clean: np.ndarray,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply identical flip/rotation augmentation to a pair."""

    if rng.random() < 0.5:
        degraded = np.flip(degraded, axis=1)
        clean = np.flip(clean, axis=1)
    if rng.random() < 0.5:
        degraded = np.flip(degraded, axis=0)
        clean = np.flip(clean, axis=0)

    rotations = rng.randint(0, 3)
    if rotations:
        degraded = np.rot90(degraded, rotations, axes=(0, 1))
        clean = np.rot90(clean, rotations, axes=(0, 1))
    return degraded.copy(), clean.copy()


class RestorationDataset(Dataset):
    """Paired rain/snow restoration dataset."""

    def __init__(
        self,
        pairs: list[ImagePair],
        patch_size: int = 128,
        augment: bool = True,
        base_seed: int = 42,
        limit: int | None = None,
    ) -> None:
        self.pairs = pairs[:limit] if limit else pairs
        self.patch_size = patch_size
        self.augment = augment
        self.base_seed = base_seed

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        pair = self.pairs[index]
        degraded = load_rgb(pair.degraded_path)
        clean = load_rgb(pair.clean_path)

        # Use the worker's Python RNG so each epoch sees fresh paired crops.
        # PyTorch seeds this RNG per worker, while validation is deterministic
        # because it disables cropping and augmentation.
        rng = random
        if self.patch_size:
            degraded, clean = paired_random_crop(
                degraded,
                clean,
                self.patch_size,
                rng,
            )
        if self.augment:
            degraded, clean = paired_augment(degraded, clean, rng)

        return {
            "name": pair.name,
            "kind": pair.kind,
            "degraded": image_to_tensor(degraded),
            "clean": image_to_tensor(clean),
        }


class ImageFolderDataset(Dataset):
    """Unpaired image folder dataset used for test prediction."""

    def __init__(self, image_paths: Iterable[str | Path]) -> None:
        self.image_paths = sorted([Path(path) for path in image_paths], key=natural_key)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        path = self.image_paths[index]
        image = load_rgb(path)
        return {
            "name": path.name,
            "image": image_to_tensor(image),
            "height": image.shape[0],
            "width": image.shape[1],
        }


def psnr_from_mse(mse: float) -> float:
    """Compute PSNR for images in [0, 1]."""

    if mse <= 0:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def batch_psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return per-image PSNR for BCHW tensors in [0, 1]."""

    pred = pred.detach().clamp(0.0, 1.0)
    target = target.detach().clamp(0.0, 1.0)
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    return torch.where(
        mse <= 0,
        torch.full_like(mse, 99.0),
        10.0 * torch.log10(1.0 / mse),
    )
