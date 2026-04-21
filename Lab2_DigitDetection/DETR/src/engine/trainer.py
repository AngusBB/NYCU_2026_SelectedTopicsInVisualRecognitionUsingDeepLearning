"""Training loop utilities."""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import nn

from src.utils.misc import MetricLogger, move_targets_to_device, reduce_dict


def build_optimizer(model: nn.Module, config: Dict) -> torch.optim.Optimizer:
    train_cfg = config["train"]
    backbone_params = []
    other_params = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(parameter)
        else:
            other_params.append(parameter)

    return torch.optim.AdamW(
        [
            {"params": other_params, "lr": float(train_cfg["lr"])},
            {"params": backbone_params, "lr": float(train_cfg["lr_backbone"])},
        ],
        weight_decay=float(train_cfg["weight_decay"]),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
    config: Dict,
) -> torch.optim.lr_scheduler.LambdaLR:
    train_cfg = config["train"]
    total_steps = max(1, int(train_cfg["epochs"]) * steps_per_epoch)
    warmup_steps = int(train_cfg["warmup_epochs"]) * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    data_loader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    epoch: int,
    scaler: torch.cuda.amp.GradScaler,
    config: Dict,
    ema=None,
    preview_callback: Optional[Callable[[List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]], int], None]] = None,
    preview_images: int = 0,
) -> Dict[str, float]:
    model.train()
    criterion.train()

    train_cfg = config["train"]
    metric_logger = MetricLogger(delimiter="  ")
    header = f"Epoch [{epoch:03d}] "
    grad_accum_steps = max(1, int(train_cfg["grad_accum_steps"]))
    print_freq = int(train_cfg["print_freq"])
    clip_max_norm = float(train_cfg["clip_max_norm"])
    use_amp = bool(train_cfg["amp"]) and device.type == "cuda"
    preview_samples: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]] = []

    optimizer.zero_grad(set_to_none=True)

    for step, (images, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if preview_callback is not None and len(preview_samples) < preview_images:
            remaining = preview_images - len(preview_samples)
            for image, target in zip(images[:remaining], targets[:remaining]):
                preview_samples.append(
                    (
                        image.detach().cpu().clone(),
                        {
                            key: value.detach().cpu().clone() if torch.is_tensor(value) else value
                            for key, value in target.items()
                        },
                    )
                )

        images = [image.to(device) for image in images]
        targets = move_targets_to_device(targets, device)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[key] * weight_dict[key] for key in loss_dict.keys() if key in weight_dict)

        if not math.isfinite(losses.item()):
            raise RuntimeError(f"Non-finite loss encountered: {losses.item()}")

        scaled_loss = losses / grad_accum_steps
        scaler.scale(scaled_loss).backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(data_loader):
            scaler.unscale_(optimizer)
            if clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_ran = scaler.get_scale() >= scale_before
            optimizer.zero_grad(set_to_none=True)
            if optimizer_ran:
                scheduler.step()
            if ema is not None:
                model_for_ema = model.module if hasattr(model, "module") else model
                ema.update(model_for_ema)

        reduced_losses = reduce_dict(loss_dict)
        scaled_reduced = sum(
            reduced_losses[key] * weight_dict[key]
            for key in reduced_losses.keys()
            if key in weight_dict
        )
        metric_logger.update(
            loss=scaled_reduced,
            class_error=reduced_losses.get("class_error", torch.tensor(0.0, device=device)),
            lr=optimizer.param_groups[0]["lr"],
        )
        metric_logger.update(
            **{
                key: value
                for key, value in reduced_losses.items()
                if key.startswith("loss_")
            }
        )

    metric_logger.synchronize_between_processes()
    if preview_callback is not None and preview_samples:
        preview_callback(preview_samples, epoch)
    return {name: meter.global_avg for name, meter in metric_logger.meters.items()}
