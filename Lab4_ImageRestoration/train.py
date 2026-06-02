"""Train PromptIR on the HW4 rain/snow restoration dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from data import (
    RestorationDataset,
    batch_psnr,
    collect_pairs,
    save_chw_image,
    split_pairs,
    tensor_to_uint8,
)
from models import PromptIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--run-name", type=str, default="promptir_baseline")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--kind-filter",
        choices=["all", "rain", "snow"],
        default="all",
        help="Train/validate on all pairs or only one degradation kind.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", choices=["off", "fp16", "bf16"], default="bf16")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--save-samples", type=int, default=8)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Initialize model weights from a checkpoint without loading optimizer state.",
    )
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def setup_distributed() -> tuple[bool, int, int, int]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return True, rank, world_size, local_rank


def cleanup_distributed(is_distributed: bool) -> None:
    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_run_dir(run_name: str) -> Path:
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config(args: argparse.Namespace, run_dir: Path) -> None:
    config = vars(args).copy()
    for key, value in config.items():
        if isinstance(value, Path):
            config[key] = str(value)
    config["command"] = " ".join(sys.argv)
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as file:
        for key in sorted(config):
            file.write(f"{key}: {config[key]}\n")
    with (run_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)


def get_autocast(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "off":
        return nullcontext
    dtype = torch.float16 if amp == "fp16" else torch.bfloat16
    return lambda: torch.autocast(device_type="cuda", dtype=dtype)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
    epochs: int,
    warmup_epochs: int,
) -> LambdaLR:
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = max(1, steps_per_epoch * warmup_epochs)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def append_metrics(run_dir: Path, row: dict[str, Any]) -> None:
    metrics_path = run_dir / "metrics.csv"
    write_header = not metrics_path.exists()
    with metrics_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def maybe_plot_metrics(run_dir: Path) -> None:
    """Write a compact training curve for the report if plotting deps exist."""

    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        return

    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        return
    data = pd.read_csv(metrics_path)
    if data.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    axes[0].plot(data["epoch"], data["train_loss"], label="train L1")
    axes[0].plot(data["epoch"], data["val_loss"], label="val L1")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(data["epoch"], data["val_psnr"], label="overall")
    axes[1].plot(data["epoch"], data["val_psnr_rain"], label="rain")
    axes[1].plot(data["epoch"], data["val_psnr_snow"], label="snow")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(run_dir / "metrics.png")
    plt.close(fig)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: str,
) -> dict[str, float]:
    model.eval()
    autocast = get_autocast(device, amp)
    total_loss = 0.0
    total_count = 0
    psnr_values: list[float] = []
    psnr_by_kind: dict[str, list[float]] = {"rain": [], "snow": []}
    loss_fn = nn.L1Loss(reduction="sum")

    for batch in tqdm(loader, desc="validate", leave=False):
        degraded = batch["degraded"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)
        with autocast():
            restored = model(degraded)
            loss = loss_fn(restored, clean)

        restored_for_metrics = restored.clamp(0.0, 1.0)
        values = batch_psnr(restored_for_metrics, clean).cpu().tolist()
        total_loss += loss.item()
        total_count += clean.numel()
        psnr_values.extend(values)
        for kind, value in zip(batch["kind"], values):
            psnr_by_kind.setdefault(str(kind), []).append(float(value))

    def average(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    return {
        "val_loss": total_loss / max(1, total_count),
        "val_psnr": average(psnr_values),
        "val_psnr_rain": average(psnr_by_kind["rain"]),
        "val_psnr_snow": average(psnr_by_kind["snow"]),
    }


@torch.no_grad()
def save_validation_samples(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: str,
    output_dir: Path,
    max_images: int,
) -> None:
    if max_images <= 0:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    autocast = get_autocast(device, amp)
    saved = 0
    for batch in loader:
        degraded = batch["degraded"].to(device, non_blocking=True)
        with autocast():
            restored = model(degraded).clamp(0.0, 1.0)
        restored_uint8 = tensor_to_uint8(restored)
        for idx, name in enumerate(batch["name"]):
            save_chw_image(restored_uint8[idx], output_dir / f"restored_{name}")
            saved += 1
            if saved >= max_images:
                return


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    epoch: int,
    best_psnr: float,
    args: argparse.Namespace,
) -> None:
    model_to_save = model.module if hasattr(model, "module") else model
    model_for_args = getattr(model_to_save, "_orig_mod", model_to_save)
    torch.save(
        {
            "epoch": epoch,
            "best_psnr": best_psnr,
            "state_dict": model_to_save.state_dict(),
            "model_args": model_for_args.model_args,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    device: torch.device,
) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return int(checkpoint["epoch"]) + 1, float(checkpoint.get("best_psnr", 0.0))


def load_model_weights(path: Path, model: nn.Module, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint["state_dict"]
    cleaned_state = {
        key.removeprefix("module.").removeprefix("_orig_mod."): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(cleaned_state, strict=True)


def main() -> None:
    args = parse_args()
    if args.resume is not None and args.init_checkpoint is not None:
        raise ValueError("--resume and --init-checkpoint cannot be used together.")

    is_distributed, rank, world_size, local_rank = setup_distributed()
    seed_everything(args.seed + rank)

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    run_dir = make_run_dir(args.run_name) if is_main_process(rank) else None
    if is_main_process(rank) and run_dir is not None:
        save_config(args, run_dir)

    pairs = collect_pairs(args.data_root)
    if args.kind_filter != "all":
        pairs = [pair for pair in pairs if pair.kind == args.kind_filter]
    if not pairs:
        raise ValueError(f"No training pairs found for kind_filter={args.kind_filter}.")
    train_pairs, val_pairs = split_pairs(pairs, args.val_ratio, args.seed)

    train_dataset = RestorationDataset(
        train_pairs,
        patch_size=args.patch_size,
        augment=True,
        base_seed=args.seed,
        limit=args.limit_train,
    )
    val_dataset = RestorationDataset(
        val_pairs,
        patch_size=0,
        augment=False,
        base_seed=args.seed,
        limit=args.limit_val,
    )

    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )
        if is_distributed
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, min(args.batch_size, 8)),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = PromptIR(decoder=True).to(device)
    if args.compile:
        model = torch.compile(model)
    if is_distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = build_scheduler(
        optimizer,
        steps_per_epoch=max(1, len(train_loader)),
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
    )
    start_epoch = 0
    best_psnr = 0.0
    raw_model_for_loading = model.module if hasattr(model, "module") else model
    if args.init_checkpoint is not None:
        load_model_weights(args.init_checkpoint, raw_model_for_loading, device)
    if args.resume is not None:
        start_epoch, best_psnr = load_checkpoint(
            args.resume,
            raw_model_for_loading,
            optimizer,
            scheduler,
            device,
        )

    summary_writer = None
    if is_main_process(rank) and run_dir is not None:
        try:
            from torch.utils.tensorboard import SummaryWriter

            summary_writer = SummaryWriter(log_dir=run_dir / "tensorboard")
        except ImportError:
            summary_writer = None

    loss_fn = nn.L1Loss()
    autocast = get_autocast(device, args.amp)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and args.amp == "fp16",
    )
    global_step = start_epoch * max(1, len(train_loader))

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        progress = tqdm(
            train_loader,
            desc=f"epoch {epoch + 1}/{args.epochs}",
            disable=not is_main_process(rank),
        )
        for step, batch in enumerate(progress, start=1):
            degraded = batch["degraded"].to(device, non_blocking=True)
            clean = batch["clean"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast():
                restored = model(degraded)
                loss = loss_fn(restored, clean)

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            global_step += 1
            if is_main_process(rank) and summary_writer is not None:
                summary_writer.add_scalar("train/loss", loss.item(), global_step)
                summary_writer.add_scalar(
                    "train/lr",
                    scheduler.get_last_lr()[0],
                    global_step,
                )
            if step % args.log_every == 0:
                progress.set_postfix(
                    loss=f"{running_loss / step:.5f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

        if is_distributed:
            dist.barrier()

        if is_main_process(rank) and run_dir is not None:
            raw_model = model.module if hasattr(model, "module") else model
            val_metrics = validate(raw_model, val_loader, device, args.amp)
            train_loss = running_loss / max(1, len(train_loader))
            row = {
                "epoch": epoch + 1,
                "step": global_step,
                "train_loss": train_loss,
                "lr": scheduler.get_last_lr()[0],
                **val_metrics,
            }
            append_metrics(run_dir, row)
            maybe_plot_metrics(run_dir)
            if summary_writer is not None:
                summary_writer.add_scalar("epoch/train_loss", train_loss, epoch + 1)
                for key, value in val_metrics.items():
                    summary_writer.add_scalar(f"val/{key}", value, epoch + 1)

            save_checkpoint(
                run_dir / "last.pt",
                raw_model,
                optimizer,
                scheduler,
                epoch,
                best_psnr,
                args,
            )
            if val_metrics["val_psnr"] > best_psnr:
                best_psnr = val_metrics["val_psnr"]
                save_checkpoint(
                    run_dir / "best.pt",
                    raw_model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_psnr,
                    args,
                )
                save_validation_samples(
                    raw_model,
                    val_loader,
                    device,
                    args.amp,
                    run_dir / "samples" / f"epoch_{epoch + 1:03d}",
                    args.save_samples,
                )
            print(
                f"epoch={epoch + 1} train_loss={train_loss:.5f} "
                f"val_psnr={val_metrics['val_psnr']:.3f} "
                f"best={best_psnr:.3f}"
            )

        if is_distributed:
            dist.barrier()

    if summary_writer is not None:
        summary_writer.close()
    cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()
