from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18
from tqdm import tqdm

LABEL_TO_ID = {"snow": 0, "rain": 1}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


@dataclass(frozen=True)
class WeatherExample:
    path: Path
    label: int
    kind: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a rain/snow classifier on HW4 degraded training images."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--train-dir", type=Path, default=None)
    parser.add_argument("--test-dir", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default="weather_classifier_resnet18")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", choices=["off", "fp16", "bf16"], default="bf16")
    parser.add_argument("--model", choices=["resnet18", "tiny"], default="resnet18")
    parser.add_argument(
        "--input-mode",
        choices=["rgb", "residual", "rgb_residual"],
        default="rgb_residual",
        help="rgb_residual appends a positive high-pass artifact channel.",
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def natural_key(path: str | Path) -> list[int | str]:
    import re

    name = Path(path).name
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_run_dir(run_name: str) -> Path:
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config(args: argparse.Namespace, run_dir: Path) -> None:
    config = vars(args).copy()
    for key, value in config.items():
        if isinstance(value, Path):
            config[key] = str(value)
    config["command"] = " ".join(sys.argv)
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        for key in sorted(config):
            handle.write(f"{key}: {config[key]}\n")


def label_from_train_name(path: Path) -> str:
    if path.name.startswith("rain-"):
        return "rain"
    if path.name.startswith("snow-"):
        return "snow"
    return ""


def collect_train_examples(train_dir: Path, limit: int | None = None) -> list[WeatherExample]:
    examples: list[WeatherExample] = []
    for path in sorted(train_dir.glob("*.png"), key=natural_key):
        kind = label_from_train_name(path)
        if not kind:
            continue
        examples.append(WeatherExample(path=path, label=LABEL_TO_ID[kind], kind=kind))

    if not limit or limit >= len(examples):
        return examples

    by_kind = {
        "rain": [example for example in examples if example.kind == "rain"],
        "snow": [example for example in examples if example.kind == "snow"],
    }
    rain_limit = limit // 2 + limit % 2
    snow_limit = limit // 2
    selected = by_kind["rain"][:rain_limit] + by_kind["snow"][:snow_limit]
    return sorted(selected, key=lambda item: natural_key(item.path))


def split_examples(
    examples: list[WeatherExample],
    val_ratio: float,
    seed: int,
) -> tuple[list[WeatherExample], list[WeatherExample]]:
    rng = random.Random(seed)
    train: list[WeatherExample] = []
    val: list[WeatherExample] = []
    for kind in ("rain", "snow"):
        items = [example for example in examples if example.kind == kind]
        rng.shuffle(items)
        val_count = max(1, int(round(len(items) * val_ratio)))
        val.extend(items[:val_count])
        train.extend(items[val_count:])
    rng.shuffle(train)
    val.sort(key=lambda item: (item.kind, natural_key(item.path)))
    return train, val


def collect_test_paths(test_dir: Path, limit: int | None = None) -> list[Path]:
    paths = sorted(test_dir.glob("*.png"), key=natural_key)
    if limit:
        paths = paths[:limit]
    return paths


def make_input_tensor(image: Image.Image, input_mode: str) -> torch.Tensor:
    rgb = transforms.functional.pil_to_tensor(image.convert("RGB")).float().div(255.0)
    gray = (
        0.299 * rgb[0:1]
        + 0.587 * rgb[1:2]
        + 0.114 * rgb[2:3]
    )
    blur = F.avg_pool2d(gray.unsqueeze(0), kernel_size=9, stride=1, padding=4).squeeze(0)
    residual = (gray - blur).clamp_min(0.0)
    residual = (residual * 8.0).clamp(0.0, 1.0)

    if input_mode == "rgb":
        tensor = rgb
    elif input_mode == "residual":
        tensor = residual
    else:
        tensor = torch.cat([rgb, residual], dim=0)
    return tensor.mul(2.0).sub(1.0)


class WeatherDataset(Dataset):
    def __init__(
        self,
        examples: list[WeatherExample],
        image_size: int,
        input_mode: str,
        augment: bool,
    ) -> None:
        self.examples = examples
        self.input_mode = input_mode
        self.augment = augment
        if augment:
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        image_size,
                        scale=(0.70, 1.0),
                        ratio=(0.90, 1.10),
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ColorJitter(
                        brightness=0.10,
                        contrast=0.10,
                        saturation=0.05,
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        (image_size, image_size),
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    )
                ]
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        example = self.examples[index]
        with Image.open(example.path) as image:
            image = self.transform(image.convert("RGB"))
        return {
            "image": make_input_tensor(image, self.input_mode),
            "label": torch.tensor(example.label, dtype=torch.long),
            "kind": example.kind,
            "name": example.path.name,
        }


class WeatherTestDataset(Dataset):
    def __init__(
        self,
        paths: list[Path],
        image_size: int,
        input_mode: str,
    ) -> None:
        self.paths = paths
        self.input_mode = input_mode
        self.transform = transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        path = self.paths[index]
        with Image.open(path) as image:
            image = self.transform(image.convert("RGB"))
        return {
            "image": make_input_tensor(image, self.input_mode),
            "label": torch.tensor(-1, dtype=torch.long),
            "kind": "",
            "name": path.name,
        }


class TinyWeatherCNN(nn.Module):
    def __init__(self, in_channels: int, dropout: float) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._block(in_channels, 32, stride=2),
            self._block(32, 64, stride=2),
            self._block(64, 128, stride=2),
            self._block(128, 256, stride=2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def input_channels(input_mode: str) -> int:
    if input_mode == "rgb":
        return 3
    if input_mode == "residual":
        return 1
    return 4


def build_model(model_name: str, input_mode: str, dropout: float) -> nn.Module:
    channels = input_channels(input_mode)
    if model_name == "tiny":
        return TinyWeatherCNN(channels, dropout)

    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(
        channels,
        64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False,
    )
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, 2))
    return model


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
    with metrics_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def summarize_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[int, int, dict[str, int]]:
    pred = logits.argmax(dim=1)
    valid = labels >= 0
    correct = int((pred[valid] == labels[valid]).sum().item())
    count = int(valid.sum().item())
    stats = {"rain_correct": 0, "rain_total": 0, "snow_correct": 0, "snow_total": 0}
    for label_id, kind in ID_TO_LABEL.items():
        mask = valid & (labels == label_id)
        stats[f"{kind}_total"] = int(mask.sum().item())
        stats[f"{kind}_correct"] = int((pred[mask] == labels[mask]).sum().item())
    return correct, count, stats


def class_accuracy(stats: dict[str, int], kind: str) -> float:
    return stats[f"{kind}_correct"] / max(1, stats[f"{kind}_total"])


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: str,
    desc: str,
) -> dict[str, float]:
    model.eval()
    autocast = get_autocast(device, amp)
    loss_sum = 0.0
    total = 0
    correct = 0
    class_stats = {"rain_correct": 0, "rain_total": 0, "snow_correct": 0, "snow_total": 0}
    loss_fn = nn.CrossEntropyLoss(reduction="sum", ignore_index=-1)

    for batch in tqdm(loader, desc=desc, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with autocast():
            logits = model(images)
            loss = loss_fn(logits, labels)
        batch_correct, batch_total, batch_stats = summarize_logits(logits, labels)
        correct += batch_correct
        total += batch_total
        loss_sum += float(loss.item())
        for key, value in batch_stats.items():
            class_stats[key] += value

    return {
        "loss": loss_sum / max(1, total),
        "accuracy": correct / max(1, total),
        "rain_accuracy": class_accuracy(class_stats, "rain"),
        "snow_accuracy": class_accuracy(class_stats, "snow"),
        "count": float(total),
    }


@torch.no_grad()
def predict_rows(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: str,
) -> list[dict[str, str | float]]:
    model.eval()
    autocast = get_autocast(device, amp)
    rows: list[dict[str, str | float]] = []
    for batch in tqdm(loader, desc="predict", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        with autocast():
            logits = model(images).float()
        probs = logits.softmax(dim=1).cpu().numpy()
        logits_np = logits.cpu().numpy()
        for i, name in enumerate(batch["name"]):
            pred_id = int(probs[i].argmax())
            pred_kind = ID_TO_LABEL[pred_id]
            rows.append(
                {
                    "image": str(name),
                    "pred_kind": pred_kind,
                    "rain_probability": float(probs[i, LABEL_TO_ID["rain"]]),
                    "snow_probability": float(probs[i, LABEL_TO_ID["snow"]]),
                    "logit_rain": float(logits_np[i, LABEL_TO_ID["rain"]]),
                    "logit_snow": float(logits_np[i, LABEL_TO_ID["snow"]]),
                }
            )
    return sorted(rows, key=lambda row: natural_key(str(row["image"])))


def write_predictions(path: Path, rows: list[dict[str, str | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image",
        "pred_kind",
        "rain_probability",
        "snow_probability",
        "logit_rain",
        "logit_snow",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metrics(run_dir: Path) -> None:
    try:
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
    axes[0].plot(data["epoch"], data["train_loss"], label="train")
    axes[0].plot(data["epoch"], data["val_loss"], label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("CE loss")
    axes[0].legend()
    axes[1].plot(data["epoch"], data["train_acc"], label="train")
    axes[1].plot(data["epoch"], data["val_acc"], label="val")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(run_dir / "metrics.png")
    plt.close(fig)


def plot_test_probabilities(run_dir: Path, rows: list[dict[str, str | float]]) -> None:
    all_probs = [float(row["rain_probability"]) for row in rows]
    if not all_probs:
        return
    plt.figure(figsize=(8, 4), dpi=150)
    bins = np.linspace(0.0, 1.0, 41)
    plt.hist(all_probs, bins=bins, alpha=0.7, label="test")
    plt.axvline(0.5, color="black", linestyle="--", linewidth=1.2)
    plt.xlabel("rain probability")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "test_probability_distribution.png")
    plt.close()


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: LambdaLR | None,
    epoch: int,
    best_val_acc: float,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "best_val_acc": best_val_acc,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "scheduler": scheduler.state_dict() if scheduler else None,
            "model_args": {
                "model": args.model,
                "input_mode": args.input_mode,
                "dropout": args.dropout,
                "image_size": args.image_size,
            },
            "args": vars(args),
        },
        path,
    )


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    fallback_args: argparse.Namespace,
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_args = checkpoint.get("model_args", {})
    model_name = model_args.get("model", fallback_args.model)
    input_mode = model_args.get("input_mode", fallback_args.input_mode)
    dropout = float(model_args.get("dropout", fallback_args.dropout))
    model = build_model(model_name, input_mode, dropout)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    return model, model_args


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    device: torch.device,
    amp: str,
    log_every: int,
) -> dict[str, float]:
    model.train()
    autocast = get_autocast(device, amp)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and amp == "fp16"))
    loss_fn = nn.CrossEntropyLoss()
    loss_sum = 0.0
    total = 0
    correct = 0
    for step, batch in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = model(images)
            loss = loss_fn(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        batch_correct, batch_total, _ = summarize_logits(logits.detach(), labels)
        loss_sum += float(loss.item()) * batch_total
        correct += batch_correct
        total += batch_total
        if log_every and step % log_every == 0:
            acc = correct / max(1, total)
            tqdm.write(f"step {step:04d}: loss={loss_sum / max(1, total):.4f} acc={acc:.4f}")

    return {
        "loss": loss_sum / max(1, total),
        "accuracy": correct / max(1, total),
    }


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=workers > 0,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    train_dir = args.train_dir or args.data_root / "train" / "degraded"
    test_dir = args.test_dir or args.data_root / "test" / "degraded"
    run_dir = make_run_dir(args.run_name)
    save_config(args, run_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_paths = collect_test_paths(test_dir, args.limit_test)

    if args.eval_only:
        if args.checkpoint is None:
            raise ValueError("--eval-only requires --checkpoint.")
        model, model_args = load_model_from_checkpoint(args.checkpoint, device, args)
        input_mode = str(model_args.get("input_mode", args.input_mode))
        image_size = int(model_args.get("image_size", args.image_size))
        test_dataset = WeatherTestDataset(test_paths, image_size, input_mode)
        test_loader = make_loader(test_dataset, args.batch_size, False, args.num_workers)
        rows = predict_rows(model, test_loader, device, args.amp)
        write_predictions(run_dir / "test_weather_classifier_predictions.csv", rows)
        plot_test_probabilities(run_dir, rows)
        print(f"wrote {len(rows)} predictions to {run_dir / 'test_weather_classifier_predictions.csv'}")
        return

    examples = collect_train_examples(train_dir, args.limit_train)
    train_examples, val_examples = split_examples(examples, args.val_ratio, args.seed)
    train_dataset = WeatherDataset(
        train_examples,
        args.image_size,
        args.input_mode,
        augment=True,
    )
    val_dataset = WeatherDataset(
        val_examples,
        args.image_size,
        args.input_mode,
        augment=False,
    )
    test_dataset = WeatherTestDataset(test_paths, args.image_size, args.input_mode)
    train_loader = make_loader(train_dataset, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_dataset, args.batch_size, False, args.num_workers)
    test_loader = make_loader(test_dataset, args.batch_size, False, args.num_workers)

    model = build_model(args.model, args.input_mode, args.dropout).to(device)
    if args.compile:
        model = torch.compile(model)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(
        optimizer,
        max(1, len(train_loader)),
        args.epochs,
        args.warmup_epochs,
    )

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        writer = None
    else:
        writer = SummaryWriter(log_dir=str(run_dir / "tb"))

    best_val_acc = -1.0
    print(
        f"Training {args.model} on {len(train_examples)} images, "
        f"validating on {len(val_examples)} images, device={device}."
    )
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            args.amp,
            args.log_every,
        )
        val_metrics = evaluate(model, val_loader, device, args.amp, "val")
        lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_rain_acc": val_metrics["rain_accuracy"],
            "val_snow_acc": val_metrics["snow_accuracy"],
        }
        append_metrics(run_dir, row)
        if writer:
            for key, value in row.items():
                if key != "epoch":
                    writer.add_scalar(key, value, epoch)
        print(
            f"epoch {epoch:03d}: "
            f"train_acc={row['train_acc']:.4f} val_acc={row['val_acc']:.4f} "
            f"lr={lr:.2e}"
        )

        improved = val_metrics["accuracy"] > best_val_acc
        if improved:
            best_val_acc = val_metrics["accuracy"]
        save_checkpoint(run_dir / "last.pt", model, optimizer, scheduler, epoch, best_val_acc, args)
        if improved:
            save_checkpoint(
                run_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_acc,
                args,
            )
    if writer:
        writer.close()

    best_model, _ = load_model_from_checkpoint(run_dir / "best.pt", device, args)
    rows = predict_rows(best_model, test_loader, device, args.amp)
    write_predictions(run_dir / "test_weather_classifier_predictions.csv", rows)
    plot_metrics(run_dir)
    plot_test_probabilities(run_dir, rows)
    summary = {
        "best_val_acc": best_val_acc,
        "run_dir": str(run_dir),
        "test_prediction_csv": str(run_dir / "test_weather_classifier_predictions.csv"),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
