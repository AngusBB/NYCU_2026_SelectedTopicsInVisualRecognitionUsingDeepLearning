from __future__ import annotations

import argparse
import csv
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import ImageFolderDataset, natural_key, save_chw_image, tensor_to_uint8
from inference_utils import load_promptir_checkpoint, restore_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Route test images to rain/snow specialist PromptIR checkpoints using "
            "a weather-classifier CSV, then write a single pred.npz submission."
        )
    )
    parser.add_argument("--rain-checkpoint", type=Path, required=True)
    parser.add_argument("--snow-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--classifier-csv",
        type=Path,
        default=Path(
            "runs/weather_classifier_resnet18_rgbres_ep20/"
            "test_weather_classifier_predictions.csv"
        ),
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/test/degraded"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/promptir_specialists_routed"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--amp", choices=["off", "fp16", "bf16"], default="bf16")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument(
        "--rain-threshold",
        type=float,
        default=0.5,
        help="Images with rain_probability >= threshold use the rain specialist.",
    )
    parser.add_argument(
        "--soft-blend",
        action="store_true",
        help=(
            "Run both specialists on every image and blend by classifier probability. "
            "Default is hard routing, which matches the requested corresponding type."
        ),
    )
    parser.add_argument("--save-images", action="store_true", default=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate classifier routing and write routing.csv; do not load models.",
    )
    return parser.parse_args()


def read_classifier_rows(path: Path, rain_threshold: float) -> dict[str, dict[str, str | float]]:
    if not path.exists():
        raise FileNotFoundError(f"Classifier CSV not found: {path}")
    rows: dict[str, dict[str, str | float]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "image" not in reader.fieldnames or "rain_probability" not in reader.fieldnames:
            raise ValueError(
                f"{path} must contain at least image and rain_probability columns."
            )
        for row in reader:
            image = Path(row["image"]).name
            rain_probability = float(row["rain_probability"])
            pred_kind = row.get("pred_kind") or (
                "rain" if rain_probability >= rain_threshold else "snow"
            )
            rows[image] = {
                "image": image,
                "label": row.get("label", ""),
                "classifier_pred_kind": pred_kind,
                "rain_probability": rain_probability,
                "route_kind": "rain" if rain_probability >= rain_threshold else "snow",
            }
    return rows


def image_paths_for_kind(
    all_paths: list[Path],
    classifier_rows: dict[str, dict[str, str | float]],
    kind: str,
) -> list[Path]:
    return [
        path
        for path in all_paths
        if classifier_rows[path.name]["route_kind"] == kind
    ]


@torch.no_grad()
def predict_subset(
    model: torch.nn.Module,
    image_paths: list[Path],
    batch_size: int,
    num_workers: int,
    amp: str,
    tta: bool,
) -> dict[str, torch.Tensor]:
    if not image_paths:
        return {}
    dataset = ImageFolderDataset(image_paths)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    predictions: dict[str, torch.Tensor] = {}
    for batch in tqdm(loader, desc="predict", leave=False):
        restored = restore_batch(model, batch["image"], amp=amp, tta=tta).cpu()
        for idx, name in enumerate(batch["name"]):
            predictions[str(name)] = restored[idx]
    return predictions


@torch.no_grad()
def predict_soft_blend(
    rain_model: torch.nn.Module,
    snow_model: torch.nn.Module,
    image_paths: list[Path],
    classifier_rows: dict[str, dict[str, str | float]],
    batch_size: int,
    num_workers: int,
    amp: str,
    tta: bool,
) -> dict[str, torch.Tensor]:
    dataset = ImageFolderDataset(image_paths)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    predictions: dict[str, torch.Tensor] = {}
    for batch in tqdm(loader, desc="blend-predict"):
        rain_restored = restore_batch(rain_model, batch["image"], amp=amp, tta=tta).cpu()
        snow_restored = restore_batch(snow_model, batch["image"], amp=amp, tta=tta).cpu()
        for idx, name in enumerate(batch["name"]):
            key = str(name)
            weight = float(classifier_rows[key]["rain_probability"])
            predictions[key] = rain_restored[idx] * weight + snow_restored[idx] * (1.0 - weight)
    return predictions


def write_routing_csv(
    path: Path,
    image_paths: list[Path],
    classifier_rows: dict[str, dict[str, str | float]],
    soft_blend: bool,
) -> None:
    fieldnames = [
        "image",
        "label",
        "classifier_pred_kind",
        "rain_probability",
        "route_kind",
        "soft_blend",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for image_path in image_paths:
            row = classifier_rows[image_path.name]
            writer.writerow({**row, "soft_blend": str(soft_blend)})


def write_submission(
    output_dir: Path,
    image_paths: list[Path],
    predictions: dict[str, torch.Tensor],
    save_images: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_predictions: dict[str, np.ndarray] = {}
    for path in image_paths:
        if path.name not in predictions:
            raise KeyError(f"Missing routed prediction for {path.name}")
        array = tensor_to_uint8(predictions[path.name])
        npz_predictions[path.name] = array
        if save_images:
            save_chw_image(array, output_dir / path.name)

    npz_path = output_dir / "pred.npz"
    np.savez(npz_path, **npz_predictions)
    zip_path = output_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(npz_path, arcname="pred.npz")
    print(f"wrote {len(npz_predictions)} predictions to {npz_path}")
    print(f"wrote submission zip to {zip_path}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(args.input_dir.glob("*.png"), key=natural_key)
    if args.limit:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise FileNotFoundError(f"No PNG files found in {args.input_dir}.")

    classifier_rows = read_classifier_rows(args.classifier_csv, args.rain_threshold)
    missing = [path.name for path in image_paths if path.name not in classifier_rows]
    if missing:
        raise ValueError(
            f"Classifier CSV is missing {len(missing)} test images. "
            f"First missing: {', '.join(missing[:10])}"
        )

    rain_paths = image_paths_for_kind(image_paths, classifier_rows, "rain")
    snow_paths = image_paths_for_kind(image_paths, classifier_rows, "snow")
    print(f"routing: rain={len(rain_paths)} snow={len(snow_paths)}")
    write_routing_csv(
        args.output_dir / "routing.csv",
        image_paths,
        classifier_rows,
        args.soft_blend,
    )
    if args.dry_run:
        print(f"dry run wrote routing CSV to {args.output_dir / 'routing.csv'}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rain_model, _ = load_promptir_checkpoint(args.rain_checkpoint, device)
    snow_model, _ = load_promptir_checkpoint(args.snow_checkpoint, device)

    if args.soft_blend:
        predictions = predict_soft_blend(
            rain_model,
            snow_model,
            image_paths,
            classifier_rows,
            args.batch_size,
            args.num_workers,
            args.amp,
            args.tta,
        )
    else:
        predictions = {}
        predictions.update(
            predict_subset(
                rain_model,
                rain_paths,
                args.batch_size,
                args.num_workers,
                args.amp,
                args.tta,
            )
        )
        predictions.update(
            predict_subset(
                snow_model,
                snow_paths,
                args.batch_size,
                args.num_workers,
                args.amp,
                args.tta,
            )
        )

    write_submission(args.output_dir, image_paths, predictions, args.save_images)


if __name__ == "__main__":
    main()
