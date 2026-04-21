"""Evaluate a checkpoint on a COCO-format dataset."""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.coco import COCODetectionDataset
from src.data.transforms import build_eval_transforms
from src.engine.evaluator import evaluate_coco
from src.models.detr import build_detr
from src.utils.checkpoint import load_checkpoint
from src.utils.misc import collate_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/detr_r50.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image-dir", type=str, default="nycu-hw2-data/valid")
    parser.add_argument("--annotation-file", type=str, default="nycu-hw2-data/valid.json")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--use-ema", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = COCODetectionDataset(
        image_dir=args.image_dir,
        annotation_file=args.annotation_file,
        transforms=build_eval_transforms(config),
        category_ids=config["model"]["category_ids"],
    )
    data_loader = DataLoader(
        dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model, criterion, postprocessor = build_detr(config)
    model.to(device)
    criterion.to(device)
    checkpoint = load_checkpoint(args.checkpoint, model=model, map_location="cpu")
    if args.use_ema and checkpoint.get("ema") is not None:
        model.load_state_dict(checkpoint["ema"], strict=True)

    stats = evaluate_coco(
        model=model,
        criterion=criterion,
        postprocessor=postprocessor,
        data_loader=data_loader,
        device=device,
        score_threshold=float(config["inference"]["score_threshold"]),
        max_detections_per_image=int(config["inference"]["max_detections_per_image"]),
        use_amp=bool(config["train"]["amp"]),
    )
    print(stats)


if __name__ == "__main__":
    main()
