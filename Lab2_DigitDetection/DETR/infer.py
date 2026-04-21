"""Generate COCO-format predictions and an optional competition zip."""

from __future__ import annotations

import argparse
from typing import Dict

import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.coco import COCODetectionDataset
from src.data.transforms import build_eval_transforms
from src.models.detr import build_detr
from src.utils.checkpoint import load_checkpoint
from src.utils.misc import collate_fn
from src.utils.submission import merge_tta_predictions, predictions_to_records, write_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/detr_r50.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image-dir", type=str, default="nycu-hw2-data/test")
    parser.add_argument("--annotation-file", type=str, default=None)
    parser.add_argument("--output-json", type=str, default="pred.json")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--zip-output", action="store_true")
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--tta-scales", nargs="+", type=int, default=None)
    return parser.parse_args()


def clone_target(target: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cloned = {}
    for key, value in target.items():
        cloned[key] = value.clone() if torch.is_tensor(value) else value
    return cloned


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("infer.py currently expects --batch-size 1 for per-image multi-scale TTA.")

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = COCODetectionDataset(
        image_dir=args.image_dir,
        annotation_file=args.annotation_file,
        transforms=None,
        category_ids=config["model"]["category_ids"],
    )
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model, _, postprocessor = build_detr(config)
    model.to(device)
    checkpoint = load_checkpoint(args.checkpoint, model=model, map_location="cpu")
    if args.use_ema and checkpoint.get("ema") is not None:
        model.load_state_dict(checkpoint["ema"], strict=True)
    model.eval()

    score_threshold = (
        float(args.score_threshold)
        if args.score_threshold is not None
        else float(config["inference"]["score_threshold"])
    )
    tta_scales = args.tta_scales if args.tta_scales is not None else list(config["inference"]["tta_scales"])
    tta_transforms = {scale: build_eval_transforms(config, scale=scale) for scale in tta_scales}

    all_predictions = []
    with torch.no_grad():
        for index, (images, targets) in enumerate(data_loader, start=1):
            image = images[0]
            target = clone_target(targets[0])
            image_id = int(target["image_id"].item())
            orig_size = target["orig_size"].unsqueeze(0).to(device)

            scale_predictions = []
            for scale, transform in tta_transforms.items():
                transformed, _ = transform(image, None)
                transformed = transformed.to(device)
                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=bool(config["train"]["amp"]) and device.type == "cuda",
                ):
                    outputs = model([transformed])
                result = postprocessor(outputs, orig_size)[0]
                scale_predictions.append({key: value.cpu() for key, value in result.items()})

            merged = merge_tta_predictions(
                predictions=scale_predictions,
                score_threshold=score_threshold,
                iou_threshold=float(config["inference"]["nms_iou_threshold"]),
                max_detections_per_image=int(config["inference"]["max_detections_per_image"]),
            )
            all_predictions.extend(predictions_to_records(image_id=image_id, prediction=merged))

            if index % 100 == 0:
                print(f"Processed {index}/{len(data_loader)} images")

    write_submission(all_predictions, output_json=args.output_json, zip_output=args.zip_output)
    print(f"Wrote {len(all_predictions)} predictions to {args.output_json}")


if __name__ == "__main__":
    main()
