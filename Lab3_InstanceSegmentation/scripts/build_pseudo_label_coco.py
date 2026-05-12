#!/usr/bin/env python3
"""Build a COCO train file from train-all labels plus pseudo-labeled test data."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from prediction_utils import (bbox_from_mask, load_predictions, normalize_rle,
                              segmentation_to_mask)

try:
    from pycocotools import mask as mask_utils
except ImportError:
    from mmpycocotools import mask as mask_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--submission",
        type=Path,
        default=Path(
            "sweep_outputs/ensembles/ensembles_37-1p0_34-1p0_nms0p60.zip"),
        help="Teacher prediction JSON/ZIP in submission format.")
    parser.add_argument(
        "--train-ann",
        type=Path,
        default=Path("data/processed/annotations/instances_train_all.json"))
    parser.add_argument(
        "--test-images-ann",
        type=Path,
        default=Path("data/processed/annotations/test_images.json"))
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."))
    parser.add_argument(
        "--train-image-prefix",
        default="data/processed/images/train_all")
    parser.add_argument(
        "--test-image-prefix",
        default="data/processed/images/test")
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.20,
        help="Minimum teacher score kept as a pseudo label.")
    parser.add_argument(
        "--max-per-img",
        type=int,
        default=1000,
        help="Keep at most this many pseudo labels per image after score sorting.")
    parser.add_argument(
        "--min-area",
        type=float,
        default=4.0,
        help="Drop pseudo masks with area below this value.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed/annotations/"
            "instances_train_all_plus_pseudo_ens37_34_thr0p20.json"))
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def prefixed_file_name(prefix: str, file_name: str) -> str:
    return str(Path(prefix) / file_name)


def add_true_annotations(
    combined: dict[str, Any],
    train_payload: dict[str, Any],
    repo_root: Path,
    image_prefix: str,
) -> tuple[set[int], int]:
    used_image_ids = set()
    next_ann_id = 1

    for image in train_payload["images"]:
        item = deepcopy(image)
        item["file_name"] = prefixed_file_name(image_prefix, item["file_name"])
        image_path = repo_root / item["file_name"]
        if not image_path.exists():
            raise FileNotFoundError(f"Missing train image: {image_path}")
        combined["images"].append(item)
        used_image_ids.add(int(item["id"]))

    for annotation in train_payload["annotations"]:
        item = deepcopy(annotation)
        item["id"] = next_ann_id
        item["is_pseudo"] = 0
        combined["annotations"].append(item)
        next_ann_id += 1

    return used_image_ids, next_ann_id


def add_test_images(
    combined: dict[str, Any],
    test_payload: dict[str, Any],
    repo_root: Path,
    image_prefix: str,
    used_image_ids: set[int],
) -> dict[int, int]:
    next_image_id = max(used_image_ids, default=0) + 1
    image_id_map = {}

    for image in test_payload["images"]:
        item = deepcopy(image)
        old_image_id = int(item["id"])
        new_image_id = old_image_id
        if new_image_id in used_image_ids:
            while next_image_id in used_image_ids:
                next_image_id += 1
            new_image_id = next_image_id

        item["id"] = new_image_id
        item["file_name"] = prefixed_file_name(image_prefix, item["file_name"])
        image_path = repo_root / item["file_name"]
        if not image_path.exists():
            raise FileNotFoundError(f"Missing test image: {image_path}")

        combined["images"].append(item)
        used_image_ids.add(new_image_id)
        image_id_map[old_image_id] = new_image_id

    return image_id_map


def selected_predictions(predictions: list[dict], score_thr: float,
                         max_per_img: int) -> list[dict]:
    grouped: dict[int, list[dict]] = {}
    for prediction in predictions:
        if float(prediction.get("score", 0.0)) < score_thr:
            continue
        image_id = int(prediction["image_id"])
        grouped.setdefault(image_id, []).append(prediction)

    selected = []
    for image_id in sorted(grouped):
        image_predictions = sorted(
            grouped[image_id],
            key=lambda item: float(item.get("score", 0.0)),
            reverse=True)
        selected.extend(image_predictions[:max_per_img])
    return selected


def add_pseudo_annotations(
    combined: dict[str, Any],
    predictions: list[dict],
    image_id_map: dict[int, int],
    next_ann_id: int,
    min_area: float,
) -> tuple[int, int]:
    added = 0
    for prediction in predictions:
        old_image_id = int(prediction["image_id"])
        if old_image_id not in image_id_map:
            raise KeyError(f"Prediction image_id {old_image_id} is not in test images.")

        mask = segmentation_to_mask(prediction["segmentation"])
        area = float(mask_utils.area(normalize_rle(mask)))
        if area < min_area:
            continue
        bbox = bbox_from_mask(mask)
        if bbox is None:
            continue

        combined["annotations"].append({
            "id": next_ann_id,
            "image_id": image_id_map[old_image_id],
            "category_id": int(prediction["category_id"]),
            "bbox": bbox,
            "area": area,
            "segmentation": normalize_rle(mask),
            "iscrowd": 0,
            "score": float(prediction.get("score", 0.0)),
            "is_pseudo": 1,
        })
        next_ann_id += 1
        added += 1

    return next_ann_id, added


def main() -> None:
    args = parse_args()
    train_payload = load_json(args.train_ann)
    test_payload = load_json(args.test_images_ann)

    if train_payload["categories"] != test_payload["categories"]:
        raise ValueError("Train and test category definitions differ.")

    combined = {
        "info": {
            "description": "Train-all labels plus teacher pseudo labels on test images",
            "pseudo_source": str(args.submission),
            "pseudo_score_thr": args.score_thr,
            "pseudo_max_per_img": args.max_per_img,
        },
        "licenses": train_payload.get("licenses", []),
        "categories": train_payload["categories"],
        "images": [],
        "annotations": [],
    }

    used_image_ids, next_ann_id = add_true_annotations(
        combined, train_payload, args.repo_root, args.train_image_prefix)
    image_id_map = add_test_images(
        combined, test_payload, args.repo_root, args.test_image_prefix,
        used_image_ids)

    predictions = selected_predictions(
        load_predictions(args.submission), args.score_thr, args.max_per_img)
    _, pseudo_count = add_pseudo_annotations(
        combined, predictions, image_id_map, next_ann_id, args.min_area)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(combined, handle)

    print(f"Wrote {args.output}")
    print(f"images: {len(combined['images'])}")
    print(f"true annotations: {len(train_payload['annotations'])}")
    print(f"pseudo annotations: {pseudo_count}")
    print(f"total annotations: {len(combined['annotations'])}")
    print(f"score_thr: {args.score_thr}")


if __name__ == "__main__":
    main()
