"""Submission and inference helpers."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from torchvision.ops import batched_nms


def merge_tta_predictions(
    predictions: Iterable[Dict[str, torch.Tensor]],
    score_threshold: float,
    iou_threshold: float,
    max_detections_per_image: int,
) -> Dict[str, torch.Tensor]:
    boxes_xyxy = []
    boxes_xywh = []
    labels = []
    scores = []

    for prediction in predictions:
        keep = prediction["scores"] >= score_threshold
        if keep.any():
            boxes_xyxy.append(prediction["boxes_xyxy"][keep])
            boxes_xywh.append(prediction["boxes"][keep])
            labels.append(prediction["labels"][keep])
            scores.append(prediction["scores"][keep])

    if not scores:
        empty_float = torch.empty((0,), dtype=torch.float32)
        empty_boxes = torch.empty((0, 4), dtype=torch.float32)
        empty_labels = torch.empty((0,), dtype=torch.int64)
        return {
            "scores": empty_float,
            "labels": empty_labels,
            "boxes": empty_boxes,
            "boxes_xyxy": empty_boxes,
        }

    boxes_xyxy = torch.cat(boxes_xyxy, dim=0).to(dtype=torch.float32)
    boxes_xywh = torch.cat(boxes_xywh, dim=0).to(dtype=torch.float32)
    labels = torch.cat(labels, dim=0).to(dtype=torch.int64)
    scores = torch.cat(scores, dim=0).to(dtype=torch.float32)

    keep = batched_nms(boxes_xyxy, scores, labels, iou_threshold=iou_threshold)
    keep = keep[:max_detections_per_image]
    return {
        "scores": scores[keep],
        "labels": labels[keep],
        "boxes": boxes_xywh[keep],
        "boxes_xyxy": boxes_xyxy[keep],
    }


def predictions_to_records(image_id: int, prediction: Dict[str, torch.Tensor]) -> List[Dict]:
    records = []
    for score, label, box in zip(
        prediction["scores"].cpu(),
        prediction["labels"].cpu(),
        prediction["boxes"].cpu(),
    ):
        records.append(
            {
                "image_id": image_id,
                "category_id": int(label.item()),
                "bbox": [float(value) for value in box.tolist()],
                "score": float(score.item()),
            }
        )
    return records


def write_submission(predictions: List[Dict], output_json: str | Path, zip_output: bool = False) -> None:
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(predictions, handle)

    if zip_output:
        zip_path = output_json.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(output_json, arcname="pred.json")
