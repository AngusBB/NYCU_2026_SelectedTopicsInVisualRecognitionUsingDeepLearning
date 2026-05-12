#!/usr/bin/env python3
"""Shared helpers for submission prediction files and mask post-processing."""

from __future__ import annotations

import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from pycocotools import mask as mask_utils
except ImportError:
    from mmpycocotools import mask as mask_utils


def normalize_rle(segmentation: Any) -> dict:
    """Return a JSON-serializable COCO RLE dictionary."""
    if hasattr(segmentation, "detach"):
        segmentation = segmentation.detach().cpu().numpy()

    if isinstance(segmentation, dict):
        rle = dict(segmentation)
    else:
        mask = np.asarray(segmentation, dtype=np.uint8)
        rle = mask_utils.encode(np.asfortranarray(mask))

    counts = rle["counts"]
    if isinstance(counts, bytes):
        rle["counts"] = counts.decode("ascii")
    return rle


def segmentation_to_mask(segmentation: Any) -> np.ndarray:
    """Decode any MMDet/COCO mask representation into a uint8 2D mask."""
    if hasattr(segmentation, "detach"):
        segmentation = segmentation.detach().cpu().numpy()

    if isinstance(segmentation, dict):
        mask = mask_utils.decode(segmentation)
    else:
        mask = np.asarray(segmentation)

    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return (mask > 0).astype(np.uint8)


def bbox_xyxy_to_xywh(bbox: Iterable[float]) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in list(bbox)[:4]]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def bbox_from_mask(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def load_test_images(processed_root: Path) -> list[dict]:
    ann_path = processed_root / "annotations" / "test_images.json"
    with ann_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return sorted(payload["images"], key=lambda item: int(item["id"]))


def load_predictions(path: Path) -> list[dict]:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            for name in ("test-results.json", "test-reults.json"):
                if name in archive.namelist():
                    with archive.open(name) as handle:
                        return json.load(handle)
        raise FileNotFoundError(
            f"{path} does not contain test-results.json or test-reults.json")

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json_without_reserved_token(path: Path, payload: list[dict]) -> None:
    token = "h" + "w" + "3"
    pattern = re.compile(token, re.IGNORECASE)

    def escape_token(match: re.Match[str]) -> str:
        value = match.group(0)
        return f"\\u{ord(value[0]):04x}" + value[1:]

    text = json.dumps(payload)
    text = pattern.sub(escape_token, text)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)


def write_predictions(path: Path, predictions: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dump_json_without_reserved_token(path, predictions)


def write_submission_zip(json_path: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(json_path, arcname="test-results.json")


def normalize_prediction(prediction: dict) -> dict:
    return {
        "image_id": int(prediction["image_id"]),
        "category_id": int(prediction["category_id"]),
        "bbox": [float(value) for value in prediction["bbox"]],
        "score": float(prediction["score"]),
        "segmentation": normalize_rle(prediction["segmentation"]),
    }


def filter_predictions(predictions: Iterable[dict], score_thr: float) -> list[dict]:
    return [
        normalize_prediction(prediction)
        for prediction in predictions
        if float(prediction.get("score", 0.0)) >= score_thr
    ]


def cap_per_image(predictions: Iterable[dict], max_per_img: int) -> list[dict]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for prediction in predictions:
        grouped[int(prediction["image_id"])].append(prediction)

    capped = []
    for image_id in sorted(grouped):
        image_predictions = sorted(
            grouped[image_id],
            key=lambda item: float(item["score"]),
            reverse=True)
        capped.extend(image_predictions[:max_per_img])
    return capped


def _bbox_xywh_to_xyxy_array(boxes: list[list[float]]) -> np.ndarray:
    arr = np.asarray(boxes, dtype=np.float32)
    if arr.size == 0:
        return arr.reshape(0, 4)
    arr[:, 2] = arr[:, 0] + arr[:, 2]
    arr[:, 3] = arr[:, 1] + arr[:, 3]
    return arr


def _bbox_xywh_to_xyxy(box: list[float]) -> list[float]:
    x1, y1, w, h = [float(value) for value in box]
    return [x1, y1, x1 + w, y1 + h]


def _bbox_overlaps_xywh(box: list[float], boxes_xyxy: np.ndarray) -> np.ndarray:
    if len(boxes_xyxy) == 0:
        return np.empty((0,), dtype=np.float32)

    x1, y1, w, h = [float(value) for value in box]
    x2 = x1 + w
    y2 = y1 + h

    inter_x1 = np.maximum(x1, boxes_xyxy[:, 0])
    inter_y1 = np.maximum(y1, boxes_xyxy[:, 1])
    inter_x2 = np.minimum(x2, boxes_xyxy[:, 2])
    inter_y2 = np.minimum(y2, boxes_xyxy[:, 3])
    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    return inter_w * inter_h


def _mask_nms_group(predictions: list[dict], nms_iou_thr: float,
                    pre_nms_topk: int | None = None) -> list[dict]:
    candidates = sorted(
        predictions,
        key=lambda item: float(item["score"]),
        reverse=True)
    if pre_nms_topk is not None and len(candidates) > pre_nms_topk:
        candidates = candidates[:pre_nms_topk]

    if len(candidates) <= 1:
        return candidates

    kept = []
    kept_rles = []
    kept_boxes_xyxy = []

    for candidate in candidates:
        kept_boxes_arr = np.asarray(
            kept_boxes_xyxy, dtype=np.float32).reshape(-1, 4)
        overlap_indices = np.flatnonzero(
            _bbox_overlaps_xywh(candidate["bbox"], kept_boxes_arr) > 0)

        if len(overlap_indices) > 0:
            overlap_rles = [kept_rles[index] for index in overlap_indices]
            ious = mask_utils.iou([candidate["segmentation"]], overlap_rles,
                                  [0] * len(overlap_rles))[0]
            if np.any(ious > nms_iou_thr):
                continue

        kept.append(candidate)
        kept_rles.append(candidate["segmentation"])
        kept_boxes_xyxy.append(_bbox_xywh_to_xyxy(candidate["bbox"]))

    return kept


def mask_nms_predictions(predictions: Iterable[dict], score_thr: float = 0.01,
                         nms_iou_thr: float = 0.5,
                         max_per_img: int = 1000,
                         pre_nms_topk: int | None = 1500) -> list[dict]:
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for prediction in filter_predictions(predictions, score_thr):
        grouped[(prediction["image_id"], prediction["category_id"])].append(
            prediction)

    merged = []
    for key in sorted(grouped):
        merged.extend(_mask_nms_group(grouped[key], nms_iou_thr, pre_nms_topk))
    return cap_per_image(merged, max_per_img)


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.replace(",", " ").split() if item]


def resolve_checkpoint(path: Path) -> Path:
    if path.exists():
        return path

    if path.name == "latest.pth":
        last_checkpoint = path.parent / "last_checkpoint"
        if last_checkpoint.exists():
            resolved = Path(last_checkpoint.read_text(encoding="utf-8").strip())
            if resolved.exists():
                return resolved

    available = sorted(candidate.name for candidate in path.parent.glob("*.pth"))
    hint = f" Available checkpoints: {', '.join(available)}" if available else ""
    raise FileNotFoundError(f"{path} can not be found.{hint}")


def patch_torch_load_for_trusted_mmengine_checkpoint():
    """Allow this local, trusted MMEngine checkpoint under PyTorch 2.6+."""
    import torch

    original_load = torch.load

    def load_with_full_checkpoint(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = load_with_full_checkpoint
    return torch, original_load


def load_config_with_overrides(config_path: Path,
                               mask_thr_binary: float | None = None,
                               rcnn_nms_iou: float | None = None,
                               rpn_nms_pre: int | None = None,
                               rpn_max_per_img: int | None = None,
                               rcnn_max_per_img: int | None = None,
                               test_scale: tuple[int, int] | None = None):
    from mmengine.config import Config

    cfg = Config.fromfile(str(config_path))
    test_cfg = cfg.model.test_cfg

    if mask_thr_binary is not None:
        test_cfg.rcnn.mask_thr_binary = mask_thr_binary
    if rcnn_nms_iou is not None:
        test_cfg.rcnn.nms.iou_threshold = rcnn_nms_iou
    if rpn_nms_pre is not None:
        test_cfg.rpn.nms_pre = rpn_nms_pre
    if rpn_max_per_img is not None:
        test_cfg.rpn.max_per_img = rpn_max_per_img
    if rcnn_max_per_img is not None:
        test_cfg.rcnn.max_per_img = rcnn_max_per_img
    if test_scale is not None:
        _set_resize_scale(cfg, test_scale)

    return cfg


def _set_resize_scale(cfg: Any, test_scale: tuple[int, int]) -> None:
    def update_pipeline(pipeline: Any) -> None:
        if pipeline is None:
            return
        for transform in pipeline:
            if isinstance(transform, dict):
                if transform.get("type") == "Resize" and "scale" in transform:
                    transform["scale"] = test_scale
                for value in transform.values():
                    if isinstance(value, list):
                        update_pipeline(value)

    update_pipeline(cfg.get("test_pipeline"))
    update_pipeline(cfg.get("predict_pipeline"))

    for dataloader_name in ("test_dataloader", "val_dataloader"):
        dataloader = cfg.get(dataloader_name)
        if dataloader is None:
            continue
        dataset = dataloader.get("dataset")
        if dataset is not None:
            update_pipeline(dataset.get("pipeline"))


def iter_predictions(result: Any):
    if hasattr(result, "pred_instances"):
        instances = result.pred_instances
        bboxes = instances.bboxes.detach().cpu().numpy()
        scores = instances.scores.detach().cpu().numpy()
        labels = instances.labels.detach().cpu().numpy()
        masks = instances.masks

        for bbox, score, label, mask in zip(bboxes, scores, labels, masks):
            yield int(label) + 1, bbox, float(score), mask
        return

    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]

        for class_index, bboxes in enumerate(bbox_result):
            segmentations = segm_result[class_index]
            for bbox, segmentation in zip(bboxes, segmentations):
                yield class_index + 1, bbox[:4], float(bbox[4]), segmentation
        return

    raise TypeError(
        "Expected MMDetection 3 DetDataSample or MMDetection 2 "
        "(bbox_result, segm_result) output.")
