"""Validation and COCO evaluation utilities."""

from __future__ import annotations

import copy
from typing import Dict, List, Optional

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.utils.distributed import all_gather_object, is_main_process
from src.utils.misc import MetricLogger, move_targets_to_device, reduce_dict


def build_coco_api(coco_raw: Dict) -> COCO:
    coco = COCO()
    coco.dataset = copy.deepcopy(coco_raw)
    coco.createIndex()
    return coco


def evaluate_coco(
    model: torch.nn.Module,
    criterion: Optional[torch.nn.Module],
    postprocessor: torch.nn.Module,
    data_loader,
    device: torch.device,
    score_threshold: float = 0.05,
    max_detections_per_image: int = 300,
    use_amp: bool = True,
) -> Dict[str, float]:
    model.eval()
    if criterion is not None:
        criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    predictions: List[Dict] = []

    with torch.no_grad():
        for images, targets in metric_logger.log_every(data_loader, print_freq=50, header="Eval "):
            images = [image.to(device) for image in images]
            targets = move_targets_to_device(targets, device)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
                outputs = model(images)
                if criterion is not None:
                    loss_dict = criterion(outputs, targets)
                    reduced = reduce_dict(loss_dict)
                    metric_logger.update(
                        **{
                            key: value
                            for key, value in reduced.items()
                            if key.startswith("loss_") or key == "class_error"
                        }
                    )

            target_sizes = torch.stack([target["orig_size"] for target in targets], dim=0)
            results = postprocessor(outputs, target_sizes)
            for target, result in zip(targets, results):
                record_list = convert_to_coco_predictions(
                    image_id=int(target["image_id"].item()),
                    result=result,
                    score_threshold=score_threshold,
                    max_detections_per_image=max_detections_per_image,
                )
                predictions.extend(record_list)

    metric_logger.synchronize_between_processes()

    gathered_predictions: List[Dict] = []
    for shard in all_gather_object(predictions):
        gathered_predictions.extend(shard)

    stats = {name: meter.global_avg for name, meter in metric_logger.meters.items()}

    dataset = getattr(data_loader.dataset, "dataset", data_loader.dataset)
    coco_raw = getattr(dataset, "coco_raw", None)
    if coco_raw is None or "annotations" not in coco_raw:
        return stats

    if not is_main_process():
        return stats

    coco_gt = build_coco_api(coco_raw)
    coco_stats = run_coco_eval(coco_gt, gathered_predictions)
    stats.update(coco_stats)
    return stats


def convert_to_coco_predictions(
    image_id: int,
    result: Dict[str, torch.Tensor],
    score_threshold: float,
    max_detections_per_image: int,
) -> List[Dict]:
    scores = result["scores"]
    labels = result["labels"]
    boxes = result["boxes"]

    keep = scores >= score_threshold
    scores = scores[keep]
    labels = labels[keep]
    boxes = boxes[keep]

    if scores.numel() == 0:
        return []

    order = torch.argsort(scores, descending=True)[:max_detections_per_image]
    scores = scores[order].cpu()
    labels = labels[order].cpu()
    boxes = boxes[order].cpu()

    records = []
    for score, label, box in zip(scores, labels, boxes):
        records.append(
            {
                "image_id": image_id,
                "category_id": int(label.item()),
                "bbox": [float(value) for value in box.tolist()],
                "score": float(score.item()),
            }
        )
    return records


def run_coco_eval(coco_gt: COCO, predictions: List[Dict]) -> Dict[str, float]:
    if len(predictions) == 0:
        return {}

    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stat_names = [
        "bbox_mAP",
        "bbox_mAP50",
        "bbox_mAP75",
        "bbox_mAP_small",
        "bbox_mAP_medium",
        "bbox_mAP_large",
        "bbox_AR1",
        "bbox_AR10",
        "bbox_AR100",
        "bbox_AR_small",
        "bbox_AR_medium",
        "bbox_AR_large",
    ]
    return {name: float(value) for name, value in zip(stat_names, coco_eval.stats.tolist())}
