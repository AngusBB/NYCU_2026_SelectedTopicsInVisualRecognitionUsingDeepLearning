"""Hungarian matcher for set prediction."""

from __future__ import annotations

from typing import Dict, List

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from src.utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, generalized_box_iou


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 2.0) -> None:
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        if cost_class == 0 and cost_bbox == 0 and cost_giou == 0:
            raise ValueError("All matching costs cannot be zero.")

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]):
        batch_size, num_queries = outputs["pred_logits"].shape[:2]

        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        target_labels = torch.cat([target["labels"] for target in targets], dim=0)
        target_boxes = torch.cat([_normalize_target_boxes(target) for target in targets], dim=0)

        if target_labels.numel() == 0:
            empty = torch.empty(0, dtype=torch.int64)
            return [(empty, empty) for _ in range(batch_size)]

        cost_class = -out_prob[:, target_labels]
        cost_bbox = torch.cdist(out_bbox, target_boxes, p=1)
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox),
            box_cxcywh_to_xyxy(target_boxes),
        )

        cost_matrix = (
            self.cost_class * cost_class
            + self.cost_bbox * cost_bbox
            + self.cost_giou * cost_giou
        )
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        sizes = [len(target["boxes"]) for target in targets]
        indices = []
        start = 0
        for batch_index, size in enumerate(sizes):
            if size == 0:
                empty = torch.empty(0, dtype=torch.int64)
                indices.append((empty, empty))
                continue
            current = cost_matrix[batch_index, :, start : start + size]
            source_index, target_index = linear_sum_assignment(current)
            indices.append(
                (
                    torch.as_tensor(source_index, dtype=torch.int64),
                    torch.as_tensor(target_index, dtype=torch.int64),
                )
            )
            start += size
        return indices


def _normalize_target_boxes(target: Dict[str, torch.Tensor]) -> torch.Tensor:
    boxes = target["boxes"]
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)

    height, width = target["size"].unbind()
    scale = torch.stack((width, height, width, height)).to(dtype=boxes.dtype)
    return box_xyxy_to_cxcywh(boxes / scale)
