"""Bounding box utilities."""

from __future__ import annotations

import torch


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x_center, y_center, width, height = boxes.unbind(-1)
    half_width = width / 2.0
    half_height = height / 2.0
    return torch.stack(
        [
            x_center - half_width,
            y_center - half_height,
            x_center + half_width,
            y_center + half_height,
        ],
        dim=-1,
    )


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x_min, y_min, x_max, y_max = boxes.unbind(-1)
    return torch.stack(
        [
            (x_min + x_max) / 2.0,
            (y_min + y_max) / 2.0,
            x_max - x_min,
            y_max - y_min,
        ],
        dim=-1,
    )


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (
        boxes[:, 3] - boxes[:, 1]
    ).clamp(min=0)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    top_left = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (bottom_right - top_left).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    iou = inter / union.clamp(min=1e-6)
    return iou, union


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    iou, union = box_iou(boxes1, boxes2)

    top_left = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (bottom_right - top_left).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]
    return iou - (area - union) / area.clamp(min=1e-6)


def clip_boxes_to_image(boxes: torch.Tensor, height: int, width: int) -> torch.Tensor:
    boxes[..., 0::2] = boxes[..., 0::2].clamp(min=0, max=width)
    boxes[..., 1::2] = boxes[..., 1::2].clamp(min=0, max=height)
    return boxes

