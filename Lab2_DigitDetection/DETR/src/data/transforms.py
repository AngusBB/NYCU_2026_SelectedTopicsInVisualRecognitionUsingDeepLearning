"""Image and target transforms for detection."""

from __future__ import annotations

import random
from typing import Dict, Iterable, Optional

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image, ImageFilter

from src.utils.box_ops import clip_boxes_to_image


def _get_size_with_aspect_ratio(
    image_size: tuple[int, int],
    size: int,
    max_size: Optional[int] = None,
) -> tuple[int, int]:
    width, height = image_size

    if max_size is not None:
        min_original_size = float(min(width, height))
        max_original_size = float(max(width, height))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))

    if (width <= height and width == size) or (height <= width and height == size):
        return height, width

    if width < height:
        output_width = size
        output_height = int(size * height / width)
    else:
        output_height = size
        output_width = int(size * width / height)

    return output_height, output_width


def resize(
    image: Image.Image,
    target: Optional[Dict[str, torch.Tensor]],
    size: int,
    max_size: Optional[int] = None,
) -> tuple[Image.Image, Optional[Dict[str, torch.Tensor]]]:
    output_height, output_width = _get_size_with_aspect_ratio(image.size, size, max_size)
    return resize_to_shape(image, target, output_size=(output_height, output_width))


def resize_to_shape(
    image: Image.Image,
    target: Optional[Dict[str, torch.Tensor]],
    output_size: tuple[int, int],
) -> tuple[Image.Image, Optional[Dict[str, torch.Tensor]]]:
    output_height, output_width = output_size
    resized_image = F.resize(image, [output_height, output_width], antialias=True)

    if target is None:
        return resized_image, target

    scale_x = output_width / image.size[0]
    scale_y = output_height / image.size[1]

    target = target.copy()
    if "boxes" in target and target["boxes"].numel() > 0:
        scale = torch.tensor(
            [scale_x, scale_y, scale_x, scale_y],
            dtype=target["boxes"].dtype,
        )
        target["boxes"] = clip_boxes_to_image(
            target["boxes"] * scale,
            height=output_height,
            width=output_width,
        )
        widths = (target["boxes"][:, 2] - target["boxes"][:, 0]).clamp(min=0)
        heights = (target["boxes"][:, 3] - target["boxes"][:, 1]).clamp(min=0)
        target["area"] = widths * heights

    target["size"] = torch.tensor([output_height, output_width], dtype=torch.int64)
    return resized_image, target


def build_mosaic(
    samples: list[tuple[Image.Image, Dict[str, torch.Tensor]]],
    mosaic_size: int,
    center_ratio_range: tuple[float, float] = (0.4, 0.6),
    min_box_size: float = 2.0,
) -> tuple[Image.Image, Dict[str, torch.Tensor]]:
    """Combine four samples into a square mosaic canvas."""
    if len(samples) != 4:
        raise ValueError("Mosaic augmentation expects exactly four samples.")

    canvas = Image.new("RGB", (mosaic_size, mosaic_size), color=(114, 114, 114))
    center_x = int(random.uniform(*center_ratio_range) * mosaic_size)
    center_y = int(random.uniform(*center_ratio_range) * mosaic_size)
    regions = [
        (0, 0, center_x, center_y),
        (center_x, 0, mosaic_size, center_y),
        (0, center_y, center_x, mosaic_size),
        (center_x, center_y, mosaic_size, mosaic_size),
    ]

    mosaic_boxes = []
    mosaic_labels = []
    mosaic_areas = []
    mosaic_iscrowd = []

    for (image, target), (x1, y1, x2, y2) in zip(samples, regions):
        region_width = max(x2 - x1, 1)
        region_height = max(y2 - y1, 1)
        resized_image = image.resize((region_width, region_height), Image.BILINEAR)
        canvas.paste(resized_image, (x1, y1))

        boxes = target["boxes"]
        if boxes.numel() == 0:
            continue

        orig_height, orig_width = target["size"].tolist()
        scale_x = region_width / max(orig_width, 1)
        scale_y = region_height / max(orig_height, 1)

        boxes = boxes.clone()
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_x + x1
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_y + y1
        boxes = clip_boxes_to_image(boxes, height=mosaic_size, width=mosaic_size)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        valid = (widths >= min_box_size) & (heights >= min_box_size)
        if not valid.any():
            continue

        boxes = boxes[valid]
        labels = target["labels"][valid]
        iscrowd = target["iscrowd"][valid]
        areas = widths[valid] * heights[valid]

        mosaic_boxes.append(boxes)
        mosaic_labels.append(labels)
        mosaic_areas.append(areas)
        mosaic_iscrowd.append(iscrowd)

    if mosaic_boxes:
        boxes = torch.cat(mosaic_boxes, dim=0)
        labels = torch.cat(mosaic_labels, dim=0)
        areas = torch.cat(mosaic_areas, dim=0)
        iscrowd = torch.cat(mosaic_iscrowd, dim=0)
    else:
        boxes = torch.empty((0, 4), dtype=torch.float32)
        labels = torch.empty((0,), dtype=torch.int64)
        areas = torch.empty((0,), dtype=torch.float32)
        iscrowd = torch.empty((0,), dtype=torch.int64)

    primary_target = samples[0][1]
    mosaic_target = {
        "image_id": primary_target["image_id"].clone(),
        "orig_size": torch.tensor([mosaic_size, mosaic_size], dtype=torch.int64),
        "size": torch.tensor([mosaic_size, mosaic_size], dtype=torch.int64),
        "boxes": boxes,
        "labels": labels,
        "area": areas,
        "iscrowd": iscrowd,
    }
    return canvas, mosaic_target


def build_mixup(
    primary_sample: tuple[Image.Image, Dict[str, torch.Tensor]],
    secondary_sample: tuple[Image.Image, Dict[str, torch.Tensor]],
    alpha: float = 8.0,
    min_box_size: float = 2.0,
) -> tuple[Image.Image, Dict[str, torch.Tensor]]:
    """Blend two samples together and keep the union of their boxes."""
    if alpha <= 0.0:
        raise ValueError("mixup alpha must be positive.")

    primary_image, primary_target = primary_sample
    secondary_image, secondary_target = secondary_sample
    primary_height, primary_width = (int(value) for value in primary_target["size"].tolist())
    secondary_image, secondary_target = resize_to_shape(
        secondary_image,
        secondary_target,
        output_size=(primary_height, primary_width),
    )

    lambda_value = random.betavariate(alpha, alpha)
    mixed_image = Image.blend(primary_image, secondary_image, alpha=1.0 - lambda_value)

    mixed_boxes = []
    mixed_labels = []
    mixed_areas = []
    mixed_iscrowd = []

    for target in (primary_target, secondary_target):
        boxes = target["boxes"]
        if boxes.numel() == 0:
            continue

        boxes = clip_boxes_to_image(boxes.clone(), height=primary_height, width=primary_width)
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        valid = (widths >= min_box_size) & (heights >= min_box_size)
        if not valid.any():
            continue

        mixed_boxes.append(boxes[valid])
        mixed_labels.append(target["labels"][valid])
        mixed_areas.append(widths[valid] * heights[valid])
        mixed_iscrowd.append(target["iscrowd"][valid])

    if mixed_boxes:
        boxes = torch.cat(mixed_boxes, dim=0)
        labels = torch.cat(mixed_labels, dim=0)
        areas = torch.cat(mixed_areas, dim=0)
        iscrowd = torch.cat(mixed_iscrowd, dim=0)
    else:
        boxes = torch.empty((0, 4), dtype=torch.float32)
        labels = torch.empty((0,), dtype=torch.int64)
        areas = torch.empty((0,), dtype=torch.float32)
        iscrowd = torch.empty((0,), dtype=torch.int64)

    size = torch.tensor([primary_height, primary_width], dtype=torch.int64)
    mixed_target = {
        "image_id": primary_target["image_id"].clone(),
        "orig_size": size.clone(),
        "size": size,
        "boxes": boxes,
        "labels": labels,
        "area": areas,
        "iscrowd": iscrowd,
    }
    return mixed_image, mixed_target


class Compose:
    def __init__(self, transforms: Iterable) -> None:
        self.transforms = list(transforms)

    def __call__(
        self,
        image: Image.Image,
        target: Optional[Dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


class RandomResize:
    def __init__(self, scales: list[int], max_size: Optional[int] = None) -> None:
        self.scales = scales
        self.max_size = max_size

    def __call__(
        self,
        image: Image.Image,
        target: Optional[Dict[str, torch.Tensor]],
    ) -> tuple[Image.Image, Optional[Dict[str, torch.Tensor]]]:
        size = random.choice(self.scales)
        return resize(image, target, size=size, max_size=self.max_size)


class ResizeShortestEdge:
    def __init__(self, size: int, max_size: Optional[int] = None) -> None:
        self.size = size
        self.max_size = max_size

    def __call__(
        self,
        image: Image.Image,
        target: Optional[Dict[str, torch.Tensor]],
    ) -> tuple[Image.Image, Optional[Dict[str, torch.Tensor]]]:
        return resize(image, target, size=self.size, max_size=self.max_size)


class RandomColorJitter:
    def __init__(self, probability: float = 0.8) -> None:
        self.probability = probability
        self.transform = T.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
        )

    def __call__(
        self,
        image: Image.Image,
        target: Optional[Dict[str, torch.Tensor]],
    ) -> tuple[Image.Image, Optional[Dict[str, torch.Tensor]]]:
        if random.random() < self.probability:
            image = self.transform(image)
        return image, target


class RandomGaussianBlur:
    def __init__(self, probability: float = 0.15) -> None:
        self.probability = probability

    def __call__(
        self,
        image: Image.Image,
        target: Optional[Dict[str, torch.Tensor]],
    ) -> tuple[Image.Image, Optional[Dict[str, torch.Tensor]]]:
        if random.random() < self.probability:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.2)))
        return image, target


class ToTensor:
    def __call__(
        self,
        image: Image.Image,
        target: Optional[Dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        tensor = F.pil_to_tensor(image).float() / 255.0
        return tensor, target


class Normalize:
    def __init__(self, mean: list[float], std: list[float]) -> None:
        self.mean = mean
        self.std = std

    def __call__(
        self,
        image: torch.Tensor,
        target: Optional[Dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        return F.normalize(image, mean=self.mean, std=self.std), target


def build_train_transforms(config: Dict) -> Compose:
    data_cfg = config["data"]
    return Compose(
        [
            RandomColorJitter(),
            RandomGaussianBlur(),
            RandomResize(
                scales=list(data_cfg["train_scales"]),
                max_size=data_cfg["max_size"],
            ),
            ToTensor(),
            Normalize(mean=data_cfg["image_mean"], std=data_cfg["image_std"]),
        ]
    )


def build_eval_transforms(config: Dict, scale: Optional[int] = None) -> Compose:
    data_cfg = config["data"]
    eval_scale = scale if scale is not None else int(data_cfg["eval_scales"][0])
    return Compose(
        [
            ResizeShortestEdge(eval_scale, max_size=data_cfg["max_size"]),
            ToTensor(),
            Normalize(mean=data_cfg["image_mean"], std=data_cfg["image_std"]),
        ]
    )
