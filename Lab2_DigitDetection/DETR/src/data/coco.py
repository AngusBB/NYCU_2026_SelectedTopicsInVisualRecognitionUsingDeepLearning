"""COCO-format dataset wrappers."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.transforms import build_mixup, build_mosaic


class COCODetectionDataset(Dataset):
    """Minimal COCO-format dataset for training, validation, and inference."""

    def __init__(
        self,
        image_dir: str | Path,
        annotation_file: Optional[str | Path],
        transforms=None,
        category_ids: Optional[Sequence[int]] = None,
        mosaic_probability: float = 0.0,
        mosaic_size: Optional[int] = None,
        mosaic_center_ratio_range: tuple[float, float] = (0.4, 0.6),
        mosaic_min_box_size: float = 2.0,
        mixup_probability: float = 0.0,
        mixup_alpha: float = 8.0,
        mixup_min_box_size: float = 2.0,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.annotation_file = Path(annotation_file) if annotation_file else None
        self.transforms = transforms
        self.mosaic_probability = mosaic_probability
        self.mosaic_size = mosaic_size
        self.mosaic_center_ratio_range = mosaic_center_ratio_range
        self.mosaic_min_box_size = mosaic_min_box_size
        self.mixup_probability = mixup_probability
        self.mixup_alpha = mixup_alpha
        self.mixup_min_box_size = mixup_min_box_size

        if self.annotation_file is not None:
            with self.annotation_file.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
        else:
            raw = {
                "images": [
                    {
                        "id": _infer_image_id(path, index),
                        "file_name": path.name,
                        "width": None,
                        "height": None,
                    }
                    for index, path in enumerate(sorted(self.image_dir.iterdir()))
                    if path.is_file()
                ]
            }

        if category_ids is None:
            if "categories" in raw and raw["categories"]:
                category_ids = [cat["id"] for cat in sorted(raw["categories"], key=lambda item: item["id"])]
            else:
                category_ids = list(range(1, 11))
        self.category_ids = list(category_ids)
        self.cat_id_to_label = {cat_id: idx for idx, cat_id in enumerate(self.category_ids)}
        self.label_to_cat_id = {idx: cat_id for cat_id, idx in self.cat_id_to_label.items()}

        self.images = sorted(raw["images"], key=lambda item: item["id"])
        self.annotations_by_image: Dict[int, List[Dict]] = defaultdict(list)
        for ann in raw.get("annotations", []):
            self.annotations_by_image[int(ann["image_id"])].append(ann)

        self.coco_raw = raw

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image, target = self._load_training_sample(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        if target["boxes"].numel() > 0:
            valid = (
                (target["boxes"][:, 2] > target["boxes"][:, 0])
                & (target["boxes"][:, 3] > target["boxes"][:, 1])
            )
            for key in ("boxes", "labels", "area", "iscrowd"):
                target[key] = target[key][valid]

        return image, target

    def _load_training_sample(self, index: int, allow_mixup: bool = True):
        if self._use_mosaic():
            image, target = self._load_mosaic(index)
        else:
            image, target = self._load_sample(index)

        if allow_mixup and self._use_mixup():
            mix_index = random.randrange(len(self.images))
            mix_sample = self._load_training_sample(mix_index, allow_mixup=False)
            image, target = build_mixup(
                primary_sample=(image, target),
                secondary_sample=mix_sample,
                alpha=self.mixup_alpha,
                min_box_size=self.mixup_min_box_size,
            )
        return image, target

    def _use_mosaic(self) -> bool:
        return (
            self.mosaic_probability > 0.0
            and self.annotation_file is not None
            and len(self.images) >= 4
            and random.random() < self.mosaic_probability
        )

    def _use_mixup(self) -> bool:
        return (
            self.mixup_probability > 0.0
            and self.mixup_alpha > 0.0
            and self.annotation_file is not None
            and len(self.images) >= 2
            and random.random() < self.mixup_probability
        )

    def _load_sample(self, index: int):
        image_info = self.images[index]
        image_path = self.image_dir / image_info["file_name"]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        image_id = int(image_info["id"])

        annotations = self.annotations_by_image.get(image_id, [])
        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in annotations:
            x_min, y_min, box_width, box_height = ann["bbox"]
            if box_width <= 0 or box_height <= 0:
                continue
            x_max = x_min + box_width
            y_max = y_min + box_height
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(self.cat_id_to_label[int(ann["category_id"])])
            areas.append(float(ann.get("area", box_width * box_height)))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        target = {
            "image_id": torch.tensor(image_id, dtype=torch.int64),
            "orig_size": torch.tensor([height, width], dtype=torch.int64),
            "size": torch.tensor([height, width], dtype=torch.int64),
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64),
        }
        return image, target

    def _load_mosaic(self, index: int):
        if self.mosaic_size is None:
            raise ValueError("mosaic_size must be provided when mosaic augmentation is enabled.")
        sample_indices = [index] + [random.randrange(len(self.images)) for _ in range(3)]
        samples = [self._load_sample(sample_index) for sample_index in sample_indices]
        return build_mosaic(
            samples=samples,
            mosaic_size=self.mosaic_size,
            center_ratio_range=self.mosaic_center_ratio_range,
            min_box_size=self.mosaic_min_box_size,
        )

    def get_coco_api(self) -> Dict:
        return self.coco_raw


def _infer_image_id(path: Path, index: int) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return index
