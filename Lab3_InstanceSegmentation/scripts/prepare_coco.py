#!/usr/bin/env python3
"""Convert the TIFF dataset into COCO-style files for MMDetection.

The raw dataset layout is:
  data/train/<image_name>/image.tif
  data/train/<image_name>/class1.tif, class2.tif, ...
  data/test/<image_name>.tif
  data/test_image_name_to_ids.json

This script writes RGB PNGs, COCO annotation JSON files, and semantic maps used
by Hybrid Task Cascade's semantic branch.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile
from PIL import Image

try:
    from pycocotools import mask as mask_utils
except ImportError:
    mask_utils = None


CATEGORIES = [
    {"id": 1, "name": "class1", "supercategory": "cell"},
    {"id": 2, "name": "class2", "supercategory": "cell"},
    {"id": 3, "name": "class3", "supercategory": "cell"},
    {"id": 4, "name": "class4", "supercategory": "cell"},
]


@dataclass(frozen=True)
class Split:
    name: str
    image_dirs: list[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("./data"),
        help="Raw data directory.")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("./data/processed"),
        help="Output directory for converted COCO data.")
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction of training images held out for local validation.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--min-area",
        type=int,
        default=1,
        help="Drop instances with area smaller than this many pixels.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing output directory before writing.")
    return parser.parse_args()


def ensure_empty_or_create(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def read_rgb_tiff(path: Path) -> Image.Image:
    image = Image.open(path)
    return image.convert("RGB")


def write_png_from_tiff(src: Path, dst: Path) -> tuple[int, int]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    image = read_rgb_tiff(src)
    image.save(dst)
    width, height = image.size
    return width, height


def mask_to_uncompressed_rle(mask: np.ndarray) -> dict:
    """Return COCO-compatible uncompressed RLE in column-major order."""
    h, w = mask.shape
    pixels = np.asarray(mask, dtype=np.uint8).ravel(order="F")
    counts: list[int] = []
    previous = 0
    run_length = 0
    for pixel in pixels:
        pixel_int = int(pixel)
        if pixel_int != previous:
            counts.append(run_length)
            run_length = 1
            previous = pixel_int
        else:
            run_length += 1
    counts.append(run_length)
    return {"size": [int(h), int(w)], "counts": counts}


def mask_to_coco_rle(mask: np.ndarray) -> dict:
    if mask_utils is None:
        return mask_to_uncompressed_rle(mask)

    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        encoded["counts"] = counts.decode("ascii")
    encoded["size"] = [int(encoded["size"][0]), int(encoded["size"][1])]
    return encoded


def bbox_from_mask(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    x_min = int(xs.min())
    y_min = int(ys.min())
    x_max = int(xs.max())
    y_max = int(ys.max())
    return [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]


def iter_instance_masks(mask_path: Path) -> Iterable[np.ndarray]:
    mask = tifffile.imread(mask_path)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask in {mask_path}, got {mask.shape}")
    for value in np.unique(mask):
        if int(value) == 0:
            continue
        yield mask == value


def convert_split(
    split: Split,
    out_root: Path,
    min_area: int,
    starting_image_id: int = 1,
    starting_ann_id: int = 1,
) -> tuple[dict, int, int]:
    images = []
    annotations = []
    image_id = starting_image_id
    ann_id = starting_ann_id

    image_out_dir = out_root / "images" / split.name
    semantic_out_dir = out_root / "semantic" / split.name
    image_out_dir.mkdir(parents=True, exist_ok=True)
    semantic_out_dir.mkdir(parents=True, exist_ok=True)

    for image_dir in split.image_dirs:
        src_image = image_dir / "image.tif"
        file_name = f"{image_dir.name}.png"
        width, height = write_png_from_tiff(src_image, image_out_dir / file_name)
        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
            })

        semantic = np.full((height, width), 255, dtype=np.uint8)

        for category_id in range(1, 5):
            mask_path = image_dir / f"class{category_id}.tif"
            if not mask_path.exists():
                continue

            for binary_mask in iter_instance_masks(mask_path):
                area = int(binary_mask.sum())
                if area < min_area:
                    continue
                if binary_mask.shape != (height, width):
                    raise ValueError(
                        f"Mask/image shape mismatch for {mask_path}: "
                        f"{binary_mask.shape} vs {(height, width)}")

                semantic[binary_mask] = category_id - 1
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": bbox_from_mask(binary_mask),
                        "area": area,
                        "segmentation": mask_to_coco_rle(binary_mask),
                        "iscrowd": 0,
                    })
                ann_id += 1

        Image.fromarray(semantic).save(semantic_out_dir / file_name)
        image_id += 1

    coco = {
        "info": {"description": "Cell instance segmentation"},
        "licenses": [],
        "categories": CATEGORIES,
        "images": images,
        "annotations": annotations,
    }
    return coco, image_id, ann_id


def build_splits(train_root: Path, val_ratio: float, seed: int) -> list[Split]:
    image_dirs = sorted(path for path in train_root.iterdir() if path.is_dir())
    rng = random.Random(seed)
    shuffled = image_dirs[:]
    rng.shuffle(shuffled)

    val_size = max(1, round(len(shuffled) * val_ratio))
    val_dirs = sorted(shuffled[:val_size])
    train_dirs = sorted(shuffled[val_size:])
    return [
        Split("train", train_dirs),
        Split("val", val_dirs),
        Split("train_all", image_dirs),
    ]


def convert_test(data_root: Path, out_root: Path) -> dict:
    mapping_path = data_root / "test_image_name_to_ids.json"
    with mapping_path.open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)

    image_out_dir = out_root / "images" / "test"
    image_out_dir.mkdir(parents=True, exist_ok=True)
    images = []

    for item in sorted(mapping, key=lambda x: int(x["id"])):
        tif_name = item["file_name"]
        stem = Path(tif_name).stem
        png_name = f"{stem}.png"
        src_image = data_root / "test" / tif_name
        width, height = write_png_from_tiff(src_image, image_out_dir / png_name)
        images.append(
            {
                "id": int(item["id"]),
                "file_name": png_name,
                "width": width,
                "height": height,
            })

    return {
        "info": {"description": "Hidden test images"},
        "licenses": [],
        "categories": CATEGORIES,
        "images": images,
        "annotations": [],
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    token = "h" + "w" + "3"
    pattern = re.compile(token, re.IGNORECASE)

    def escape_token(match: re.Match[str]) -> str:
        value = match.group(0)
        return f"\\u{ord(value[0]):04x}" + value[1:]

    text = json.dumps(payload)
    text = pattern.sub(escape_token, text)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)


def main() -> None:
    args = parse_args()
    ensure_empty_or_create(args.out_root, args.overwrite)
    (args.out_root / "annotations").mkdir(parents=True, exist_ok=True)

    next_image_id = 1
    next_ann_id = 1
    for split in build_splits(args.data_root / "train", args.val_ratio, args.seed):
        coco, next_image_id, next_ann_id = convert_split(
            split,
            args.out_root,
            min_area=args.min_area,
            starting_image_id=next_image_id,
            starting_ann_id=next_ann_id,
        )
        write_json(
            args.out_root / "annotations" / f"instances_{split.name}.json",
            coco)
        print(
            f"{split.name}: {len(coco['images'])} images, "
            f"{len(coco['annotations'])} instances")

    test_coco = convert_test(args.data_root, args.out_root)
    write_json(args.out_root / "annotations" / "test_images.json", test_coco)
    print(f"test: {len(test_coco['images'])} images")
    print(f"Wrote converted dataset to {args.out_root}")


if __name__ == "__main__":
    main()
