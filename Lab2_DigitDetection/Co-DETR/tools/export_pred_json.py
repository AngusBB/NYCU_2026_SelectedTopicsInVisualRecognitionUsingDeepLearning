#!/usr/bin/env python3
"""Run Co-DETR inference and export homework-style pred.json."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mmcv
import numpy as np
import torch
from mmcv import DictAction
from PIL import Image
from torchvision.ops import nms

from mmdet.apis import inference_detector, init_detector
from projects import *  # noqa: F401,F403 - register project modules before model init


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=str, help="Path to the MMDetection config file.")
    parser.add_argument("checkpoint", type=str, help="Checkpoint path.")
    parser.add_argument(
        "--image-dir",
        type=str,
        default="../nycu-hw2-data/test",
        help="Directory containing test images.",
    )
    parser.add_argument(
        "--output-coordinate-image-dir",
        type=str,
        default=None,
        help=(
            "Optional directory whose image sizes define the coordinate system for exported boxes. "
            "Use this when inference runs on resized/SR-upscaled images but pred.json must be "
            "in the original image coordinates. Image ids are inferred from filename stems."
        ),
    )
    parser.add_argument(
        "--coordinate-map-mode",
        choices=["direct-scale", "square-pad"],
        default="direct-scale",
        help=(
            "How to map inference-image boxes to --output-coordinate-image-dir. "
            "direct-scale rescales x/y by width/height ratios. square-pad also removes "
            "padding introduced by super_resolve_coco_dataset.py --resize-mode square-pad."
        ),
    )
    parser.add_argument(
        "--coordinate-pad-position",
        choices=["top-left", "center"],
        default="top-left",
        help="Padding position used with --coordinate-map-mode square-pad.",
    )
    parser.add_argument("--output-json", type=str, default="pred.json")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.65)
    parser.add_argument("--max-detections-per-image", type=int, default=300)
    parser.add_argument(
        "--tta-scales",
        nargs="+",
        type=int,
        default=None,
        help="Optional square test scales, e.g. --tta-scales 896 1024 1152.",
    )
    parser.add_argument(
        "--tta-hflip",
        action="store_true",
        help="Enable horizontal flip test-time augmentation for each selected scale.",
    )
    parser.add_argument(
        "--category-ids",
        nargs="+",
        type=int,
        default=None,
        help="Optional explicit category ids matching the config class order.",
    )
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--zip-output", action="store_true")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="Override config settings, like MMDetection tools.",
    )
    return parser.parse_args()


def load_samples(image_dir: Path) -> list[tuple[int, Path]]:
    samples = []
    for index, path in enumerate(sorted(path for path in image_dir.iterdir() if path.is_file()), start=0):
        samples.append((infer_image_id(path, index), path))
    return samples


def load_image_sizes_from_samples(samples: Sequence[tuple[int, Path]]) -> dict[int, tuple[int, int]]:
    sizes = {}
    for image_id, path in samples:
        with Image.open(path) as image:
            width, height = image.size
        sizes[int(image_id)] = (int(width), int(height))
    return sizes


def load_image_sizes_from_dir(image_dir: Path) -> dict[int, tuple[int, int]]:
    samples = load_samples(image_dir=image_dir)
    return load_image_sizes_from_samples(samples)


def infer_image_id(path: Path, index: int) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return index


def resolve_class_names(cfg: mmcv.Config) -> list[str]:
    candidates = []
    for split_name in ("test", "val", "train"):
        split_cfg = cfg.data.get(split_name)
        if isinstance(split_cfg, dict) and split_cfg.get("classes") is not None:
            candidates.append(split_cfg.get("classes"))
    if not candidates:
        raise ValueError("Could not find dataset classes in the config.")
    return [str(name) for name in candidates[0]]


def resolve_category_ids(
    class_names: Sequence[str],
    categories: list[dict] | None,
    explicit_category_ids: Sequence[int] | None,
) -> list[int]:
    if explicit_category_ids is not None:
        category_ids = [int(cat_id) for cat_id in explicit_category_ids]
    elif categories:
        name_to_id = {str(category["name"]): int(category["id"]) for category in categories}
        if all(name in name_to_id for name in class_names):
            category_ids = [name_to_id[name] for name in class_names]
        else:
            category_ids = [int(category["id"]) for category in sorted(categories, key=lambda item: int(item["id"]))]
    else:
        category_ids = list(range(1, len(class_names) + 1))

    if len(category_ids) != len(class_names):
        raise ValueError(
            f"Expected {len(class_names)} category ids for classes {class_names}, got {category_ids}"
        )
    return category_ids


def extract_test_scales(cfg: mmcv.Config) -> list[int]:
    for transform in cfg.data.test.pipeline:
        transform_type = transform.get("type")
        if transform_type == "MultiScaleFlipAug":
            return normalize_scale_config(transform.get("img_scale"))
        if transform_type == "Resize":
            return normalize_scale_config(transform.get("img_scale"))
    raise ValueError("Could not find a Resize or MultiScaleFlipAug step in cfg.data.test.pipeline")


def normalize_scale_config(scale_cfg) -> list[int]:
    if isinstance(scale_cfg, (list, tuple)):
        if scale_cfg and isinstance(scale_cfg[0], (list, tuple)):
            return [int(max(scale)) for scale in scale_cfg]
        if len(scale_cfg) == 2 and all(isinstance(value, (int, float)) for value in scale_cfg):
            return [int(max(scale_cfg))]
        return [int(value) for value in scale_cfg]
    if scale_cfg is None:
        raise ValueError("The test pipeline does not define img_scale.")
    return [int(scale_cfg)]


def set_test_scale(cfg: mmcv.Config, scale: int) -> None:
    updated = False
    for transform in cfg.data.test.pipeline:
        transform_type = transform.get("type")
        if transform_type == "MultiScaleFlipAug":
            transform["img_scale"] = (int(scale), int(scale))
            transform["flip"] = False
            transform["flip_direction"] = "horizontal"
            updated = True
            break
        if transform_type == "Resize":
            transform["img_scale"] = (int(scale), int(scale))
            updated = True
            break
    if not updated:
        raise ValueError("Could not update the test pipeline scale.")


def chunked(items: Sequence[tuple[int, Path]], batch_size: int) -> Iterable[Sequence[tuple[int, Path]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def normalize_single_result(result) -> list[np.ndarray]:
    if isinstance(result, tuple):
        result = result[0]
    if not isinstance(result, list):
        raise TypeError(f"Unsupported detection result type: {type(result)}")
    return result


def horizontal_flip_image(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(image[:, ::-1, ...])


def unflip_single_result(result: list[np.ndarray], image_width: int) -> list[np.ndarray]:
    restored = []
    for dets in result:
        dets = np.asarray(dets)
        if dets.size == 0:
            restored.append(dets.copy())
            continue
        dets = dets.copy()
        x1 = dets[:, 0].copy()
        x2 = dets[:, 2].copy()
        dets[:, 0] = image_width - x2
        dets[:, 2] = image_width - x1
        restored.append(dets)
    return restored


def merge_scale_results(
    scale_results: Sequence[list[np.ndarray]],
    score_threshold: float,
    iou_threshold: float,
    max_detections_per_image: int,
) -> dict[str, torch.Tensor]:
    boxes = []
    scores = []
    labels = []

    for result in scale_results:
        for label_index, dets in enumerate(result):
            if dets is None:
                continue
            dets = np.asarray(dets)
            if dets.size == 0:
                continue
            keep = dets[:, 4] >= score_threshold
            if not np.any(keep):
                continue
            dets = dets[keep]
            boxes.append(torch.as_tensor(dets[:, :4], dtype=torch.float32))
            scores.append(torch.as_tensor(dets[:, 4], dtype=torch.float32))
            labels.append(torch.full((dets.shape[0],), label_index, dtype=torch.int64))

    if not boxes:
        empty_boxes = torch.empty((0, 4), dtype=torch.float32)
        empty_scores = torch.empty((0,), dtype=torch.float32)
        empty_labels = torch.empty((0,), dtype=torch.int64)
        return {"boxes_xyxy": empty_boxes, "scores": empty_scores, "labels": empty_labels}

    boxes_xyxy = torch.cat(boxes, dim=0)
    scores = torch.cat(scores, dim=0)
    labels = torch.cat(labels, dim=0)
    # Avoid relying on mmcv/torchvision batched_nms API differences by applying
    # the standard coordinate-offset trick with plain torchvision NMS.
    max_coordinate = boxes_xyxy.max()
    offsets = labels.to(boxes_xyxy) * (max_coordinate + 1)
    keep = nms(boxes_xyxy + offsets[:, None], scores, iou_threshold)
    keep = keep[:max_detections_per_image]
    return {
        "boxes_xyxy": boxes_xyxy[keep],
        "scores": scores[keep],
        "labels": labels[keep],
    }


def prediction_to_records(image_id: int, prediction: dict[str, torch.Tensor], category_ids: Sequence[int]) -> list[dict]:
    records = []
    for box, score, label in zip(
        prediction["boxes_xyxy"].cpu(),
        prediction["scores"].cpu(),
        prediction["labels"].cpu(),
    ):
        x1, y1, x2, y2 = box.tolist()
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        records.append(
            {
                "image_id": int(image_id),
                "category_id": int(category_ids[int(label.item())]),
                "bbox": [float(x1), float(y1), float(width), float(height)],
                "score": float(score.item()),
            }
        )
    return records


def remap_prediction_coordinates(
    prediction: dict[str, torch.Tensor],
    image_id: int,
    source_sizes: dict[int, tuple[int, int]],
    target_sizes: dict[int, tuple[int, int]],
    mode: str,
    pad_position: str,
) -> dict[str, torch.Tensor]:
    """Map prediction boxes from source-image coordinates to target coordinates."""

    if image_id not in source_sizes:
        raise KeyError(f"image_id={image_id} missing from source annotation file.")
    if image_id not in target_sizes:
        raise KeyError(f"image_id={image_id} missing from output-coordinate annotation file.")

    source_width, source_height = source_sizes[image_id]
    target_width, target_height = target_sizes[image_id]
    boxes = prediction["boxes_xyxy"].clone()

    if boxes.numel() == 0:
        return {**prediction, "boxes_xyxy": boxes}

    if mode == "direct-scale":
        scale_x = target_width / source_width
        scale_y = target_height / source_height
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
    elif mode == "square-pad":
        resize_scale = min(source_width / target_width, source_height / target_height)
        resized_width = round(target_width * resize_scale)
        resized_height = round(target_height * resize_scale)
        if pad_position == "center":
            offset_x = (source_width - resized_width) / 2.0
            offset_y = (source_height - resized_height) / 2.0
        else:
            offset_x = 0.0
            offset_y = 0.0
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - offset_x) / resize_scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - offset_y) / resize_scale
    else:
        raise ValueError(f"Unsupported coordinate map mode: {mode}")

    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, target_width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, target_height)
    boxes[:, 2] = torch.maximum(boxes[:, 2], boxes[:, 0])
    boxes[:, 3] = torch.maximum(boxes[:, 3], boxes[:, 1])
    return {**prediction, "boxes_xyxy": boxes}


def write_submission(records: list[dict], output_json: str | Path, zip_output: bool) -> None:
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(records, handle)

    if zip_output:
        zip_path = output_json.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(output_json, arcname="pred.json")


def run_inference_batch(model, batch_inputs: Sequence[str | np.ndarray], scale: int) -> list[list[np.ndarray]]:
    set_test_scale(model.cfg, scale)
    outputs = inference_detector(model, list(batch_inputs))
    if not isinstance(outputs, list):
        outputs = [outputs]
    return [normalize_single_result(output) for output in outputs]


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    config = mmcv.Config.fromfile(args.config)
    if args.cfg_options:
        config.merge_from_dict(args.cfg_options)

    image_dir = Path(args.image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    samples = load_samples(image_dir=image_dir)
    if args.max_images is not None:
        samples = samples[: args.max_images]
    if not samples:
        raise ValueError(f"No images found in {image_dir}")

    class_names = resolve_class_names(config)
    category_ids = resolve_category_ids(
        class_names=class_names,
        categories=None,
        explicit_category_ids=args.category_ids,
    )
    scales = args.tta_scales if args.tta_scales is not None else extract_test_scales(config)

    source_image_sizes = None
    output_image_sizes = None
    if args.output_coordinate_image_dir is not None:
        source_image_sizes = load_image_sizes_from_samples(samples)
        output_dir = Path(args.output_coordinate_image_dir)
        if not output_dir.is_dir():
            raise FileNotFoundError(f"Output-coordinate image directory does not exist: {output_dir}")
        output_image_sizes = load_image_sizes_from_dir(output_dir)
        print(
            "Export boxes will be remapped from inference-image coordinates "
            f"({args.image_dir}) to output coordinates ({args.output_coordinate_image_dir}) "
            f"with mode={args.coordinate_map_mode}."
        )

    model = init_detector(
        config=config,
        checkpoint=args.checkpoint,
        device=args.device,
        cfg_options=args.cfg_options,
    )

    if args.tta_hflip:
        print("Horizontal flip TTA will run as manual image flipping plus box unflipping/merging.")

    all_records: list[dict] = []
    total = len(samples)
    for batch_index, batch in enumerate(chunked(samples, args.batch_size), start=1):
        batch_paths = [str(path) for _, path in batch]
        if args.tta_hflip:
            batch_images = [mmcv.imread(path) for path in batch_paths]
            flipped_batch_images = [horizontal_flip_image(image) for image in batch_images]
            image_widths = [int(image.shape[1]) for image in batch_images]
            batch_inputs: Sequence[str | np.ndarray] = batch_images
        else:
            batch_inputs = batch_paths
        per_scale_outputs = []
        for scale in scales:
            original_outputs = run_inference_batch(
                model=model,
                batch_inputs=batch_inputs,
                scale=scale,
            )
            per_scale_outputs.append(original_outputs)
            if args.tta_hflip:
                flipped_outputs = run_inference_batch(
                    model=model,
                    batch_inputs=flipped_batch_images,
                    scale=scale,
                )
                flipped_outputs = [
                    unflip_single_result(output, image_width)
                    for output, image_width in zip(flipped_outputs, image_widths)
                ]
                per_scale_outputs.append(flipped_outputs)

        for item_index, (image_id, _) in enumerate(batch):
            scale_results = [outputs[item_index] for outputs in per_scale_outputs]
            merged = merge_scale_results(
                scale_results=scale_results,
                score_threshold=args.score_threshold,
                iou_threshold=args.nms_iou_threshold,
                max_detections_per_image=args.max_detections_per_image,
            )
            if source_image_sizes is not None and output_image_sizes is not None:
                merged = remap_prediction_coordinates(
                    prediction=merged,
                    image_id=image_id,
                    source_sizes=source_image_sizes,
                    target_sizes=output_image_sizes,
                    mode=args.coordinate_map_mode,
                    pad_position=args.coordinate_pad_position,
                )
            all_records.extend(prediction_to_records(image_id=image_id, prediction=merged, category_ids=category_ids))

        processed = min(batch_index * args.batch_size, total)
        if processed % args.progress_every == 0 or processed == total:
            print(f"Processed {processed}/{total} images")

    write_submission(all_records, output_json=args.output_json, zip_output=args.zip_output)
    print(f"Wrote {len(all_records)} predictions to {args.output_json}")


if __name__ == "__main__":
    main()
