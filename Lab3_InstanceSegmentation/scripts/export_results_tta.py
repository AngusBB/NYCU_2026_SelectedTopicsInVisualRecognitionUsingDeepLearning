#!/usr/bin/env python3
"""Run mask-safe manual TTA and export merged CodaBench predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from prediction_utils import (bbox_from_mask, iter_predictions,
                              load_config_with_overrides, load_test_images,
                              mask_nms_predictions, normalize_rle,
                              patch_torch_load_for_trusted_mmengine_checkpoint,
                              resolve_checkpoint, segmentation_to_mask,
                              write_predictions, write_submission_zip)

DEFAULT_CONFIG = Path(
    "./configs/"
    "htc_swin_b_copypaste_dense-rpn6000-rcnn1000.py")
DEFAULT_CHECKPOINT = Path(
    "./work_dirs/"
    "htc_swin_b_copypaste_all_random-erasing/epoch_30.pth")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("./data/processed"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./test-results-tta.json"))
    parser.add_argument("--zip-output", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-thr", type=float, default=0.01)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--max-per-img", type=int, default=1000)
    parser.add_argument("--pre-nms-topk", type=int, default=1500)
    parser.add_argument(
        "--views",
        default="orig,hflip,vflip,hvflip",
        help="Comma-separated views from: orig,hflip,vflip,hvflip.")
    parser.add_argument("--mask-thr-binary", type=float, default=None)
    parser.add_argument("--rcnn-nms-iou", type=float, default=None)
    parser.add_argument("--rpn-nms-pre", type=int, default=None)
    parser.add_argument("--rpn-max-per-img", type=int, default=None)
    parser.add_argument("--rcnn-max-per-img", type=int, default=None)
    parser.add_argument(
        "--limit-images",
        type=int,
        default=None,
        help="Optional smoke-test limit on the number of test images.")
    return parser.parse_args()


def parse_views(value: str) -> list[str]:
    valid = {"orig", "hflip", "vflip", "hvflip"}
    views = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(views) - valid)
    if unknown:
        raise ValueError(f"Unknown TTA view(s): {', '.join(unknown)}")
    if not views:
        raise ValueError("At least one TTA view is required.")
    return views


def load_bgr_image(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"))
    return rgb[:, :, ::-1].copy()


def apply_view(image: np.ndarray, view: str) -> np.ndarray:
    if view == "orig":
        return image
    if view == "hflip":
        return image[:, ::-1, :].copy()
    if view == "vflip":
        return image[::-1, :, :].copy()
    if view == "hvflip":
        return image[::-1, ::-1, :].copy()
    raise ValueError(f"Unknown TTA view: {view}")


def invert_view_mask(mask: np.ndarray, view: str) -> np.ndarray:
    if view == "orig":
        return mask
    if view == "hflip":
        return mask[:, ::-1].copy()
    if view == "vflip":
        return mask[::-1, :].copy()
    if view == "hvflip":
        return mask[::-1, ::-1].copy()
    raise ValueError(f"Unknown TTA view: {view}")


def main() -> None:
    args = parse_args()
    views = parse_views(args.views)

    from mmdet.apis import inference_detector, init_detector

    checkpoint = resolve_checkpoint(args.checkpoint)
    cfg = load_config_with_overrides(
        args.config,
        mask_thr_binary=args.mask_thr_binary,
        rcnn_nms_iou=args.rcnn_nms_iou,
        rpn_nms_pre=args.rpn_nms_pre,
        rpn_max_per_img=args.rpn_max_per_img,
        rcnn_max_per_img=args.rcnn_max_per_img)
    torch, original_load = patch_torch_load_for_trusted_mmengine_checkpoint()
    try:
        model = init_detector(cfg, str(checkpoint), device=args.device)
    finally:
        torch.load = original_load

    test_images = load_test_images(args.processed_root)
    if args.limit_images is not None:
        test_images = test_images[:args.limit_images]

    raw_predictions = []
    for image_index, image_info in enumerate(test_images, start=1):
        image_path = args.processed_root / "images" / "test" / image_info["file_name"]
        image = load_bgr_image(image_path)

        for view in views:
            result = inference_detector(model, apply_view(image, view))
            for category_id, _bbox, score, segmentation in iter_predictions(result):
                if score < args.score_thr:
                    continue

                mask = invert_view_mask(segmentation_to_mask(segmentation), view)
                bbox = bbox_from_mask(mask)
                if bbox is None:
                    continue

                raw_predictions.append({
                    "image_id": int(image_info["id"]),
                    "category_id": int(category_id),
                    "bbox": bbox,
                    "score": float(score),
                    "segmentation": normalize_rle(mask),
                })

        print(
            f"Processed {image_index}/{len(test_images)} images; "
            f"raw predictions={len(raw_predictions)}")

    merged = mask_nms_predictions(
        raw_predictions,
        score_thr=args.score_thr,
        nms_iou_thr=args.nms_iou,
        max_per_img=args.max_per_img,
        pre_nms_topk=args.pre_nms_topk)

    write_predictions(args.output, merged)
    print(
        f"Wrote {len(merged)} TTA predictions to {args.output} "
        f"(views={','.join(views)}, score_thr={args.score_thr}, "
        f"nms_iou={args.nms_iou})")

    if args.zip_output is not None:
        write_submission_zip(args.output, args.zip_output)
        print(f"Wrote submission zip to {args.zip_output}")


if __name__ == "__main__":
    main()
