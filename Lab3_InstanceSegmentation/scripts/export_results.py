#!/usr/bin/env python3
"""Run MMDetection inference and export CodaBench JSON results."""

from __future__ import annotations

import argparse
from pathlib import Path

from prediction_utils import (bbox_xyxy_to_xywh, dump_json_without_reserved_token,
                              iter_predictions, load_config_with_overrides,
                              load_test_images, normalize_rle,
                              patch_torch_load_for_trusted_mmengine_checkpoint,
                              resolve_checkpoint, write_submission_zip)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("./data/processed"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./test-results.json"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-thr", type=float, default=0.05)
    parser.add_argument("--mask-thr-binary", type=float, default=None)
    parser.add_argument("--rcnn-nms-iou", type=float, default=None)
    parser.add_argument("--rpn-nms-pre", type=int, default=None)
    parser.add_argument("--rpn-max-per-img", type=int, default=None)
    parser.add_argument("--rcnn-max-per-img", type=int, default=None)
    parser.add_argument(
        "--test-scale",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Optional inference resize scale, e.g. --test-scale 1600 960.")
    parser.add_argument(
        "--limit-images",
        type=int,
        default=None,
        help="Optional smoke-test limit on the number of test images.")
    parser.add_argument(
        "--zip-output",
        type=Path,
        default=None,
        help="Optional zip file containing test-results.json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from mmdet.apis import inference_detector, init_detector

    checkpoint = resolve_checkpoint(args.checkpoint)
    cfg = load_config_with_overrides(
        args.config,
        mask_thr_binary=args.mask_thr_binary,
        rcnn_nms_iou=args.rcnn_nms_iou,
        rpn_nms_pre=args.rpn_nms_pre,
        rpn_max_per_img=args.rpn_max_per_img,
        rcnn_max_per_img=args.rcnn_max_per_img,
        test_scale=tuple(args.test_scale) if args.test_scale is not None else None)
    torch, original_load = patch_torch_load_for_trusted_mmengine_checkpoint()
    try:
        model = init_detector(cfg, str(checkpoint), device=args.device)
    finally:
        torch.load = original_load

    predictions = []
    test_images = load_test_images(args.processed_root)
    if args.limit_images is not None:
        test_images = test_images[:args.limit_images]

    for image_info in test_images:
        image_path = args.processed_root / "images" / "test" / image_info["file_name"]
        result = inference_detector(model, str(image_path))

        for category_id, bbox, score, segmentation in iter_predictions(result):
            if score < args.score_thr:
                continue

            x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
            predictions.append(
                {
                    "image_id": int(image_info["id"]),
                    "category_id": int(category_id),
                    "bbox": bbox_xyxy_to_xywh([x1, y1, x2, y2]),
                    "score": score,
                    "segmentation": normalize_rle(segmentation),
                })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dump_json_without_reserved_token(args.output, predictions)
    print(f"Wrote {len(predictions)} predictions to {args.output}")

    if args.zip_output is not None:
        write_submission_zip(args.output, args.zip_output)
        print(f"Wrote submission zip to {args.zip_output}")


if __name__ == "__main__":
    main()
