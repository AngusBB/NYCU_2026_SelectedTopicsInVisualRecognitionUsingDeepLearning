#!/usr/bin/env python3
"""Merge one or more submission files with mask-IoU NMS."""

from __future__ import annotations

import argparse
from pathlib import Path

from prediction_utils import (load_predictions, mask_nms_predictions,
                              write_predictions, write_submission_zip)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        help="Input JSON/ZIP prediction files.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zip-output", type=Path, default=None)
    parser.add_argument("--score-thr", type=float, default=0.01)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--max-per-img", type=int, default=1000)
    parser.add_argument("--pre-nms-topk", type=int, default=1500)
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional per-input score weights. The number of weights must "
            "match the number of input files. Example: --weights 1.0 0.8 0.8"))
    parser.add_argument(
        "--no-score-clip",
        action="store_true",
        help="Do not clip weighted scores to the [0, 1] range.")
    return parser.parse_args()


def _validate_weights(args: argparse.Namespace) -> list[float]:
    if args.weights is None:
        return [1.0] * len(args.inputs)

    if len(args.weights) != len(args.inputs):
        raise ValueError(
            f"--weights expects {len(args.inputs)} values, got "
            f"{len(args.weights)}.")

    if any(weight < 0 for weight in args.weights):
        raise ValueError("--weights values must be non-negative.")

    return [float(weight) for weight in args.weights]


def _apply_score_weight(predictions: list[dict], weight: float,
                        clip_score: bool) -> list[dict]:
    weighted = []
    for prediction in predictions:
        item = dict(prediction)
        score = float(item.get("score", 0.0)) * weight
        if clip_score:
            score = min(1.0, max(0.0, score))
        item["score"] = score
        weighted.append(item)
    return weighted


def main() -> None:
    args = parse_args()
    weights = _validate_weights(args)

    predictions = []
    for input_path, weight in zip(args.inputs, weights):
        loaded = load_predictions(input_path)
        weighted = _apply_score_weight(
            loaded, weight, clip_score=not args.no_score_clip)
        predictions.extend(weighted)
        print(
            f"Loaded {len(loaded)} predictions from {input_path} "
            f"(weight={weight:g})")

    merged = mask_nms_predictions(
        predictions,
        score_thr=args.score_thr,
        nms_iou_thr=args.nms_iou,
        max_per_img=args.max_per_img,
        pre_nms_topk=args.pre_nms_topk)

    write_predictions(args.output, merged)
    print(
        f"Wrote {len(merged)} merged predictions to {args.output} "
        f"(score_thr={args.score_thr}, nms_iou={args.nms_iou}, "
        f"weights={','.join(f'{weight:g}' for weight in weights)})")

    if args.zip_output is not None:
        write_submission_zip(args.output, args.zip_output)
        print(f"Wrote submission zip to {args.zip_output}")


if __name__ == "__main__":
    main()
