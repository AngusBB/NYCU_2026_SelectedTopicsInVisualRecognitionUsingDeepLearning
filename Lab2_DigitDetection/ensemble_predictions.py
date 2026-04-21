#!/usr/bin/env python3
"""Ensemble homework-style prediction files from JSON and ZIP submissions."""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Prediction files to ensemble. Each input can be a pred.json or a ZIP containing pred.json.",
    )
    parser.add_argument("--output-json", type=str, default="pred_ensemble.json")
    parser.add_argument("--zip-output", action="store_true")
    parser.add_argument("--weights", nargs="+", type=float, default=None)
    parser.add_argument("--method", choices=("wbf", "nms"), default="wbf")
    parser.add_argument("--iou-threshold", type=float, default=0.55)
    parser.add_argument("--skip-box-threshold", type=float, default=0.001)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--max-detections-per-image", type=int, default=300)
    parser.add_argument(
        "--score-aggregation",
        choices=("avg", "max"),
        default="avg",
        help="How to aggregate cluster scores for WBF.",
    )
    return parser.parse_args()


def load_prediction_records(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Prediction file does not exist: {path}")

    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as archive:
            json_members = [name for name in archive.namelist() if name.endswith(".json")]
            if "pred.json" in json_members:
                member = "pred.json"
            elif len(json_members) == 1:
                member = json_members[0]
            else:
                raise ValueError(f"Could not uniquely resolve a JSON file inside {path}")
            with archive.open(member, "r") as handle:
                return json.load(handle)

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_submission(records: list[dict], output_json: str | Path, zip_output: bool) -> None:
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(records, handle)

    if zip_output:
        zip_path = output_json.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(output_json, arcname="pred.json")


def bbox_xywh_to_xyxy(box: Sequence[float]) -> list[float]:
    x, y, w, h = [float(value) for value in box]
    return [x, y, x + max(0.0, w), y + max(0.0, h)]


def bbox_xyxy_to_xywh(box: Sequence[float]) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def iou_xyxy(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def group_records_by_image(records: Iterable[dict]) -> dict[int, list[dict]]:
    grouped: DefaultDict[int, list[dict]] = defaultdict(list)
    for record in records:
        grouped[int(record["image_id"])].append(record)
    return dict(grouped)


def prefilter_records(records: Iterable[dict], min_score: float) -> list[dict]:
    filtered = []
    for record in records:
        score = float(record["score"])
        if score < min_score:
            continue
        filtered.append(
            {
                "image_id": int(record["image_id"]),
                "category_id": int(record["category_id"]),
                "bbox_xyxy": bbox_xywh_to_xyxy(record["bbox"]),
                "score": score,
            }
        )
    return filtered


def ensemble_image_nms(
    image_records: Sequence[Sequence[dict]],
    weights: Sequence[float],
    iou_threshold: float,
    score_threshold: float,
    max_detections_per_image: int,
) -> list[dict]:
    merged: DefaultDict[int, list[dict]] = defaultdict(list)
    for model_index, model_records in enumerate(image_records):
        weight = float(weights[model_index])
        for record in model_records:
            candidate = {
                "category_id": int(record["category_id"]),
                "bbox_xyxy": list(record["bbox_xyxy"]),
                "score": float(record["score"]) * weight,
            }
            merged[candidate["category_id"]].append(candidate)

    kept: list[dict] = []
    for category_id, candidates in merged.items():
        candidates.sort(key=lambda item: item["score"], reverse=True)
        selected: list[dict] = []
        while candidates:
            current = candidates.pop(0)
            selected.append(current)
            candidates = [
                candidate
                for candidate in candidates
                if iou_xyxy(current["bbox_xyxy"], candidate["bbox_xyxy"]) < iou_threshold
            ]
        for candidate in selected:
            if candidate["score"] >= score_threshold:
                kept.append(
                    {
                        "category_id": category_id,
                        "bbox_xyxy": candidate["bbox_xyxy"],
                        "score": candidate["score"],
                    }
                )

    kept.sort(key=lambda item: item["score"], reverse=True)
    kept = kept[:max_detections_per_image]
    return kept


def update_wbf_cluster(cluster: dict) -> None:
    total = cluster["coord_weight_sum"]
    cluster["bbox_xyxy"] = [
        cluster["coord_sum"][0] / total,
        cluster["coord_sum"][1] / total,
        cluster["coord_sum"][2] / total,
        cluster["coord_sum"][3] / total,
    ]


def ensemble_image_wbf(
    image_records: Sequence[Sequence[dict]],
    weights: Sequence[float],
    iou_threshold: float,
    score_threshold: float,
    max_detections_per_image: int,
    score_aggregation: str,
) -> list[dict]:
    grouped: DefaultDict[int, list[dict]] = defaultdict(list)
    for model_index, model_records in enumerate(image_records):
        model_weight = float(weights[model_index])
        for record in model_records:
            grouped[int(record["category_id"])].append(
                {
                    "category_id": int(record["category_id"]),
                    "bbox_xyxy": list(record["bbox_xyxy"]),
                    "score": float(record["score"]),
                    "model_weight": model_weight,
                }
            )

    fused: list[dict] = []
    for category_id, candidates in grouped.items():
        candidates.sort(key=lambda item: item["score"] * item["model_weight"], reverse=True)
        clusters: list[dict] = []

        for candidate in candidates:
            best_cluster = None
            best_iou = 0.0
            for cluster in clusters:
                cluster_iou = iou_xyxy(cluster["bbox_xyxy"], candidate["bbox_xyxy"])
                if cluster_iou >= iou_threshold and cluster_iou > best_iou:
                    best_iou = cluster_iou
                    best_cluster = cluster

            coord_weight = candidate["score"] * candidate["model_weight"]
            if best_cluster is None:
                cluster = {
                    "category_id": category_id,
                    "members": 1,
                    "score_sum": candidate["score"] * candidate["model_weight"],
                    "score_max": candidate["score"] * candidate["model_weight"],
                    "coord_weight_sum": coord_weight,
                    "coord_sum": [
                        candidate["bbox_xyxy"][0] * coord_weight,
                        candidate["bbox_xyxy"][1] * coord_weight,
                        candidate["bbox_xyxy"][2] * coord_weight,
                        candidate["bbox_xyxy"][3] * coord_weight,
                    ],
                    "bbox_xyxy": list(candidate["bbox_xyxy"]),
                }
                clusters.append(cluster)
                continue

            best_cluster["members"] += 1
            best_cluster["score_sum"] += candidate["score"] * candidate["model_weight"]
            best_cluster["score_max"] = max(
                best_cluster["score_max"], candidate["score"] * candidate["model_weight"]
            )
            best_cluster["coord_weight_sum"] += coord_weight
            best_cluster["coord_sum"][0] += candidate["bbox_xyxy"][0] * coord_weight
            best_cluster["coord_sum"][1] += candidate["bbox_xyxy"][1] * coord_weight
            best_cluster["coord_sum"][2] += candidate["bbox_xyxy"][2] * coord_weight
            best_cluster["coord_sum"][3] += candidate["bbox_xyxy"][3] * coord_weight
            update_wbf_cluster(best_cluster)

        for cluster in clusters:
            if score_aggregation == "max":
                score = cluster["score_max"]
            else:
                score = cluster["score_sum"] / max(1, cluster["members"])

            if score < score_threshold:
                continue
            fused.append(
                {
                    "category_id": category_id,
                    "bbox_xyxy": list(cluster["bbox_xyxy"]),
                    "score": float(score),
                }
            )

    fused.sort(key=lambda item: item["score"], reverse=True)
    return fused[:max_detections_per_image]


def ensemble_records(
    grouped_by_model: Sequence[dict[int, list[dict]]],
    weights: Sequence[float],
    method: str,
    iou_threshold: float,
    score_threshold: float,
    max_detections_per_image: int,
    score_aggregation: str,
) -> list[dict]:
    all_image_ids = sorted({image_id for grouped in grouped_by_model for image_id in grouped})
    ensemble_output: list[dict] = []

    for image_id in all_image_ids:
        image_records = [grouped.get(image_id, []) for grouped in grouped_by_model]
        if method == "nms":
            merged = ensemble_image_nms(
                image_records=image_records,
                weights=weights,
                iou_threshold=iou_threshold,
                score_threshold=score_threshold,
                max_detections_per_image=max_detections_per_image,
            )
        else:
            merged = ensemble_image_wbf(
                image_records=image_records,
                weights=weights,
                iou_threshold=iou_threshold,
                score_threshold=score_threshold,
                max_detections_per_image=max_detections_per_image,
                score_aggregation=score_aggregation,
            )

        for record in merged:
            ensemble_output.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(record["category_id"]),
                    "bbox": bbox_xyxy_to_xywh(record["bbox_xyxy"]),
                    "score": float(record["score"]),
                }
            )

    ensemble_output.sort(key=lambda item: (int(item["image_id"]), -float(item["score"]), int(item["category_id"])))
    return ensemble_output


def validate_weights(weights: Sequence[float] | None, num_inputs: int) -> list[float]:
    if weights is None:
        return [1.0] * num_inputs
    if len(weights) != num_inputs:
        raise ValueError(f"Expected {num_inputs} weights, got {len(weights)}")
    return [float(weight) for weight in weights]


def main() -> None:
    args = parse_args()
    weights = validate_weights(args.weights, len(args.inputs))

    grouped_by_model = []
    total_input_records = 0
    for input_path in args.inputs:
        records = load_prediction_records(input_path)
        total_input_records += len(records)
        filtered = prefilter_records(records, min_score=float(args.skip_box_threshold))
        grouped_by_model.append(group_records_by_image(filtered))

    ensembled = ensemble_records(
        grouped_by_model=grouped_by_model,
        weights=weights,
        method=args.method,
        iou_threshold=float(args.iou_threshold),
        score_threshold=float(args.score_threshold),
        max_detections_per_image=int(args.max_detections_per_image),
        score_aggregation=args.score_aggregation,
    )
    write_submission(ensembled, output_json=args.output_json, zip_output=args.zip_output)

    print(f"Loaded {len(args.inputs)} files with {total_input_records} total predictions")
    print(f"Wrote {len(ensembled)} ensembled predictions to {args.output_json}")
    if args.zip_output:
        print(f"Wrote ZIP submission to {Path(args.output_json).with_suffix('.zip')}")


if __name__ == "__main__":
    main()
