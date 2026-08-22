"""Analyze prediction overload and false-positive types on a train split.

This tool is read-only and is intended for train annotations plus predictions generated
on the same train images. It ranks images with the most retained predictions, labels
each prediction against ground truth, and separates false positives into:

* duplicate: overlaps a higher-scoring prediction of the same class;
* class_confusion: overlaps a ground-truth object, but with the wrong class;
* background: does not sufficiently overlap any ground-truth object.

It does not change annotations or predictions and it never uses hidden-test labels.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

EXPECTED_CLASSES = ("bottle", "cup", "chair", "laptop", "backpack")
TYPE_COLORS = {
    "true_positive": "#00c853",
    "duplicate": "#ff9800",
    "background": "#f44336",
    "class_confusion": "#9c27b0",
    "invalid": "#607d8b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank prediction-overloaded train images and classify false positives.")
    parser.add_argument("--annotations", required=True, help="Train annotation JSON only.")
    parser.add_argument("--predictions", required=True, help="Predictions generated on the same train images.")
    parser.add_argument("--image-dir", required=True, help="Train image directory used for review renders.")
    parser.add_argument("--output-dir", required=True, help="Output directory for diagnostics and renders.")
    parser.add_argument("--topk", type=int, default=30, help="Number of overload images to render and report.")
    parser.add_argument("--score-threshold", type=float, default=0.0, help="Ignore predictions below this score.")
    parser.add_argument("--match-iou", type=float, default=0.50, help="IoU needed for same-class true-positive matching.")
    parser.add_argument("--duplicate-iou", type=float, default=0.45, help="IoU with a higher-scoring same-class prediction for duplicate labeling.")
    parser.add_argument("--high-score", type=float, default=0.20, help="Score used for high-confidence FP counts.")
    return parser.parse_args()


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    return intersection / max(area(first) + area(second) - intersection, 1e-12)


def image_path(image_info: dict[str, Any], image_dir: Path) -> Path | None:
    image_id = str(image_info.get("id", ""))
    direct = image_dir / image_id
    if direct.exists():
        return direct
    file_name = Path(str(image_info.get("file_name", ""))).name
    fallback = image_dir / file_name
    return fallback if fallback.exists() else None


def load_predictions(path: Path, score_threshold: float) -> dict[str, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("predictions", payload.get("images", []))
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not isinstance(payload, list):
        return result
    for image_item in payload:
        if not isinstance(image_item, dict):
            continue
        image_id = str(image_item.get("image_id", ""))
        for prediction in image_item.get("boxes", []) or []:
            if not isinstance(prediction, dict):
                continue
            confidence = as_float(prediction.get("confidence"))
            raw_bbox = prediction.get("bbox")
            if confidence is None or confidence < score_threshold or not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
                continue
            values = [as_float(value) for value in raw_bbox]
            if any(value is None for value in values):
                continue
            result[image_id].append(
                {
                    "class": str(prediction.get("class", "")),
                    "confidence": float(confidence),
                    "bbox": [float(value) for value in values],
                }
            )
    for boxes in result.values():
        boxes.sort(key=lambda item: (-item["confidence"], item["class"], item["bbox"]))
    return result


def load_ground_truth(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    image_by_id = {str(item.get("id", "")): item for item in data.get("images", []) if isinstance(item, dict)}
    gt_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for annotation in data.get("annotations", []):
        if not isinstance(annotation, dict):
            continue
        raw_bbox = annotation.get("bbox")
        if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
            continue
        values = [as_float(value) for value in raw_bbox]
        if any(value is None for value in values):
            continue
        bbox = [float(value) for value in values]
        gt_by_image[str(annotation.get("image_id", ""))].append(
            {"class": str(annotation.get("class", "")), "bbox": bbox}
        )
    return image_by_id, gt_by_image


def classify_predictions(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    match_iou: float,
    duplicate_iou: float,
) -> list[dict[str, Any]]:
    # Predictions are already sorted by descending confidence. This greedy one-to-one
    # matching mirrors the standard detection evaluator: one GT cannot produce two TPs.
    matched_gt = [False] * len(ground_truth)
    labeled: list[dict[str, Any]] = []
    for index, prediction in enumerate(predictions):
        bbox = prediction["bbox"]
        class_name = prediction["class"]
        same_scores = [
            (iou(bbox, item["bbox"]), gt_index, item)
            for gt_index, item in enumerate(ground_truth)
            if item["class"] == class_name and not matched_gt[gt_index]
        ]
        all_scores = [(iou(bbox, item["bbox"]), gt_index, item) for gt_index, item in enumerate(ground_truth)]
        same_iou, same_index, same_match = max(same_scores, default=(0.0, -1, None), key=lambda pair: pair[0])
        any_iou, any_index, any_match = max(all_scores, default=(0.0, -1, None), key=lambda pair: pair[0])
        if class_name in EXPECTED_CLASSES and same_iou >= match_iou:
            kind = "true_positive"
            matched_gt[same_index] = True
        else:
            has_higher_same_class = any(
                previous["class"] == class_name
                and previous["confidence"] >= prediction["confidence"]
                and iou(previous["bbox"], bbox) >= duplicate_iou
                for previous in predictions[:index]
            )
            if has_higher_same_class:
                kind = "duplicate"
            elif any_iou >= match_iou and any_match is not None:
                kind = "class_confusion"
            else:
                kind = "background"
            if class_name not in EXPECTED_CLASSES:
                kind = "invalid"
        labeled.append(
            {
                **prediction,
                "kind": kind,
                "same_class_iou": float(same_iou),
                "any_class_iou": float(any_iou),
                "matched_gt_class": same_match["class"] if same_match is not None else (any_match["class"] if any_match is not None else None),
            }
        )
    return labeled


def summarize_image(
    image_id: str,
    ground_truth: list[dict[str, Any]],
    labeled: list[dict[str, Any]],
    high_score: float,
) -> dict[str, Any]:
    counts = Counter(item["kind"] for item in labeled)
    fp_count = len(labeled) - counts["true_positive"]
    high_fp = sum(item["confidence"] >= high_score and item["kind"] != "true_positive" for item in labeled)
    total_score = sum(item["confidence"] for item in labeled)
    return {
        "image_id": image_id,
        "gt_count": len(ground_truth),
        "prediction_count": len(labeled),
        "true_positive_count": counts["true_positive"],
        "false_positive_count": fp_count,
        "duplicate_count": counts["duplicate"],
        "background_fp_count": counts["background"],
        "class_confusion_count": counts["class_confusion"],
        "invalid_count": counts["invalid"],
        "high_score_fp_count": high_fp,
        "max_prediction_score": max((item["confidence"] for item in labeled), default=0.0),
        "sum_prediction_scores": total_score,
        "background_fp_ratio": (counts["background"] / max(fp_count, 1)),
        "duplicate_fp_ratio": (counts["duplicate"] / max(fp_count, 1)),
    }


def render_image(
    image_info: dict[str, Any],
    image_dir: Path,
    output_path: Path,
    ground_truth: list[dict[str, Any]],
    labeled: list[dict[str, Any]],
    title: str,
) -> bool:
    source = image_path(image_info, image_dir)
    if source is None:
        return False
    try:
        with Image.open(source).convert("RGB") as original:
            scale = min(1.0, 1400.0 / max(original.width, original.height))
            width = max(1, int(round(original.width * scale)))
            height = max(1, int(round(original.height * scale)))
            image = original.resize((width, height), Image.Resampling.BILINEAR)
            draw = ImageDraw.Draw(image)
            line_width = max(2, int(round(3 * scale)))
            for item in ground_truth:
                x1, y1, x2, y2 = item["bbox"]
                draw.rectangle((x1 * scale, y1 * scale, x2 * scale, y2 * scale), outline="#00e676", width=line_width)
                draw.text((x1 * scale + 2, y1 * scale + 2), f"GT:{item['class']}", fill="#00e676")
            for item in labeled:
                x1, y1, x2, y2 = item["bbox"]
                kind = item["kind"]
                color = TYPE_COLORS[kind]
                draw.rectangle((x1 * scale, y1 * scale, x2 * scale, y2 * scale), outline=color, width=line_width)
                label = f"{kind}:{item['class']} {item['confidence']:.2f}"
                draw.text((x1 * scale + 2, max(0, y1 * scale - 14)), label, fill=color)
            draw.rectangle((0, 0, width, 22), fill="black")
            draw.text((4, 4), title, fill="white")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path, quality=92)
            return True
    except (OSError, ValueError):
        return False


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else ["image_id"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    annotation_path = Path(args.annotations).expanduser().resolve()
    prediction_path = Path(args.predictions).expanduser().resolve()
    image_dir = Path(args.image_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_by_id, gt_by_image = load_ground_truth(annotation_path)
    predictions_by_image = load_predictions(prediction_path, args.score_threshold)

    image_rows: list[dict[str, Any]] = []
    labeled_by_image: dict[str, list[dict[str, Any]]] = {}
    all_prediction_rows: list[dict[str, Any]] = []
    for image_id in image_by_id:
        labeled = classify_predictions(
            predictions_by_image.get(image_id, []),
            gt_by_image.get(image_id, []),
            args.match_iou,
            args.duplicate_iou,
        )
        labeled_by_image[image_id] = labeled
        image_rows.append(summarize_image(image_id, gt_by_image.get(image_id, []), labeled, args.high_score))
        for index, item in enumerate(labeled):
            all_prediction_rows.append(
                {
                    "image_id": image_id,
                    "prediction_index": index,
                    "class": item["class"],
                    "confidence": item["confidence"],
                    "xmin": item["bbox"][0],
                    "ymin": item["bbox"][1],
                    "xmax": item["bbox"][2],
                    "ymax": item["bbox"][3],
                    "kind": item["kind"],
                    "same_class_iou": item["same_class_iou"],
                    "any_class_iou": item["any_class_iou"],
                    "matched_gt_class": item["matched_gt_class"],
                }
            )

    ranked = sorted(
        image_rows,
        key=lambda row: (
            -row["prediction_count"],
            -row["false_positive_count"],
            -row["background_fp_count"],
            -row["max_prediction_score"],
            row["image_id"],
        ),
    )[: max(0, args.topk)]
    overload_ids = {row["image_id"] for row in ranked}
    overload_predictions = {
        image_id: labeled_by_image[image_id]
        for image_id in overload_ids
    }
    (output_dir / "overload_predictions.json").write_text(
        json.dumps(overload_predictions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(output_dir / "overload_diagnostics.csv", ranked)
    write_csv(output_dir / "all_prediction_diagnostics.csv", all_prediction_rows)

    if not getattr(args, "no_render", False):
        for row in ranked:
            image_id = row["image_id"]
            info = image_by_id.get(image_id)
            if info is None:
                continue
            safe_stem = Path(image_id).stem.replace("/", "_").replace("\\", "_")
            render_image(
                info,
                image_dir,
                output_dir / "overload_top30" / f"{safe_stem}.jpg",
                gt_by_image.get(image_id, []),
                labeled_by_image.get(image_id, []),
                f"overload | {image_id} | n={row['prediction_count']} bg={row['background_fp_count']} dup={row['duplicate_count']}",
            )

    totals = Counter()
    total_predictions = 0
    total_gt = 0
    for row in image_rows:
        total_gt += row["gt_count"]
        total_predictions += row["prediction_count"]
        for key in ("true_positive_count", "false_positive_count", "duplicate_count", "background_fp_count", "class_confusion_count", "invalid_count", "high_score_fp_count"):
            totals[key] += row[key]
    fp_total = totals["false_positive_count"]
    summary = {
        "analysis_version": 1,
        "read_only": True,
        "source_annotations": str(annotation_path),
        "source_predictions": str(prediction_path),
        "source_image_dir": str(image_dir),
        "score_threshold": args.score_threshold,
        "match_iou": args.match_iou,
        "duplicate_iou": args.duplicate_iou,
        "high_score": args.high_score,
        "images_analyzed": len(image_by_id),
        "ground_truth_boxes": total_gt,
        "predictions_analyzed": total_predictions,
        "true_positive_count": totals["true_positive_count"],
        "false_positive_count": totals["false_positive_count"],
        "duplicate_count": totals["duplicate_count"],
        "background_fp_count": totals["background_fp_count"],
        "class_confusion_count": totals["class_confusion_count"],
        "invalid_count": totals["invalid_count"],
        "high_score_fp_count": totals["high_score_fp_count"],
        "duplicate_fp_ratio": totals["duplicate_count"] / max(fp_total, 1),
        "background_fp_ratio": totals["background_fp_count"] / max(fp_total, 1),
        "class_confusion_fp_ratio": totals["class_confusion_count"] / max(fp_total, 1),
        "topk": args.topk,
        "overload_images": [row["image_id"] for row in ranked],
        "overload_csv": str(output_dir / "overload_diagnostics.csv"),
        "prediction_csv": str(output_dir / "all_prediction_diagnostics.csv"),
        "overload_render_dir": str(output_dir / "overload_top30"),
        "warning": "Labels are diagnostic categories, not automatic annotation edits.",
    }
    (output_dir / "overload_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = analyze(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
