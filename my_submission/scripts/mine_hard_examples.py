from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_CLASSES = ("chair", "backpack")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine class-specific hard-negative and hard-positive image weights."
    )
    parser.add_argument("--ground_truth", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--classes",
        default=",".join(DEFAULT_CLASSES),
        help="Comma-separated class names to mine; defaults to chair,backpack.",
    )
    parser.add_argument(
        "--class_score_thresholds",
        default="",
        help="Optional comma-separated class=threshold values, e.g. chair=0.20,backpack=0.20.",
    )
    parser.add_argument("--score_threshold", type=float, default=0.20)
    parser.add_argument("--topk", type=int, default=0)
    parser.add_argument("--negative_iou_threshold", type=float, default=0.10)
    parser.add_argument("--positive_match_iou", type=float, default=0.50)
    parser.add_argument("--max_weight", type=float, default=1.5)
    parser.add_argument("--hard_positive_boost", type=float, default=0.4)
    # Backward-compatible chair-specific flags used by the older script.
    parser.add_argument("--chair_score_threshold", type=float, default=-1.0)
    parser.add_argument("--chair_topk", type=int, default=-1)
    parser.add_argument("--backpack_score_threshold", type=float, default=-1.0)
    parser.add_argument("--backpack_topk", type=int, default=-1)
    return parser.parse_args()


def parse_class_list(raw: str) -> list[str]:
    classes = [value.strip() for value in raw.split(",") if value.strip()]
    if not classes:
        raise ValueError("--classes must contain at least one class name")
    return list(dict.fromkeys(classes))


def parse_class_values(raw: str, default: float) -> dict[str, float]:
    values: dict[str, float] = {}
    if raw.strip():
        for item in raw.split(","):
            if not item.strip():
                continue
            if "=" not in item:
                raise ValueError(f"Expected class=value, got: {item}")
            class_name, value = item.split("=", 1)
            values[class_name.strip()] = float(value)
    return defaultdict(lambda: default, values)


def resolve_class_thresholds(args: argparse.Namespace, classes: list[str]) -> dict[str, float]:
    thresholds = parse_class_values(args.class_score_thresholds, args.score_threshold)
    if args.chair_score_threshold >= 0:
        thresholds["chair"] = args.chair_score_threshold
    if args.backpack_score_threshold >= 0:
        thresholds["backpack"] = args.backpack_score_threshold
    return {class_name: float(thresholds[class_name]) for class_name in classes}


def resolve_class_topk(args: argparse.Namespace, classes: list[str]) -> dict[str, int]:
    topk = {class_name: args.topk for class_name in classes}
    if args.chair_topk >= 0:
        topk["chair"] = args.chair_topk
    if args.backpack_topk >= 0:
        topk["backpack"] = args.backpack_topk
    return topk


def mine_hard_examples(
    ground_truth: dict[str, Any],
    predictions: list[dict[str, Any]],
    classes: list[str],
    class_score_thresholds: dict[str, float],
    class_topk: dict[str, int],
    negative_iou_threshold: float = 0.10,
    positive_match_iou: float = 0.50,
    max_weight: float = 1.5,
    hard_positive_boost: float = 0.4,
) -> dict[str, Any]:
    all_image_ids = [image["id"] for image in ground_truth.get("images", [])]
    gt_by_class: dict[str, dict[str, list[list[float]]]] = {
        class_name: defaultdict(list) for class_name in classes
    }
    for annotation in ground_truth.get("annotations", []):
        class_name = annotation.get("class")
        if class_name in gt_by_class:
            gt_by_class[class_name][annotation["image_id"]].append(annotation["bbox"])

    predictions_by_image = {item["image_id"]: item for item in predictions}
    image_weights = {image_id: 1.0 for image_id in all_image_ids}
    hard_negatives: list[dict[str, Any]] = []
    hard_positives: list[dict[str, Any]] = []
    class_summaries = {
        class_name: {
            "num_ground_truth_images": sum(
                bool(gt_by_class[class_name].get(image_id, [])) for image_id in all_image_ids
            ),
            "num_hard_negative_images": 0,
            "num_hard_positive_images": 0,
        }
        for class_name in classes
    }

    for image_id in all_image_ids:
        item = predictions_by_image.get(image_id, {"boxes": []})
        for class_name in classes:
            gt_boxes = gt_by_class[class_name].get(image_id, [])
            class_preds = [
                box for box in item.get("boxes", []) if box.get("class") == class_name
            ]
            class_preds.sort(key=lambda box: float(box.get("confidence", 0.0)), reverse=True)
            threshold = class_score_thresholds[class_name]
            topk = class_topk[class_name]
            selected = [
                box
                for rank, box in enumerate(class_preds)
                if (topk > 0 and rank < topk) or float(box.get("confidence", 0.0)) >= threshold
            ]

            false_scores: list[float] = []
            matched_gt = [False] * len(gt_boxes)
            for pred in selected:
                pred_box = pred["bbox"]
                overlaps = [iou(pred_box, gt_box) for gt_box in gt_boxes]
                max_iou = max(overlaps, default=0.0)
                if max_iou >= positive_match_iou and overlaps:
                    best_index = overlaps.index(max_iou)
                    matched_gt[best_index] = True
                if max_iou < negative_iou_threshold:
                    false_scores.append(float(pred.get("confidence", 0.0)))

            if false_scores:
                max_false_score = max(false_scores)
                weight = min(max_weight, 1.0 + min(1.0, max_false_score))
                image_weights[image_id] = max(image_weights[image_id], weight)
                hard_negatives.append(
                    {
                        "image_id": image_id,
                        "class": class_name,
                        "max_false_score": max_false_score,
                        "num_false_predictions": len(false_scores),
                        "weight": image_weights[image_id],
                    }
                )
                class_summaries[class_name]["num_hard_negative_images"] += 1

            if gt_boxes and not all(matched_gt):
                weight = min(max_weight, image_weights[image_id] + hard_positive_boost)
                image_weights[image_id] = max(image_weights[image_id], weight)
                hard_positives.append(
                    {
                        "image_id": image_id,
                        "class": class_name,
                        "num_gt": len(gt_boxes),
                        "num_missed_gt": matched_gt.count(False),
                        "weight": image_weights[image_id],
                    }
                )
                class_summaries[class_name]["num_hard_positive_images"] += 1

    summary = {
        "num_images": len(image_weights),
        "num_hard_negative_records": len(hard_negatives),
        "num_hard_positive_records": len(hard_positives),
        "max_weight": max(image_weights.values(), default=1.0),
        "mean_weight": sum(image_weights.values()) / max(len(image_weights), 1),
        "classes": class_summaries,
    }
    return {
        "image_weights": image_weights,
        "hard_negatives": hard_negatives,
        "hard_positives": hard_positives,
        "summary": summary,
    }


def main() -> None:
    args = parse_args()
    classes = parse_class_list(args.classes)
    thresholds = resolve_class_thresholds(args, classes)
    topk = resolve_class_topk(args, classes)
    ground_truth = json.loads(Path(args.ground_truth).read_text(encoding="utf-8"))
    predictions = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    output = mine_hard_examples(
        ground_truth,
        predictions,
        classes,
        thresholds,
        topk,
        negative_iou_threshold=args.negative_iou_threshold,
        positive_match_iou=args.positive_match_iou,
        max_weight=args.max_weight,
        hard_positive_boost=args.hard_positive_boost,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], indent=2))


def iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-6)


if __name__ == "__main__":
    main()
