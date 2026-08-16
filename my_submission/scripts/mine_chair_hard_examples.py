from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine chair hard-negative and hard-positive image weights.")
    parser.add_argument("--ground_truth", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chair_score_threshold", type=float, default=0.20)
    parser.add_argument("--chair_topk", type=int, default=0)
    parser.add_argument("--negative_iou_threshold", type=float, default=0.10)
    parser.add_argument("--positive_match_iou", type=float, default=0.50)
    parser.add_argument("--max_weight", type=float, default=1.5)
    parser.add_argument("--hard_positive_boost", type=float, default=0.4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt = json.loads(Path(args.ground_truth).read_text(encoding="utf-8"))
    predictions = json.loads(Path(args.predictions).read_text(encoding="utf-8"))

    chair_gt_by_image = defaultdict(list)
    all_image_ids = []
    for image in gt["images"]:
        all_image_ids.append(image["id"])
    for annotation in gt["annotations"]:
        if annotation["class"] == "chair":
            chair_gt_by_image[annotation["image_id"]].append(annotation["bbox"])

    image_weights = {image_id: 1.0 for image_id in all_image_ids}
    hard_negatives = []
    hard_positives = []
    for item in predictions:
        image_id = item["image_id"]
        gt_boxes = chair_gt_by_image.get(image_id, [])
        chair_preds = [
            box
            for box in item.get("boxes", [])
            if box.get("class") == "chair"
        ]
        chair_preds.sort(key=lambda box: float(box.get("confidence", 0.0)), reverse=True)
        selected = [
            box
            for rank, box in enumerate(chair_preds)
            if rank < args.chair_topk or float(box.get("confidence", 0.0)) >= args.chair_score_threshold
        ]

        false_scores = []
        matched_gt = [False] * len(gt_boxes)
        for pred in selected:
            pred_box = pred["bbox"]
            overlaps = [iou(pred_box, gt_box) for gt_box in gt_boxes]
            max_iou = max(overlaps, default=0.0)
            if max_iou >= args.positive_match_iou and overlaps:
                matched_gt[overlaps.index(max_iou)] = True
            if max_iou < args.negative_iou_threshold:
                false_scores.append(float(pred.get("confidence", 0.0)))

        if false_scores:
            max_false_score = max(false_scores)
            weight = min(args.max_weight, 1.0 + min(1.0, max_false_score))
            image_weights[image_id] = max(image_weights.get(image_id, 1.0), weight)
            hard_negatives.append(
                {
                    "image_id": image_id,
                    "max_false_chair_score": max_false_score,
                    "num_false_chair": len(false_scores),
                    "weight": image_weights[image_id],
                }
            )

        if gt_boxes and not all(matched_gt):
            weight = min(args.max_weight, image_weights.get(image_id, 1.0) + args.hard_positive_boost)
            image_weights[image_id] = max(image_weights.get(image_id, 1.0), weight)
            hard_positives.append(
                {
                    "image_id": image_id,
                    "num_chair_gt": len(gt_boxes),
                    "num_missed_chair_gt": matched_gt.count(False),
                    "weight": image_weights[image_id],
                }
            )

    output = {
        "image_weights": image_weights,
        "hard_negatives": hard_negatives,
        "hard_positives": hard_positives,
        "summary": {
            "num_images": len(image_weights),
            "num_hard_negative_images": len(hard_negatives),
            "num_hard_positive_images": len(hard_positives),
            "max_weight": max(image_weights.values(), default=1.0),
            "mean_weight": sum(image_weights.values()) / max(len(image_weights), 1),
        },
    }
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
