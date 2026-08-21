from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch


def pair_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-6)


def weighted_boxes_fusion(
    detections: list[dict],
    iou_threshold: float = 0.55,
    skip_box_threshold: float = 0.0,
    model_weight: float = 1.0,
) -> list[dict]:
    """Class-aware WBF for detections already in the original-image coordinate system."""
    clusters: dict[str, list[dict]] = defaultdict(list)
    for det in detections:
        class_name = str(det["class"])
        confidence = float(det["confidence"])
        if confidence >= skip_box_threshold:
            clusters[class_name].append(
                {
                    "class": class_name,
                    "confidence": confidence,
                    "bbox": [float(value) for value in det["bbox"]],
                    "weight": float(model_weight),
                }
            )

    fused: list[dict] = []
    for class_name, class_detections in clusters.items():
        class_detections.sort(key=lambda item: item["confidence"], reverse=True)
        class_clusters: list[dict] = []
        for det in class_detections:
            best_index = -1
            best_iou = iou_threshold
            for index, cluster in enumerate(class_clusters):
                overlap = pair_iou(det["bbox"], cluster["bbox"])
                if overlap >= best_iou:
                    best_iou = overlap
                    best_index = index
            if best_index < 0:
                class_clusters.append(
                    {
                        "class": class_name,
                        "bbox": list(det["bbox"]),
                        "score_sum": det["confidence"] * det["weight"],
                        "weight_sum": det["weight"],
                        "confidence_sum": det["confidence"] * det["weight"],
                        "members": 1,
                    }
                )
                continue

            cluster = class_clusters[best_index]
            old_weight = cluster["score_sum"]
            new_weight = det["confidence"] * det["weight"]
            total = old_weight + new_weight
            cluster["bbox"] = [
                (old * old_weight + new * new_weight) / max(total, 1e-6)
                for old, new in zip(cluster["bbox"], det["bbox"])
            ]
            cluster["score_sum"] = total
            cluster["weight_sum"] += det["weight"]
            cluster["confidence_sum"] += new_weight
            cluster["members"] += 1

        for cluster in class_clusters:
            fused.append(
                {
                    "class": class_name,
                    "confidence": float(
                        max(0.0, min(1.0, cluster["confidence_sum"] / max(cluster["members"], 1)))
                    ),
                    "bbox": [float(value) for value in cluster["bbox"]],
                }
            )

    fused.sort(key=lambda item: item["confidence"], reverse=True)
    return fused


def ensemble_prediction_files(
    prediction_paths: list[str | Path],
    output_path: str | Path,
    iou_threshold: float = 0.55,
    skip_box_threshold: float = 0.0,
    max_detections_per_image: int = 100,
) -> None:
    loaded = [json.loads(Path(path).read_text(encoding="utf-8")) for path in prediction_paths]
    by_image: dict[str, list[list[dict]]] = defaultdict(list)
    for predictions in loaded:
        for item in predictions:
            by_image[str(item["image_id"])].append(item.get("boxes", []))

    results = []
    for image_id, views in by_image.items():
        boxes = [box for view in views for box in view]
        fused = weighted_boxes_fusion(
            boxes,
            iou_threshold=iou_threshold,
            skip_box_threshold=skip_box_threshold,
        )
        results.append(
            {
                "image_id": image_id,
                "boxes": fused[:max_detections_per_image],
            }
        )
    Path(output_path).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Class-aware WBF for detector prediction JSON files.")
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iou_threshold", type=float, default=0.55)
    parser.add_argument("--skip_box_threshold", type=float, default=0.0)
    parser.add_argument("--max_detections_per_image", type=int, default=100)
    args = parser.parse_args()
    ensemble_prediction_files(
        args.predictions,
        args.output,
        iou_threshold=args.iou_threshold,
        skip_box_threshold=args.skip_box_threshold,
        max_detections_per_image=args.max_detections_per_image,
    )


if __name__ == "__main__":
    main()
