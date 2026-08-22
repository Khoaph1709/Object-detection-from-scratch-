from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_THRESHOLDS = [0.06, 0.08, 0.10, 0.12]
DEFAULT_NMS = [0.45, 0.50, 0.55]
DEFAULT_LIMITS = [30, 50, 70, 100]


def parse_values(raw: str, defaults: list, cast) -> list:
    if not raw.strip():
        return list(defaults)
    values = [cast(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("At least one sweep value is required")
    return values


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
    return inter / max(area_a + area_b - inter, 1e-12)


def class_nms(boxes: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    ordered = sorted(boxes, key=lambda box: float(box.get("confidence", 0.0)), reverse=True)
    kept: list[dict[str, Any]] = []
    for candidate in ordered:
        bbox = [float(value) for value in candidate.get("bbox", [])]
        if len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        if all(iou(bbox, [float(value) for value in previous["bbox"]]) <= threshold for previous in kept):
            updated = dict(candidate)
            updated["bbox"] = bbox
            kept.append(updated)
    return kept


def postprocess_predictions(
    predictions: list[dict[str, Any]],
    score_threshold: float,
    nms_threshold: float,
    max_detections_per_image: int,
    class_score_thresholds: dict[str, float] | None = None,
    class_max_detections: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    class_score_thresholds = class_score_thresholds or {}
    class_max_detections = class_max_detections or {}
    result: list[dict[str, Any]] = []
    for item in predictions:
        by_class: dict[str, list[dict[str, Any]]] = {}
        for box in item.get("boxes", []):
            class_name = str(box.get("class", ""))
            threshold = float(class_score_thresholds.get(class_name, score_threshold))
            if float(box.get("confidence", 0.0)) >= threshold:
                by_class.setdefault(class_name, []).append(box)

        kept: list[dict[str, Any]] = []
        for class_name, boxes in by_class.items():
            class_kept = class_nms(boxes, nms_threshold)
            class_limit = int(class_max_detections.get(class_name, max_detections_per_image))
            kept.extend(class_kept[:class_limit])
        kept.sort(key=lambda box: float(box.get("confidence", 0.0)), reverse=True)
        result.append({"image_id": item["image_id"], "boxes": kept[:max_detections_per_image]})
    return result


def run_evaluator(evaluator: str, ground_truth: str, predictions_path: Path, score_path: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        evaluator,
        "--ground_truth",
        ground_truth,
        "--predictions",
        str(predictions_path),
        "--output",
        str(score_path),
    ]
    subprocess.run(command, check=True)
    return json.loads(score_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep post-processing settings on an existing validation prediction JSON."
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--thresholds", default="0.06,0.08,0.10,0.12")
    parser.add_argument("--nms-thresholds", default="0.45,0.50,0.55")
    parser.add_argument("--limits", default="30,50,70,100")
    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    thresholds = parse_values(args.thresholds, DEFAULT_THRESHOLDS, float)
    nms_thresholds = parse_values(args.nms_thresholds, DEFAULT_NMS, float)
    limits = parse_values(args.limits, DEFAULT_LIMITS, int)

    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="inference_sweep_") as temp_dir:
        temp_path = Path(temp_dir)
        for index, (threshold, nms_threshold, limit) in enumerate(
            (item for item in ((t, n, l) for t in thresholds for n in nms_thresholds for l in limits))
        ):
            processed = postprocess_predictions(
                predictions,
                score_threshold=threshold,
                nms_threshold=nms_threshold,
                max_detections_per_image=limit,
            )
            prediction_file = output_dir / f"pred_t{threshold:.3f}_n{nms_threshold:.2f}_k{limit}.json"
            score_file = output_dir / f"score_t{threshold:.3f}_n{nms_threshold:.2f}_k{limit}.json"
            prediction_file.write_text(json.dumps(processed, ensure_ascii=False), encoding="utf-8")
            score = run_evaluator(args.evaluator, args.ground_truth, prediction_file, score_file)
            results.append(
                {
                    "score_threshold": threshold,
                    "nms_threshold": nms_threshold,
                    "max_detections_per_image": limit,
                    "mAP@0.5": float(score.get("mAP@0.5", 0.0)),
                    "micro_precision": float(score.get("micro_precision", 0.0)),
                    "micro_recall": float(score.get("micro_recall", 0.0)),
                    "num_predictions": int(score.get("num_predictions", 0)),
                    "prediction_file": str(prediction_file),
                    "score_file": str(score_file),
                    "per_class": score.get("per_class", {}),
                }
            )

    results.sort(key=lambda item: (item["mAP@0.5"], item["micro_recall"]), reverse=True)
    summary = {
        "source_predictions": str(predictions_path),
        "results": results,
        "best": results[0] if results else None,
        "warning": (
            "This sweep can only suppress boxes present in the input JSON. "
            "It cannot recover candidates already removed by the original model NMS."
        ),
    }
    summary_path = output_dir / "sweep_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["best"], ensure_ascii=False, indent=2))
    print(f"Saved sweep summary to {summary_path}")


if __name__ == "__main__":
    main()
