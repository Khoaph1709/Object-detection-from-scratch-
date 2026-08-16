from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_THRESHOLDS = [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40]
DEFAULT_LIMITS = [10, 15, 20, 30, 50, 100]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune prediction threshold and per-image cap on validation predictions.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--ground_truth", required=True)
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--thresholds", default=",")
    parser.add_argument("--limits", default=",")
    return parser.parse_args()


def parse_values(raw: str, defaults: list[float] | list[int], cast):
    if not raw.strip(", "):
        return defaults
    return [cast(value.strip()) for value in raw.split(",") if value.strip()]


def filter_predictions(predictions: list[dict], threshold: float, limit: int) -> list[dict]:
    filtered = []
    for item in predictions:
        boxes = [
            box for box in item.get("boxes", [])
            if float(box.get("confidence", 0.0)) >= threshold
        ]
        boxes.sort(key=lambda box: float(box.get("confidence", 0.0)), reverse=True)
        filtered.append({"image_id": item["image_id"], "boxes": boxes[:limit]})
    return filtered


def run_evaluator(evaluator: str, ground_truth: str, predictions_path: Path, score_path: Path) -> dict:
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
    args = parse_args()
    predictions_path = Path(args.predictions)
    output_path = Path(args.output)
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    thresholds = parse_values(args.thresholds, DEFAULT_THRESHOLDS, float)
    limits = parse_values(args.limits, DEFAULT_LIMITS, int)

    results = []
    with tempfile.TemporaryDirectory(prefix="prediction_tune_") as temp_dir:
        temp_dir_path = Path(temp_dir)
        for threshold in thresholds:
            for limit in limits:
                filtered_path = temp_dir_path / f"pred_{threshold:.3f}_{limit}.json"
                score_path = temp_dir_path / f"score_{threshold:.3f}_{limit}.json"
                filtered_path.write_text(
                    json.dumps(filter_predictions(predictions, threshold, limit), ensure_ascii=False),
                    encoding="utf-8",
                )
                score = run_evaluator(args.evaluator, args.ground_truth, filtered_path, score_path)
                results.append(
                    {
                        "score_threshold": threshold,
                        "max_detections_per_image": limit,
                        "mAP@0.5": float(score.get("mAP@0.5", 0.0)),
                        "micro_precision": float(score.get("micro_precision", 0.0)),
                        "micro_recall": float(score.get("micro_recall", 0.0)),
                        "num_predictions": int(score.get("num_predictions", 0)),
                        "per_class": score.get("per_class", {}),
                    }
                )

    results.sort(key=lambda item: (item["mAP@0.5"], item["micro_recall"]), reverse=True)
    output = {
        "best": results[0] if results else None,
        "results": results,
        "source_predictions": str(predictions_path),
    }
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["best"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
