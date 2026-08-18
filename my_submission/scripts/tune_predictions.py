from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_THRESHOLDS = [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40]
DEFAULT_LIMITS = [10, 15, 20, 30, 50, 100]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune global/per-class score thresholds and per-image detection caps."
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--ground_truth", required=True)
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--thresholds", default="")
    parser.add_argument("--limits", default="")
    parser.add_argument(
        "--class_thresholds",
        default="",
        help="Optional class=values groups, e.g. chair=0.12|0.15|0.20,backpack=0.12|0.15.",
    )
    parser.add_argument(
        "--class_limits",
        default="",
        help="Optional class=values groups, e.g. chair=5|10|15,backpack=5|10|15.",
    )
    return parser.parse_args()


def parse_values(raw: str, defaults: list[float] | list[int], cast) -> list:
    if not raw.strip(", "):
        return defaults
    return [cast(value.strip()) for value in raw.split(",") if value.strip()]


def parse_class_grid(raw: str, cast) -> dict[str, list]:
    result: dict[str, list] = {}
    if not raw.strip():
        return result
    for group in raw.split(","):
        if not group.strip():
            continue
        if "=" not in group:
            raise ValueError(f"Expected class=values, got: {group}")
        class_name, values = group.split("=", 1)
        result[class_name.strip()] = [cast(value) for value in values.split("|") if value.strip()]
    return result


def filter_predictions(
    predictions: list[dict],
    threshold: float,
    limit: int,
    class_thresholds: dict[str, float] | None = None,
    class_limits: dict[str, int] | None = None,
) -> list[dict]:
    class_thresholds = class_thresholds or {}
    class_limits = class_limits or {}
    filtered = []
    for item in predictions:
        grouped: dict[str, list[dict]] = {}
        for box in item.get("boxes", []):
            class_name = str(box.get("class", ""))
            threshold_for_class = float(class_thresholds.get(class_name, threshold))
            if float(box.get("confidence", 0.0)) >= threshold_for_class:
                grouped.setdefault(class_name, []).append(box)

        kept: list[dict] = []
        for class_name, boxes in grouped.items():
            boxes.sort(key=lambda box: float(box.get("confidence", 0.0)), reverse=True)
            limit_for_class = int(class_limits.get(class_name, limit))
            kept.extend(boxes[:limit_for_class])
        kept.sort(key=lambda box: float(box.get("confidence", 0.0)), reverse=True)
        filtered.append({"image_id": item["image_id"], "boxes": kept})
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


def iter_grids(
    global_thresholds: list[float],
    global_limits: list[int],
    class_thresholds: dict[str, list[float]],
    class_limits: dict[str, list[int]],
):
    class_names = sorted(set(class_thresholds) | set(class_limits))
    if not class_names:
        for threshold in global_thresholds:
            for limit in global_limits:
                yield threshold, limit, {}, {}
        return

    # Keep the grid bounded: one class-specific sweep at a time while other
    # classes use the global values. This isolates the effect of chair/backpack
    # and avoids an exponential Cartesian product.
    yield_count = 0
    for threshold in global_thresholds:
        for limit in global_limits:
            yield threshold, limit, {}, {}
            yield_count += 1
            for class_name in class_names:
                thresholds = class_thresholds.get(class_name, [threshold])
                limits = class_limits.get(class_name, [limit])
                for class_threshold in thresholds:
                    for class_limit in limits:
                        yield (
                            threshold,
                            limit,
                            {class_name: class_threshold},
                            {class_name: class_limit},
                        )
                        yield_count += 1


def main() -> None:
    args = parse_args()
    predictions_path = Path(args.predictions)
    output_path = Path(args.output)
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    thresholds = parse_values(args.thresholds, DEFAULT_THRESHOLDS, float)
    limits = parse_values(args.limits, DEFAULT_LIMITS, int)
    class_thresholds = parse_class_grid(args.class_thresholds, float)
    class_limits = parse_class_grid(args.class_limits, int)

    results = []
    with tempfile.TemporaryDirectory(prefix="prediction_tune_") as temp_dir:
        temp_dir_path = Path(temp_dir)
        for index, (threshold, limit, class_threshold_override, class_limit_override) in enumerate(
            iter_grids(thresholds, limits, class_thresholds, class_limits)
        ):
            filtered_path = temp_dir_path / f"pred_{index}.json"
            score_path = temp_dir_path / f"score_{index}.json"
            filtered_path.write_text(
                json.dumps(
                    filter_predictions(
                        predictions,
                        threshold,
                        limit,
                        class_thresholds=class_threshold_override,
                        class_limits=class_limit_override,
                    ),
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            score = run_evaluator(args.evaluator, args.ground_truth, filtered_path, score_path)
            results.append(
                {
                    "score_threshold": threshold,
                    "max_detections_per_image": limit,
                    "class_score_thresholds": class_threshold_override,
                    "class_max_detections_per_image": class_limit_override,
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
