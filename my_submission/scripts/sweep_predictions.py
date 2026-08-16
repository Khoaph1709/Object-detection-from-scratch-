from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep score thresholds and top-K limits for saved predictions.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--ground_truth", required=True)
    parser.add_argument("--evaluator", default="")
    parser.add_argument("--output_dir", default="my_submission/artifacts/prediction_sweep")
    parser.add_argument("--thresholds", default="0.05,0.08,0.10,0.12,0.15,0.18,0.20,0.25,0.30")
    parser.add_argument("--topks", default="20,50,100")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_path = Path(args.predictions)
    ground_truth_path = Path(args.ground_truth)
    evaluator_path = Path(args.evaluator) if args.evaluator else ground_truth_path.parents[1] / "tools" / "evaluate_predictions.py"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()]
    topks = [int(item) for item in args.topks.split(",") if item.strip()]

    rows = []
    for topk in topks:
        for threshold in thresholds:
            filtered = filter_predictions(predictions, threshold=threshold, topk=topk)
            filtered_path = output_dir / f"pred_th{threshold:.2f}_top{topk}.json"
            score_path = output_dir / f"score_th{threshold:.2f}_top{topk}.json"
            filtered_path.write_text(json.dumps(filtered, ensure_ascii=False), encoding="utf-8")
            subprocess.run(
                [
                    sys.executable,
                    str(evaluator_path),
                    "--ground_truth",
                    str(ground_truth_path),
                    "--predictions",
                    str(filtered_path),
                    "--output",
                    str(score_path),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            score = json.loads(score_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "mAP@0.5": score.get("mAP@0.5", 0.0),
                    "threshold": threshold,
                    "topk": topk,
                    "num_predictions": score.get("num_predictions", 0),
                    "micro_precision": score.get("micro_precision", 0.0),
                    "micro_recall": score.get("micro_recall", 0.0),
                    "prediction_file": str(filtered_path),
                    "score_file": str(score_path),
                }
            )

    rows.sort(key=lambda row: row["mAP@0.5"], reverse=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in rows[:20]:
        print(
            f"mAP={row['mAP@0.5']:.6f} "
            f"th={row['threshold']:.2f} topk={row['topk']} "
            f"preds={row['num_predictions']} "
            f"precision={row['micro_precision']:.5f} recall={row['micro_recall']:.5f}"
        )
    print(f"Wrote sweep summary to {summary_path}")


def filter_predictions(predictions: list[dict], threshold: float, topk: int) -> list[dict]:
    filtered = []
    for item in predictions:
        boxes = [
            box
            for box in item.get("boxes", [])
            if float(box.get("confidence", 0.0)) >= threshold
        ]
        boxes.sort(key=lambda box: float(box["confidence"]), reverse=True)
        filtered.append({"image_id": item["image_id"], "boxes": boxes[:topk]})
    return filtered


if __name__ == "__main__":
    main()
