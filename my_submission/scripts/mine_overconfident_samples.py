from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine overconfident false positives and localization errors from predictions."
    )
    parser.add_argument("--ground_truth", required=True, help="Assignment-format annotation JSON.")
    parser.add_argument("--predictions", required=True, help="Prediction JSON produced by predict.py.")
    parser.add_argument("--image_dir", required=True, help="Directory containing the source images.")
    parser.add_argument("--output_dir", default="overconfidence_samples")
    parser.add_argument("--score_threshold", type=float, default=0.30)
    parser.add_argument("--match_iou_threshold", type=float, default=0.50)
    parser.add_argument("--max_images", type=int, default=100)
    parser.add_argument("--max_errors_per_image", type=int, default=8)
    parser.add_argument("--min_image_score", type=float, default=0.0)
    return parser.parse_args()


def box_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_iou(box_a: list[float], box_b: list[float]) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    intersection = box_area([x1, y1, x2, y2])
    union = box_area(box_a) + box_area(box_b) - intersection
    return intersection / max(union, 1e-9)


def resolve_image_path(image_info: dict[str, Any], image_dir: Path) -> Path:
    image_id = str(image_info["id"])
    direct = image_dir / image_id
    if direct.exists():
        return direct
    file_name = Path(str(image_info.get("file_name", image_id))).name
    return image_dir / file_name


def classify_prediction(
    prediction: dict[str, Any],
    gt_by_class: dict[str, list[list[float]]],
    matched_gt: dict[str, set[int]],
    match_iou_threshold: float,
) -> dict[str, Any]:
    class_name = str(prediction.get("class", ""))
    bbox = [float(value) for value in prediction.get("bbox", [])]
    confidence = float(prediction.get("confidence", 0.0))
    gt_boxes = gt_by_class.get(class_name, [])
    overlaps = [box_iou(bbox, gt_box) for gt_box in gt_boxes]
    max_iou = max(overlaps, default=0.0)
    best_gt_index = int(max(range(len(overlaps)), key=overlaps.__getitem__)) if overlaps else -1

    if best_gt_index >= 0 and max_iou >= match_iou_threshold:
        if best_gt_index in matched_gt.setdefault(class_name, set()):
            reason = "duplicate_prediction"
        else:
            matched_gt[class_name].add(best_gt_index)
            reason = "true_positive"
    elif gt_boxes:
        reason = "localization_error"
    else:
        reason = "background_false_positive"

    is_error = reason != "true_positive"
    severity = confidence * (1.0 - max_iou) if is_error else 0.0
    if reason == "background_false_positive":
        severity = confidence

    return {
        "class": class_name,
        "confidence": confidence,
        "bbox": bbox,
        "max_same_class_iou": max_iou,
        "best_gt_index": best_gt_index,
        "reason": reason,
        "overconfidence_score": severity,
    }


def find_font() -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), 14)
    return ImageFont.load_default()


def draw_box(draw: ImageDraw.ImageDraw, box: list[float], color: str, label: str, font) -> None:
    coords = tuple(int(round(value)) for value in box)
    draw.rectangle(coords, outline=color, width=3)
    left, top, right, bottom = coords
    text_bbox = draw.textbbox((left, top), label, font=font)
    label_top = max(0, top - (text_bbox[3] - text_bbox[1]) - 4)
    draw.rectangle((left, label_top, text_bbox[2] + 4, top), fill=color)
    draw.text((left + 2, label_top + 1), label, fill="white", font=font)


def render_sample(
    image_path: Path,
    image_info: dict[str, Any],
    gt_annotations: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    output_path: Path,
) -> bool:
    if not image_path.exists():
        return False
    with Image.open(image_path).convert("RGB") as image:
        draw = ImageDraw.Draw(image)
        font = find_font()
        for annotation in gt_annotations:
            draw_box(
                draw,
                [float(value) for value in annotation["bbox"]],
                "#16a34a",
                f"GT {annotation['class']}",
                font,
            )
        for error in errors:
            reason = str(error["reason"]).replace("_", " ")
            draw_box(
                draw,
                error["bbox"],
                "#dc2626",
                f"P {error['class']} {error['confidence']:.2f} IoU {error['max_same_class_iou']:.2f} {reason}",
                font,
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, quality=95)
    return True


def mine_overconfident_samples(
    ground_truth: dict[str, Any],
    predictions: list[dict[str, Any]],
    image_dir: str | Path,
    output_dir: str | Path,
    score_threshold: float = 0.30,
    match_iou_threshold: float = 0.50,
    max_images: int = 100,
    max_errors_per_image: int = 8,
    min_image_score: float = 0.0,
) -> dict[str, Any]:
    image_root = Path(image_dir)
    output_root = Path(output_dir)
    images_by_id = {str(item["id"]): item for item in ground_truth.get("images", [])}
    annotations_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    gt_by_image_class: dict[str, dict[str, list[list[float]]]] = defaultdict(lambda: defaultdict(list))
    for annotation in ground_truth.get("annotations", []):
        image_id = str(annotation["image_id"])
        annotations_by_image[image_id].append(annotation)
        gt_by_image_class[image_id][str(annotation["class"])].append(
            [float(value) for value in annotation["bbox"]]
        )

    predictions_by_id = {str(item["image_id"]): item for item in predictions}
    image_records = []
    error_records = []
    reason_counts = Counter()
    class_reason_counts = Counter()

    for image_id, image_info in images_by_id.items():
        matched_gt: dict[str, set[int]] = {}
        image_errors = []
        raw_predictions = predictions_by_id.get(image_id, {}).get("boxes", [])
        raw_predictions = sorted(
            raw_predictions,
            key=lambda item: float(item.get("confidence", 0.0)),
            reverse=True,
        )
        for raw_prediction in raw_predictions:
            confidence = float(raw_prediction.get("confidence", 0.0))
            if confidence < score_threshold:
                continue
            classified = classify_prediction(
                raw_prediction,
                gt_by_image_class[image_id],
                matched_gt,
                match_iou_threshold,
            )
            if classified["reason"] == "true_positive":
                continue
            classified["image_id"] = image_id
            classified["image_path"] = str(resolve_image_path(image_info, image_root))
            image_errors.append(classified)
            error_records.append(classified)
            reason_counts[classified["reason"]] += 1
            class_reason_counts[(classified["class"], classified["reason"])] += 1

        image_errors.sort(key=lambda item: item["overconfidence_score"], reverse=True)
        image_errors = image_errors[: max(0, max_errors_per_image)]
        image_score = sum(float(item["overconfidence_score"]) for item in image_errors)
        if image_errors and image_score >= min_image_score:
            image_records.append(
                {
                    "image_id": image_id,
                    "image_path": str(resolve_image_path(image_info, image_root)),
                    "image_score": image_score,
                    "num_errors": len(image_errors),
                    "errors": image_errors,
                }
            )

    image_records.sort(key=lambda item: item["image_score"], reverse=True)
    selected_images = image_records[: max(0, max_images)]
    selected_ids = {item["image_id"] for item in selected_images}
    selected_errors = [item for item in error_records if item["image_id"] in selected_ids]

    visualization_dir = output_root / "visualizations"
    rendered = 0
    for rank, record in enumerate(selected_images, start=1):
        image_info = images_by_id[record["image_id"]]
        output_path = visualization_dir / f"{rank:04d}_{Path(record['image_path']).name}"
        if output_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            output_path = output_path.with_suffix(".jpg")
        if render_sample(
            Path(record["image_path"]),
            image_info,
            annotations_by_image[record["image_id"]],
            record["errors"],
            output_path,
        ):
            record["visualization_path"] = str(output_path)
            rendered += 1

    output_root.mkdir(parents=True, exist_ok=True)
    report = {
        "settings": {
            "score_threshold": score_threshold,
            "match_iou_threshold": match_iou_threshold,
            "max_images": max_images,
            "max_errors_per_image": max_errors_per_image,
            "min_image_score": min_image_score,
        },
        "summary": {
            "num_annotation_images": len(images_by_id),
            "num_prediction_images": len(predictions_by_id),
            "num_overconfident_error_predictions": len(error_records),
            "num_selected_images": len(selected_images),
            "num_rendered_images": rendered,
            "reason_counts": dict(reason_counts),
            "class_reason_counts": {f"{a}__{b}": count for (a, b), count in class_reason_counts.items()},
        },
        "images": selected_images,
        "errors": selected_errors,
    }
    (output_root / "overconfidence_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_root / "overconfidence_errors.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_id",
                "image_path",
                "class",
                "confidence",
                "max_same_class_iou",
                "reason",
                "overconfidence_score",
                "bbox",
            ],
        )
        writer.writeheader()
        for error in selected_errors:
            writer.writerow({key: error.get(key, "") for key in writer.fieldnames})
    (output_root / "summary.json").write_text(
        json.dumps(report["summary"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    args = parse_args()
    ground_truth = json.loads(Path(args.ground_truth).read_text(encoding="utf-8"))
    predictions = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    report = mine_overconfident_samples(
        ground_truth,
        predictions,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        score_threshold=args.score_threshold,
        match_iou_threshold=args.match_iou_threshold,
        max_images=args.max_images,
        max_errors_per_image=args.max_errors_per_image,
        min_image_score=args.min_image_score,
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
