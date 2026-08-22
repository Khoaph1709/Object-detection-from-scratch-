"""Data-quality audit for the custom detector's COCO-derived annotation format.

The audit is intentionally read-only. It never modifies annotation files and it never
opens validation/test annotations unless the user explicitly passes such a file as the
single ``--annotations`` argument. The expected bbox format in this repository is
[xmin, ymin, xmax, ymax] in original-image pixel coordinates.

Outputs:
  summary.json             machine-readable aggregate report
  issues.json              every detected issue/warning
  boxes.csv               one row per annotation with geometry statistics
  review_manifest.json     image IDs grouped for manual review
  review_images/<group>   optional rendered images with boxes overlaid

Optional ``--predictions`` accepts this repository's prediction schema and is used
only to rank likely train-image false positives for chair/backpack review.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

EXPECTED_CLASSES = ("bottle", "cup", "chair", "laptop", "backpack")
AREA_BINS = ((0.0, 0.01, "0-1%"), (0.01, 0.05, "1-5%"), (0.05, 0.10, "5-10%"), (0.10, 0.25, "10-25%"), (0.25, 1.01, ">25%"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit one train annotation JSON without modifying it.")
    parser.add_argument("--annotations", required=True, help="Path to the train annotation JSON.")
    parser.add_argument("--image-dir", default="", help="Optional train image directory for dimensions and review renders.")
    parser.add_argument("--predictions", default="", help="Optional train prediction JSON for ranking likely chair/backpack false positives.")
    parser.add_argument("--output-dir", required=True, help="Directory for reports and optional review images.")
    parser.add_argument("--review-count", type=int, default=50, help="Maximum images in each normal review group.")
    parser.add_argument("--overlap-review-count", type=int, default=100, help="Maximum images in overlap/duplicate review group.")
    parser.add_argument("--tiny-area-ratio", type=float, default=0.05, help="Area ratio used for tiny/small-object review.")
    parser.add_argument("--duplicate-iou", type=float, default=0.90, help="Same-class IoU threshold for duplicate review.")
    parser.add_argument("--overlap-iou", type=float, default=0.90, help="Different-class IoU threshold for suspicious overlap review.")
    parser.add_argument("--prediction-match-iou", type=float, default=0.50, help="IoU required for a prediction to match same-class GT.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-render", action="store_true", help="Do not render review images.")
    return parser.parse_args()


def as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def box_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def iou(box_a: list[float], box_b: list[float]) -> float:
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area(box_a) + box_area(box_b) - intersection
    return intersection / max(union, 1e-12)


def image_path(image_info: dict[str, Any], image_dir: Path | None) -> Path | None:
    if image_dir is None:
        return None
    image_id = str(image_info.get("id", ""))
    direct = image_dir / image_id
    if direct.exists():
        return direct
    file_name = Path(str(image_info.get("file_name", ""))).name
    fallback = image_dir / file_name
    return fallback if fallback.exists() else None


def read_dimensions(image_info: dict[str, Any], image_dir: Path | None) -> tuple[int | None, int | None, str]:
    width = as_float(image_info.get("width"))
    height = as_float(image_info.get("height"))
    if width is not None and height is not None and width > 0 and height > 0:
        return int(round(width)), int(round(height)), "metadata"
    path = image_path(image_info, image_dir)
    if path is None:
        return None, None, "missing"
    try:
        with Image.open(path) as image:
            return image.width, image.height, "image"
    except (OSError, ValueError):
        return None, None, "unreadable"


def area_bin(ratio: float | None) -> str:
    if ratio is None:
        return "unknown"
    for lower, upper, label in AREA_BINS:
        if lower <= ratio < upper:
            return label
    return ">25%"


def load_predictions(path: Path) -> dict[str, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("predictions", payload.get("images", []))
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not isinstance(payload, list):
        return by_image
    for item in payload:
        if not isinstance(item, dict):
            continue
        image_id = str(item.get("image_id", ""))
        for box in item.get("boxes", []) or []:
            if isinstance(box, dict):
                by_image[image_id].append(box)
    return by_image


def add_issue(
    issues: list[dict[str, Any]],
    category: str,
    image_id: str,
    message: str,
    annotation_index: int | None = None,
    class_name: str = "",
    bbox: list[float] | None = None,
    severity: str = "warning",
) -> None:
    issues.append(
        {
            "category": category,
            "severity": severity,
            "image_id": image_id,
            "annotation_index": annotation_index,
            "class": class_name,
            "bbox": bbox,
            "message": message,
        }
    )


def prediction_false_positive_scores(
    predictions: dict[str, list[dict[str, Any]]],
    gt_by_image: dict[str, list[dict[str, Any]]],
    match_iou: float,
) -> dict[str, dict[str, dict[str, float]]]:
    scores: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: {"count": 0.0, "score_sum": 0.0, "max_score": 0.0})
    )
    for image_id, boxes in predictions.items():
        gt_boxes = gt_by_image.get(image_id, [])
        for prediction in boxes:
            class_name = str(prediction.get("class", ""))
            confidence = as_float(prediction.get("confidence"))
            raw_bbox = prediction.get("bbox")
            if class_name not in EXPECTED_CLASSES or confidence is None or not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
                continue
            values = [as_float(value) for value in raw_bbox]
            if any(value is None for value in values):
                continue
            pred_box = [float(value) for value in values]
            same_class = [item["bbox"] for item in gt_boxes if item.get("class") == class_name]
            max_overlap = max((iou(pred_box, gt_box) for gt_box in same_class), default=0.0)
            if max_overlap < match_iou:
                entry = scores[image_id][class_name]
                entry["count"] += 1.0
                entry["score_sum"] += max(0.0, float(confidence))
                entry["max_score"] = max(entry["max_score"], float(confidence))
    return scores


def sample_ranked(entries: list[tuple[str, float]], limit: int, rng: random.Random) -> list[str]:
    if limit <= 0:
        return []
    entries = sorted(entries, key=lambda item: (-item[1], item[0]))
    if len(entries) <= limit:
        return [image_id for image_id, _ in entries]
    # Keep deterministic top-ranked items; the seed is used only to break exact ties.
    grouped: dict[float, list[str]] = defaultdict(list)
    for image_id, score in entries:
        grouped[score].append(image_id)
    result: list[str] = []
    for score in sorted(grouped, reverse=True):
        candidates = grouped[score]
        rng.shuffle(candidates)
        result.extend(sorted(candidates))
        if len(result) >= limit:
            break
    return result[:limit]


def render_review_image(
    image_info: dict[str, Any],
    image_dir: Path,
    output_path: Path,
    boxes: list[dict[str, Any]],
    highlights: set[tuple[int, str]],
    title: str,
) -> bool:
    source = image_path(image_info, image_dir)
    if source is None:
        return False
    try:
        with Image.open(source).convert("RGB") as original:
            image = ImageOps.contain(original, (1400, 1000))
            sx = image.width / max(original.width, 1)
            sy = image.height / max(original.height, 1)
            draw = ImageDraw.Draw(image)
            for index, item in enumerate(boxes):
                bbox = item.get("bbox")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                try:
                    x1, y1, x2, y2 = [float(value) for value in bbox]
                except (TypeError, ValueError):
                    continue
                color = "red" if (index, str(item.get("class", ""))) in highlights else "lime"
                scaled = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
                draw.rectangle(scaled, outline=color, width=max(2, int(round(min(sx, sy) * 3))))
                label = str(item.get("class", "?"))
                confidence = item.get("confidence")
                if confidence is not None:
                    label += f" {float(confidence):.2f}"
                draw.text((scaled[0] + 2, max(0, scaled[1] - 16)), label, fill=color)
            draw.rectangle((0, 0, image.width, 24), fill="black")
            draw.text((4, 4), title, fill="white")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path, quality=92)
            return True
    except (OSError, ValueError):
        return False


def audit(args: argparse.Namespace) -> dict[str, Any]:
    annotation_path = Path(args.annotations).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = Path(args.image_dir).expanduser().resolve() if args.image_dir else None
    rng = random.Random(args.seed)
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    classes = list(data.get("classes", []))
    image_by_id: dict[str, dict[str, Any]] = {}
    issues: list[dict[str, Any]] = []

    if tuple(classes) != EXPECTED_CLASSES:
        add_issue(
            issues,
            "class_mapping",
            "",
            f"Top-level classes are {classes!r}; expected exact order {list(EXPECTED_CLASSES)!r}.",
            severity="error",
        )
    if len(set(classes)) != len(classes):
        add_issue(issues, "class_mapping", "", "Top-level classes contain duplicates.", severity="error")

    for image_index, image_info in enumerate(images):
        if not isinstance(image_info, dict):
            add_issue(issues, "image_schema", "", f"Image record {image_index} is not an object.", severity="error")
            continue
        image_id = str(image_info.get("id", ""))
        if not image_id:
            add_issue(issues, "image_schema", "", f"Image record {image_index} has no id.", severity="error")
            continue
        if image_id in image_by_id:
            add_issue(issues, "duplicate_image_id", image_id, "Image id appears more than once.", severity="error")
        image_by_id[image_id] = image_info
        width, height, source = read_dimensions(image_info, image_dir)
        if width is None or height is None:
            add_issue(issues, "missing_image_dimensions", image_id, "Could not determine positive image width/height.", severity="error")
        image_info["_audit_width"] = width
        image_info["_audit_height"] = height
        image_info["_audit_dimension_source"] = source
        if image_dir is not None and image_path(image_info, image_dir) is None:
            add_issue(issues, "missing_image", image_id, f"Image was not found under {image_dir}.", severity="error")

    annotations_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    gt_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    area_counts: Counter[str] = Counter()
    tiny_image_scores: dict[str, float] = {}
    valid_box_count = 0

    for annotation_index, annotation in enumerate(annotations):
        if not isinstance(annotation, dict):
            add_issue(issues, "annotation_schema", "", f"Annotation record {annotation_index} is not an object.", annotation_index, severity="error")
            continue
        image_id = str(annotation.get("image_id", ""))
        class_name = str(annotation.get("class", ""))
        annotations_by_image[image_id].append(annotation)
        if image_id not in image_by_id:
            add_issue(issues, "orphan_annotation", image_id, "Annotation references an image id absent from images.", annotation_index, class_name, severity="error")
        if class_name not in EXPECTED_CLASSES:
            add_issue(issues, "class_mapping", image_id, f"Unknown annotation class {class_name!r}.", annotation_index, class_name, severity="error")
        class_counts[class_name] += 1
        raw_bbox = annotation.get("bbox")
        row: dict[str, Any] = {
            "annotation_index": annotation_index,
            "image_id": image_id,
            "class": class_name,
            "xmin": "",
            "ymin": "",
            "xmax": "",
            "ymax": "",
            "width": "",
            "height": "",
            "area": "",
            "area_ratio": "",
            "area_bin": "unknown",
            "tiny_by_area": False,
            "issue_types": "",
        }
        annotation_issue_types: list[str] = []
        if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
            add_issue(issues, "bbox_schema", image_id, "bbox must be a list of four [xmin,ymin,xmax,ymax] values.", annotation_index, class_name, severity="error")
            annotation_issue_types.append("bbox_schema")
            rows.append(row)
            continue
        values = [as_float(value) for value in raw_bbox]
        if any(value is None for value in values):
            add_issue(issues, "bbox_non_numeric", image_id, "bbox contains a non-numeric coordinate.", annotation_index, class_name, severity="error")
            annotation_issue_types.append("bbox_non_numeric")
            rows.append(row)
            continue
        bbox = [float(value) for value in values]
        row.update({"xmin": bbox[0], "ymin": bbox[1], "xmax": bbox[2], "ymax": bbox[3]})
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = max(0.0, width) * max(0.0, height)
        row.update({"width": width, "height": height, "area": area})
        image_info = image_by_id.get(image_id, {})
        image_width = image_info.get("_audit_width")
        image_height = image_info.get("_audit_height")
        valid_geometry = True
        if width <= 0 or height <= 0:
            add_issue(issues, "invalid_geometry", image_id, "xmax <= xmin or ymax <= ymin.", annotation_index, class_name, bbox, "error")
            annotation_issue_types.append("invalid_geometry")
            valid_geometry = False
        if width < 2 or height < 2:
            add_issue(issues, "subpixel_box", image_id, "Box width or height is below 2 pixels.", annotation_index, class_name, bbox, "warning")
            annotation_issue_types.append("subpixel_box")
        if image_width is not None and image_height is not None:
            if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > image_width or bbox[3] > image_height:
                add_issue(issues, "out_of_bounds", image_id, f"Box is outside image bounds {image_width}x{image_height}.", annotation_index, class_name, bbox, "error")
                annotation_issue_types.append("out_of_bounds")
            if image_width > 0 and image_height > 0 and area >= 0:
                ratio = area / float(image_width * image_height)
                row["area_ratio"] = ratio
                row["area_bin"] = area_bin(ratio)
                row["tiny_by_area"] = ratio <= args.tiny_area_ratio
                area_counts[row["area_bin"]] += 1
                if row["tiny_by_area"]:
                    previous = tiny_image_scores.get(image_id, 1.0)
                    tiny_image_scores[image_id] = min(previous, ratio)
        if valid_geometry:
            valid_box_count += 1
            gt_by_image[image_id].append({"class": class_name, "bbox": bbox, "annotation_index": annotation_index})
        row["issue_types"] = ";".join(annotation_issue_types)
        rows.append(row)

    duplicate_images: set[str] = set()
    overlap_images: set[str] = set()
    for image_id, image_annotations in annotations_by_image.items():
        valid_items: list[tuple[int, str, list[float]]] = []
        for annotation in image_annotations:
            raw_bbox = annotation.get("bbox")
            if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
                continue
            values = [as_float(value) for value in raw_bbox]
            if any(value is None for value in values):
                continue
            bbox = [float(value) for value in values]
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            valid_items.append((int(annotation.get("_audit_index", -1)), str(annotation.get("class", "")), bbox))
        # Use original list positions because annotations may not carry an index.
        valid_items = []
        for local_index, annotation in enumerate(image_annotations):
            raw_bbox = annotation.get("bbox")
            if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
                continue
            values = [as_float(value) for value in raw_bbox]
            if any(value is None for value in values):
                continue
            bbox = [float(value) for value in values]
            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                valid_items.append((local_index, str(annotation.get("class", "")), bbox))
        for first in range(len(valid_items)):
            _, class_a, box_a = valid_items[first]
            for second in range(first + 1, len(valid_items)):
                _, class_b, box_b = valid_items[second]
                overlap = iou(box_a, box_b)
                if class_a == class_b and overlap >= args.duplicate_iou:
                    duplicate_images.add(image_id)
                    add_issue(issues, "duplicate_box", image_id, f"Same-class boxes have IoU {overlap:.4f} >= {args.duplicate_iou:.2f}.", class_name=class_a, bbox=box_a, severity="warning")
                elif class_a != class_b and overlap >= args.overlap_iou:
                    overlap_images.add(image_id)
                    add_issue(issues, "cross_class_overlap", image_id, f"Different-class boxes {class_a}/{class_b} have IoU {overlap:.4f} >= {args.overlap_iou:.2f}; review manually.", class_name=f"{class_a}|{class_b}", bbox=box_a, severity="warning")

    predictions_by_image: dict[str, list[dict[str, Any]]] = {}
    fp_scores: dict[str, dict[str, dict[str, float]]] = {}
    if args.predictions:
        prediction_path = Path(args.predictions).expanduser().resolve()
        predictions_by_image = load_predictions(prediction_path)
        fp_scores = prediction_false_positive_scores(predictions_by_image, gt_by_image, args.prediction_match_iou)

    all_image_ids = list(image_by_id)
    empty_target_images = [image_id for image_id in all_image_ids if not annotations_by_image.get(image_id)]
    no_chair_backpack = [image_id for image_id in all_image_ids if not any(item.get("class") in {"chair", "backpack"} for item in annotations_by_image.get(image_id, []))]
    tiny_entries = [(image_id, score) for image_id, score in tiny_image_scores.items()]
    overlap_entries = [(image_id, 1.0) for image_id in sorted(duplicate_images | overlap_images)]
    chair_fp_entries = [
        (image_id, metrics["score_sum"] + metrics["max_score"] * 1e-3)
        for image_id, per_class in fp_scores.items()
        if (metrics := per_class.get("chair")) is not None
    ]
    backpack_fp_entries = [
        (image_id, metrics["score_sum"] + metrics["max_score"] * 1e-3)
        for image_id, per_class in fp_scores.items()
        if (metrics := per_class.get("backpack")) is not None
    ]

    review_manifest: dict[str, Any] = {
        "source_annotations": str(annotation_path),
        "source_predictions": str(Path(args.predictions).expanduser().resolve()) if args.predictions else None,
        "review_count": args.review_count,
        "groups": {
            "chair_false_positive_high": sample_ranked(chair_fp_entries, args.review_count, rng),
            "backpack_false_positive_high": sample_ranked(backpack_fp_entries, args.review_count, rng),
            "tiny_objects": sample_ranked(tiny_entries, args.review_count, rng),
            "no_chair_or_backpack": sorted(no_chair_backpack)[: max(0, args.review_count)],
            "duplicate_or_overlap_suspect": sorted(overlap_entries, key=lambda item: item[0])[: max(0, args.overlap_review_count)],
        },
        "notes": {
            "chair_false_positive_high": "Requires --predictions; empty when train predictions were not supplied.",
            "backpack_false_positive_high": "Requires --predictions; empty when train predictions were not supplied.",
            "no_chair_or_backpack": "Negative-background review group; absence of annotation is not proof that the image is truly negative.",
            "duplicate_or_overlap_suspect": "High overlap is a review signal, not an automatic annotation error because real occlusion can overlap.",
        },
    }

    if not args.no_render and image_dir is not None:
        for group, image_ids in review_manifest["groups"].items():
            for image_id in image_ids:
                image_info = image_by_id.get(image_id)
                if image_info is None:
                    continue
                source_boxes = [dict(item) for item in annotations_by_image.get(image_id, [])]
                if group in {"chair_false_positive_high", "backpack_false_positive_high"}:
                    source_boxes.extend(dict(item) for item in predictions_by_image.get(image_id, []))
                highlight: set[tuple[int, str]] = set()
                for index, item in enumerate(source_boxes):
                    if group == "tiny_objects" and item.get("class") in EXPECTED_CLASSES:
                        highlight.add((index, str(item.get("class"))))
                safe_id = image_id.replace("/", "_").replace("\\", "_")
                render_review_image(
                    image_info,
                    image_dir,
                    output_dir / "review_images" / group / f"{safe_id}.jpg",
                    source_boxes,
                    highlight,
                    f"{group} | {image_id}",
                )

    with (output_dir / "issues.json").open("w", encoding="utf-8") as file:
        json.dump(issues, file, ensure_ascii=False, indent=2)
    with (output_dir / "review_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(review_manifest, file, ensure_ascii=False, indent=2)
    with (output_dir / "boxes.csv").open("w", encoding="utf-8", newline="") as file:
        fieldnames = list(rows[0].keys()) if rows else ["annotation_index", "image_id", "class"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    issue_counts = Counter(issue["category"] for issue in issues)
    summary = {
        "audit_version": 1,
        "read_only": True,
        "source_annotations": str(annotation_path),
        "source_image_dir": str(image_dir) if image_dir else None,
        "source_predictions": str(Path(args.predictions).expanduser().resolve()) if args.predictions else None,
        "expected_classes": list(EXPECTED_CLASSES),
        "annotation_classes": classes,
        "class_order_matches": tuple(classes) == EXPECTED_CLASSES,
        "images": len(images),
        "unique_image_ids": len(image_by_id),
        "annotations": len(annotations),
        "valid_geometry_boxes": valid_box_count,
        "empty_images": len(empty_target_images),
        "empty_image_ids_sample": sorted(empty_target_images)[:args.review_count],
        "class_counts": dict(sorted(class_counts.items())),
        "area_bin_counts": dict(area_counts),
        "tiny_area_ratio_threshold": args.tiny_area_ratio,
        "tiny_object_count": sum(1 for row in rows if row.get("tiny_by_area")),
        "issue_counts": dict(sorted(issue_counts.items())),
        "duplicate_image_count": len(duplicate_images),
        "cross_class_overlap_image_count": len(overlap_images),
        "prediction_false_positive_images": len(fp_scores),
        "review_manifest": str(output_dir / "review_manifest.json"),
        "issues_file": str(output_dir / "issues.json"),
        "boxes_file": str(output_dir / "boxes.csv"),
        "warning": "This audit reports candidates for review; it does not auto-delete, relabel, or repair annotations.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = audit(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
