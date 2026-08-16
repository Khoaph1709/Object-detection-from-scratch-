from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize high-confidence chair false positives.")
    parser.add_argument("--predictions", default="val_predictions_modal.json")
    parser.add_argument("--ground_truth", default="indoor5-v2-student/public/annotations/val.json")
    parser.add_argument("--image_dir", default="indoor5-v2-student/public/val/images")
    parser.add_argument("--output_dir", default="my_submission/artifacts/chair_false_positives")
    parser.add_argument("--top_images", type=int, default=100)
    parser.add_argument("--max_boxes_per_image", type=int, default=12)
    parser.add_argument("--min_score", type=float, default=0.12)
    parser.add_argument("--grid_columns", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    singles_dir = output_dir / "images"
    singles_dir.mkdir(parents=True, exist_ok=True)

    ground_truth = json.loads(Path(args.ground_truth).read_text(encoding="utf-8"))
    predictions = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    image_dir = Path(args.image_dir)

    gt_classes_by_image = {image["id"]: set() for image in ground_truth["images"]}
    gt_boxes_by_image = {image["id"]: [] for image in ground_truth["images"]}
    for ann in ground_truth["annotations"]:
        gt_classes_by_image[ann["image_id"]].add(ann["class"])
        gt_boxes_by_image[ann["image_id"]].append(ann)

    candidates = []
    for item in predictions:
        image_id = item["image_id"]
        if "chair" in gt_classes_by_image.get(image_id, set()):
            continue
        chair_boxes = [
            box
            for box in item.get("boxes", [])
            if box.get("class") == "chair" and float(box.get("confidence", 0.0)) >= args.min_score
        ]
        if not chair_boxes:
            continue
        chair_boxes.sort(key=lambda box: float(box["confidence"]), reverse=True)
        candidates.append(
            {
                "image_id": image_id,
                "max_score": float(chair_boxes[0]["confidence"]),
                "num_chair_predictions": len(chair_boxes),
                "gt_classes": sorted(gt_classes_by_image.get(image_id, set())),
                "chair_boxes": chair_boxes,
                "gt_boxes": gt_boxes_by_image.get(image_id, []),
            }
        )

    candidates.sort(key=lambda item: (item["max_score"], item["num_chair_predictions"]), reverse=True)
    selected = candidates[: args.top_images]

    summary = []
    thumbnails = []
    for rank, item in enumerate(selected, start=1):
        image_path = resolve_image_path(image_dir, item["image_id"])
        image = Image.open(image_path).convert("RGB")
        drawn = draw_debug_image(image, item, max_boxes=args.max_boxes_per_image)
        out_path = singles_dir / f"{rank:03d}_{item['max_score']:.3f}_{item['image_id']}"
        drawn.save(out_path)
        thumbnails.append(drawn)
        summary.append(
            {
                "rank": rank,
                "image_id": item["image_id"],
                "max_chair_score": item["max_score"],
                "num_chair_predictions": item["num_chair_predictions"],
                "gt_classes": item["gt_classes"],
                "output": str(out_path),
            }
        )

    save_grid(thumbnails, output_dir / "chair_false_positive_grid.jpg", columns=args.grid_columns)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Selected {len(selected)} images from {len(candidates)} no-chair images with chair predictions.")
    print(f"Wrote individual visualizations to {singles_dir}")
    print(f"Wrote grid to {output_dir / 'chair_false_positive_grid.jpg'}")
    print(f"Wrote summary to {output_dir / 'summary.json'}")
    for row in summary[:20]:
        print(
            f"#{row['rank']:03d} {row['image_id']} "
            f"score={row['max_chair_score']:.3f} "
            f"chair_preds={row['num_chair_predictions']} "
            f"gt={row['gt_classes']}"
        )


def resolve_image_path(image_dir: Path, image_id: str) -> Path:
    path = image_dir / image_id
    if path.exists():
        return path
    matches = list(image_dir.glob(image_id))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find image: {image_id}")


def draw_debug_image(image: Image.Image, item: dict, max_boxes: int) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    header = (
        f"{item['image_id']} | max chair={item['max_score']:.3f} | "
        f"chair preds={item['num_chair_predictions']} | gt={','.join(item['gt_classes']) or 'empty'}"
    )
    draw.rectangle([0, 0, canvas.width, 18], fill=(255, 255, 255))
    draw.text((4, 4), header, fill=(0, 0, 0), font=font)

    for ann in item["gt_boxes"]:
        color = class_color(ann["class"])
        draw_box(draw, ann["bbox"], color, f"GT {ann['class']}", font)

    for index, box in enumerate(item["chair_boxes"][:max_boxes], start=1):
        label = f"FP chair {float(box['confidence']):.3f}"
        draw_box(draw, box["bbox"], (192, 38, 211), label if index <= 5 else "", font, width=3)

    return canvas


def draw_box(draw: ImageDraw.ImageDraw, bbox: list[float], color: tuple[int, int, int], label: str, font, width: int = 2) -> None:
    coords = [float(v) for v in bbox]
    draw.rectangle(coords, outline=color, width=width)
    if label:
        x1, y1, _, _ = coords
        text_box = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle(text_box, fill=color)
        draw.text((x1, y1), label, fill=(255, 255, 255), font=font)


def class_color(class_name: str) -> tuple[int, int, int]:
    colors = {
        "person": (220, 38, 38),
        "car": (22, 163, 74),
        "dog": (234, 179, 8),
        "cat": (8, 145, 178),
        "chair": (192, 38, 211),
    }
    return colors.get(class_name, (0, 0, 0))


def save_grid(images: list[Image.Image], output_path: Path, columns: int) -> None:
    if not images:
        return
    thumb_size = 320
    rows = (len(images) + columns - 1) // columns
    grid = Image.new("RGB", (columns * thumb_size, rows * thumb_size), "white")
    for index, image in enumerate(images):
        thumb = image.copy()
        thumb.thumbnail((thumb_size, thumb_size))
        tile = Image.new("RGB", (thumb_size, thumb_size), "white")
        tile.paste(thumb, ((thumb_size - thumb.width) // 2, (thumb_size - thumb.height) // 2))
        grid.paste(tile, ((index % columns) * thumb_size, (index // columns) * thumb_size))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)


if __name__ == "__main__":
    main()
