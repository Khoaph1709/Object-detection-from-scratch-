"""Merge train and validation annotations for an optional final fine-tune.

This script is intentionally explicit: it only merges the two supplied labeled
splits, rejects conflicting image IDs/classes, and never reads hidden-test data.
The resulting JSON keeps each image's original relative file_name, so the
training Dataset can resolve both train/images and val/images from the common
``public/`` image root.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

EXPECTED_CLASSES = ["bottle", "cup", "chair", "laptop", "backpack"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Optional public/ root used to verify every referenced image.",
    )
    return parser.parse_args()


def load_split(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Annotation file must contain a JSON object: {path}")
    classes = payload.get("classes")
    if classes != EXPECTED_CLASSES:
        raise ValueError(f"Unexpected class order in {path}: {classes!r}")
    if not isinstance(payload.get("images"), list) or not isinstance(payload.get("annotations"), list):
        raise ValueError(f"Annotation file needs images and annotations lists: {path}")
    return payload


def merge_splits(train_path: Path, valid_path: Path, output_path: Path, image_root: Path | None = None) -> dict[str, Any]:
    train = load_split(train_path.resolve())
    valid = load_split(valid_path.resolve())

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    image_by_id: dict[str, dict[str, Any]] = {}
    annotation_count_by_split: dict[str, int] = {}

    for split_name, payload in (("train", train), ("valid", valid)):
        annotation_count_by_split[split_name] = len(payload["annotations"])
        for image in payload["images"]:
            if not isinstance(image, dict) or not image.get("id") or not image.get("file_name"):
                raise ValueError(f"Invalid image record in {split_name} split")
            image_id = str(image["id"])
            if image_id in image_by_id:
                raise ValueError(f"Duplicate image ID across splits: {image_id}")
            image_copy = dict(image)
            image_copy["id"] = image_id
            image_by_id[image_id] = image_copy
            images.append(image_copy)

        for annotation in payload["annotations"]:
            if not isinstance(annotation, dict):
                raise ValueError(f"Invalid annotation record in {split_name} split")
            image_id = str(annotation.get("image_id", ""))
            if image_id not in image_by_id:
                raise ValueError(f"Annotation references unknown image {image_id!r} in {split_name} split")
            if annotation.get("class") not in EXPECTED_CLASSES:
                raise ValueError(f"Unexpected annotation class in {split_name} split: {annotation.get('class')!r}")
            annotations.append(dict(annotation))

    if image_root is not None:
        image_root = image_root.resolve()
        missing: list[str] = []
        for image in images:
            candidate = image_root / str(image["file_name"])
            if not candidate.is_file():
                candidate = image_root / Path(str(image["file_name"])).name
            if not candidate.is_file():
                missing.append(str(candidate))
        if missing:
            preview = "\n".join(missing[:20])
            suffix = "" if len(missing) <= 20 else f"\n... and {len(missing) - 20} more"
            raise FileNotFoundError(f"Merged dataset has missing images:\n{preview}{suffix}")

    result = {
        "classes": EXPECTED_CLASSES,
        "images": images,
        "annotations": annotations,
        "metadata": {
            "merged_from": [str(train_path.resolve()), str(valid_path.resolve())],
            "train_images": len(train["images"]),
            "valid_images": len(valid["images"]),
            "train_annotations": annotation_count_by_split["train"],
            "valid_annotations": annotation_count_by_split["valid"],
            "warning": "For final fine-tuning only; do not use this split as an independent validation score.",
        },
    }
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return result


def main() -> None:
    args = parse_args()
    result = merge_splits(args.train, args.valid, args.output, args.image_root)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "images": len(result["images"]),
        "annotations": len(result["annotations"]),
        "classes": result["classes"],
    }, indent=2))


if __name__ == "__main__":
    main()
