from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

class DetectionDataset(Dataset):
    """Dataset for the assignment JSON format."""

    def __init__(
        self,
        annotation_file: str | Path,
        image_root: str | Path,
        transform=None,
    ) -> None:
        self.annotation_file = Path(annotation_file)
        self.image_root = Path(image_root)
        self.transform = transform

        with self.annotation_file.open("r", encoding="utf-8") as file:
            data = json.load(file)

        self.classes: List[str] = data["classes"]
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        self.images: List[Dict[str, Any]] = data["images"]
        self.annotations_by_image: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for annotation in data["annotations"]:
            self.annotations_by_image[annotation["image_id"]].append(annotation)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image_info = self.images[index]
        image_id = image_info["id"]
        image_path = self._resolve_image_path(image_info)
        image = Image.open(image_path).convert("RGB")

        annotations = self.annotations_by_image.get(image_id, [])
        boxes = torch.tensor(
            [annotation["bbox"] for annotation in annotations],
            dtype=torch.float32,
        )
        if boxes.numel() == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        labels = torch.tensor(
            [self.class_to_idx[annotation["class"]] for annotation in annotations],
            dtype=torch.int64,
        )

        width, height = image.size
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "original_size": torch.tensor([height, width], dtype=torch.int64),
        }

        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target

    def _resolve_image_path(self, image_info: Dict[str, Any]) -> Path:
        direct_path = self.image_root / image_info["id"]
        if direct_path.exists():
            return direct_path

        file_name = Path(image_info["file_name"]).name
        fallback_path = self.image_root / file_name
        if fallback_path.exists():
            return fallback_path

        raise FileNotFoundError(f"Could not find image for {image_info['id']} under {self.image_root}")


def detection_collate_fn(batch):
    images, targets = zip(*batch)
    max_height = max(image.shape[1] for image in images)
    max_width = max(image.shape[2] for image in images)

    padded_images = []
    image_masks = []
    for image in images:
        channels, height, width = image.shape
        padded = image.new_zeros((channels, max_height, max_width))
        padded[:, :height, :width] = image
        mask = torch.ones((max_height, max_width), dtype=torch.bool)
        mask[:height, :width] = False
        padded_images.append(padded)
        image_masks.append(mask)

    return {
        "images": torch.stack(padded_images, dim=0),
        "masks": torch.stack(image_masks, dim=0),
        "targets": list(targets),
    }
