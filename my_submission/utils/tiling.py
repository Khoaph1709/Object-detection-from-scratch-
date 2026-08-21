from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torch.utils.data import Dataset

from .augmentations import DetectionTransform


@dataclass(frozen=True)
class TileSpec:
    top: int
    left: int
    height: int
    width: int


def _axis_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, max(length - tile_size, 0) + 1, stride))
    last = length - tile_size
    if not starts or starts[-1] != last:
        starts.append(last)
    return sorted(set(starts))


def generate_tile_specs(
    image_height: int,
    image_width: int,
    tile_size: int = 640,
    overlap: float = 0.20,
) -> list[TileSpec]:
    """Generate deterministic, coverage-complete tiles in original-image coordinates."""
    tile_size = max(2, int(tile_size))
    overlap = min(0.9, max(0.0, float(overlap)))
    stride = max(1, int(round(tile_size * (1.0 - overlap))))
    rows = _axis_starts(int(image_height), tile_size, stride)
    cols = _axis_starts(int(image_width), tile_size, stride)
    return [
        TileSpec(
            top=top,
            left=left,
            height=min(tile_size, int(image_height) - top),
            width=min(tile_size, int(image_width) - left),
        )
        for top in rows
        for left in cols
    ]


class TiledImageDataset(Dataset):
    """Inference dataset that returns resized tiles with original-image metadata."""

    def __init__(
        self,
        image_dir: str | Path,
        transform: DetectionTransform,
        tile_size: int = 640,
        tile_overlap: float = 0.20,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.paths = sorted(
            path for path in self.image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        self.records: list[tuple[Path, int, int, TileSpec]] = []
        for path in self.paths:
            with Image.open(path) as image:
                width, height = image.size
            specs = generate_tile_specs(height, width, tile_size=tile_size, overlap=tile_overlap)
            self.records.extend((path, height, width, spec) for spec in specs)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        path, original_height, original_width, spec = self.records[index]
        with Image.open(path) as source:
            image = source.convert("RGB")
            tile = image.crop((spec.left, spec.top, spec.left + spec.width, spec.top + spec.height))
        target = {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "image_id": path.name,
        }
        image_tensor, target = self.transform(tile, target)
        target["original_size"] = torch.tensor([original_height, original_width], dtype=torch.int64)
        target["crop_size"] = torch.tensor([spec.height, spec.width], dtype=torch.int64)
        target["crop_offset"] = torch.tensor([spec.top, spec.left], dtype=torch.int64)
        target["tile_spec"] = torch.tensor(
            [spec.top, spec.left, spec.height, spec.width], dtype=torch.int64
        )
        return image_tensor, target


def group_tile_predictions(predictions: Iterable[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for item in predictions:
        grouped.setdefault(str(item["image_id"]), []).extend(item.get("boxes", []))
    return grouped
