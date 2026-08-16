from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from PIL import Image, ImageDraw

from .augmentations import denormalize_image
from .classes import IDX_TO_CLASS


COLORS = {
    "bottle": (220, 38, 38),
    "cup": (22, 163, 74),
    "chair": (234, 179, 8),
    "laptop": (8, 145, 178),
    "backpack": (192, 38, 211),
}


def tensor_to_pil(image: torch.Tensor, normalized: bool = True) -> Image.Image:
    if normalized:
        image = denormalize_image(image.cpu())
    else:
        image = image.cpu().clamp(0, 1)
    image = (image * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(image)


def draw_target(
    image: torch.Tensor,
    target: dict,
    normalized: bool = True,
) -> Image.Image:
    pil_image = tensor_to_pil(image, normalized=normalized)
    draw = ImageDraw.Draw(pil_image)

    boxes = target["boxes"].cpu()
    labels = target["labels"].cpu()
    for box, label in zip(boxes, labels):
        class_name = IDX_TO_CLASS[int(label)]
        color = COLORS[class_name]
        coords = [float(value) for value in box.tolist()]
        draw.rectangle(coords, outline=color, width=3)
        draw.text((coords[0] + 3, coords[1] + 3), class_name, fill=color)

    return pil_image


def save_image_grid(images: Sequence[Image.Image], output_path: str | Path, columns: int = 5) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    thumb_size = 240
    rows = (len(images) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * thumb_size, rows * thumb_size), "white")

    for index, image in enumerate(images):
        image = image.copy()
        image.thumbnail((thumb_size, thumb_size))
        tile = Image.new("RGB", (thumb_size, thumb_size), "white")
        tile.paste(image, ((thumb_size - image.width) // 2, (thumb_size - image.height) // 2))
        canvas.paste(tile, ((index % columns) * thumb_size, (index // columns) * thumb_size))

    canvas.save(output_path)

