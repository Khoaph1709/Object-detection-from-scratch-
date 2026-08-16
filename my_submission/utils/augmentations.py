from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from PIL import Image, ImageEnhance

from .box_ops import clip_boxes_to_image, horizontal_flip_boxes, resize_boxes


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


@dataclass
class DetectionTransform:
    train: bool
    short_size: int | Tuple[int, ...] = 640
    max_size: int = 896
    horizontal_flip_prob: float = 0.5
    color_jitter: float = 0.0
    normalize: bool = True

    def __call__(self, image: Image.Image, target: Dict[str, torch.Tensor]):
        image = image.convert("RGB")
        original_width, original_height = image.size

        short_size = self._sample_short_size()
        new_height, new_width = self._get_resized_size(
            original_height,
            original_width,
            short_size,
            self.max_size,
        )

        image = image.resize((new_width, new_height), Image.BILINEAR)
        target = dict(target)
        target["boxes"] = resize_boxes(
            target["boxes"],
            original_size=(original_height, original_width),
            new_size=(new_height, new_width),
        )

        if self.train and self.color_jitter > 0:
            image = self._apply_color_jitter(image)

        if self.train and random.random() < self.horizontal_flip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            target["boxes"] = horizontal_flip_boxes(target["boxes"], new_width)

        target["boxes"] = clip_boxes_to_image(target["boxes"], (new_height, new_width))
        target["resized_size"] = torch.tensor([new_height, new_width], dtype=torch.int64)
        target["original_size"] = torch.tensor([original_height, original_width], dtype=torch.int64)

        image_tensor = self._to_tensor(image)
        if self.normalize:
            image_tensor = (image_tensor - IMAGENET_MEAN) / IMAGENET_STD

        return image_tensor, target

    def _sample_short_size(self) -> int:
        if isinstance(self.short_size, int):
            return self.short_size
        return random.choice(tuple(self.short_size))

    @staticmethod
    def _get_resized_size(
        original_height: int,
        original_width: int,
        short_size: int,
        max_size: int,
    ) -> Tuple[int, int]:
        min_original = min(original_height, original_width)
        max_original = max(original_height, original_width)

        scale = short_size / min_original
        if round(max_original * scale) > max_size:
            scale = max_size / max_original

        new_height = int(round(original_height * scale))
        new_width = int(round(original_width * scale))
        return new_height, new_width

    def _apply_color_jitter(self, image: Image.Image) -> Image.Image:
        strength = self.color_jitter
        transforms = [
            ImageEnhance.Brightness,
            ImageEnhance.Contrast,
            ImageEnhance.Color,
        ]
        random.shuffle(transforms)
        for enhancer in transforms:
            factor = random.uniform(max(0.0, 1.0 - strength), 1.0 + strength)
            image = enhancer(image).enhance(factor)
        return image

    @staticmethod
    def _to_tensor(image: Image.Image) -> torch.Tensor:
        tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        tensor = tensor.view(image.size[1], image.size[0], 3)
        tensor = tensor.permute(2, 0, 1).contiguous().float()
        return tensor.div(255.0)


def denormalize_image(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)
