from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from PIL import Image, ImageDraw, ImageEnhance

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
    small_object_crop_prob: float = 0.0
    small_object_area_threshold: float = 0.05
    small_object_crop_context: float = 0.75
    small_object_crop_min_size: int = 160
    small_object_crop_min_visible_fraction: float = 0.5
    random_erasing_prob: float = 0.0
    random_erasing_area_range: Tuple[float, float] = (0.01, 0.04)
    random_erasing_aspect_range: Tuple[float, float] = (0.3, 3.3)
    random_erasing_max_gt_overlap: float = 0.05
    random_erasing_attempts: int = 20

    def __call__(self, image: Image.Image, target: Dict[str, torch.Tensor]):
        image = image.convert("RGB")
        original_width, original_height = image.size

        target = dict(target)
        crop_offset = (0, 0)
        if self.train and self.small_object_crop_prob > 0.0:
            image, target, crop_offset = self._maybe_small_object_crop(image, target)

        source_width, source_height = image.size
        short_size = self._sample_short_size()
        new_height, new_width = self._get_resized_size(
            source_height,
            source_width,
            short_size,
            self.max_size,
        )

        image = image.resize((new_width, new_height), Image.BILINEAR)
        target["boxes"] = resize_boxes(
            target["boxes"],
            original_size=(source_height, source_width),
            new_size=(new_height, new_width),
        )

        if self.train and self.random_erasing_prob > 0.0:
            image = self._maybe_random_erase(image, target)

        if self.train and self.color_jitter > 0:
            image = self._apply_color_jitter(image)

        if self.train and random.random() < self.horizontal_flip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            target["boxes"] = horizontal_flip_boxes(target["boxes"], new_width)

        target["boxes"] = clip_boxes_to_image(target["boxes"], (new_height, new_width))
        target["resized_size"] = torch.tensor([new_height, new_width], dtype=torch.int64)
        target["original_size"] = torch.tensor([original_height, original_width], dtype=torch.int64)
        target["crop_size"] = torch.tensor([source_height, source_width], dtype=torch.int64)
        target["crop_offset"] = torch.tensor(crop_offset, dtype=torch.int64)

        image_tensor = self._to_tensor(image)
        if self.normalize:
            image_tensor = (image_tensor - IMAGENET_MEAN) / IMAGENET_STD

        return image_tensor, target

    def _maybe_small_object_crop(self, image: Image.Image, target: Dict[str, torch.Tensor]):
        if random.random() >= min(1.0, max(0.0, self.small_object_crop_prob)):
            return image, target, (0, 0)

        boxes = target.get("boxes")
        labels = target.get("labels")
        if boxes is None or labels is None or boxes.numel() == 0:
            return image, target, (0, 0)

        image_width, image_height = image.size
        image_area = float(image_width * image_height)
        widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
        heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
        area_ratio = widths * heights / max(image_area, 1.0)
        candidate_indices = torch.where(
            (area_ratio <= max(0.0, self.small_object_area_threshold))
            & (widths >= 2.0)
            & (heights >= 2.0)
        )[0]
        if candidate_indices.numel() == 0:
            return image, target, (0, 0)

        selected = int(candidate_indices[random.randrange(candidate_indices.numel())].item())
        x1, y1, x2, y2 = [float(value) for value in boxes[selected].tolist()]
        box_width = max(x2 - x1, 2.0)
        box_height = max(y2 - y1, 2.0)
        context = max(0.0, float(self.small_object_crop_context)) * max(box_width, box_height)
        crop_width = int(round(max(box_width + 2.0 * context, float(self.small_object_crop_min_size))))
        crop_height = int(round(max(box_height + 2.0 * context, float(self.small_object_crop_min_size))))
        crop_width = min(max(crop_width, 2), image_width)
        crop_height = min(max(crop_height, 2), image_height)

        center_x = 0.5 * (x1 + x2)
        center_y = 0.5 * (y1 + y2)
        left = int(round(center_x - crop_width / 2.0))
        top = int(round(center_y - crop_height / 2.0))
        left = min(max(left, 0), image_width - crop_width)
        top = min(max(top, 0), image_height - crop_height)
        right, bottom = left + crop_width, top + crop_height

        cropped_image = image.crop((left, top, right, bottom))
        cropped_target = dict(target)
        cropped_boxes = boxes.clone()
        cropped_boxes[:, [0, 2]] -= float(left)
        cropped_boxes[:, [1, 3]] -= float(top)
        cropped_boxes = clip_boxes_to_image(cropped_boxes, (crop_height, crop_width))
        original_areas = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (
            boxes[:, 3] - boxes[:, 1]
        ).clamp(min=0)
        visible_areas = (cropped_boxes[:, 2] - cropped_boxes[:, 0]).clamp(min=0) * (
            cropped_boxes[:, 3] - cropped_boxes[:, 1]
        ).clamp(min=0)
        visible_fraction = visible_areas / original_areas.clamp(min=1e-6)
        keep = (cropped_boxes[:, 2] > cropped_boxes[:, 0] + 1.0) & (
            cropped_boxes[:, 3] > cropped_boxes[:, 1] + 1.0
        ) & (
            (visible_fraction >= max(0.0, min(1.0, self.small_object_crop_min_visible_fraction)))
            | (torch.arange(boxes.shape[0], device=boxes.device) == selected)
        )
        cropped_target["boxes"] = cropped_boxes[keep]
        cropped_target["labels"] = labels[keep]
        return cropped_image, cropped_target, (top, left)

    def _maybe_random_erase(self, image: Image.Image, target: Dict[str, torch.Tensor]) -> Image.Image:
        if random.random() >= min(1.0, max(0.0, self.random_erasing_prob)):
            return image

        width, height = image.size
        image_area = float(max(width * height, 1))
        boxes = target.get("boxes")
        if boxes is None:
            boxes = torch.empty((0, 4), dtype=torch.float32)
        boxes = boxes.detach().cpu().float()
        area_min, area_max = sorted(
            (max(0.0, float(self.random_erasing_area_range[0])), max(0.0, float(self.random_erasing_area_range[1])))
        )
        aspect_min, aspect_max = sorted(
            (max(1e-3, float(self.random_erasing_aspect_range[0])), max(1e-3, float(self.random_erasing_aspect_range[1])))
        )

        for _ in range(max(1, int(self.random_erasing_attempts))):
            target_area = random.uniform(area_min, max(area_min, area_max)) * image_area
            aspect_ratio = math.exp(random.uniform(math.log(aspect_min), math.log(aspect_max)))
            erase_width = max(1, int(round(math.sqrt(target_area * aspect_ratio))))
            erase_height = max(1, int(round(math.sqrt(target_area / aspect_ratio))))
            if erase_width >= width or erase_height >= height:
                continue
            left = random.randint(0, width - erase_width)
            top = random.randint(0, height - erase_height)
            erase_box = torch.tensor([left, top, left + erase_width, top + erase_height], dtype=torch.float32)
            if boxes.numel() > 0:
                inter_lt = torch.maximum(boxes[:, :2], erase_box[:2])
                inter_rb = torch.minimum(boxes[:, 2:], erase_box[2:])
                inter_wh = (inter_rb - inter_lt).clamp(min=0)
                intersection = inter_wh[:, 0] * inter_wh[:, 1]
                gt_area = ((boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)).clamp(min=1e-6)
                if bool((intersection / gt_area).max().item() > max(0.0, float(self.random_erasing_max_gt_overlap))):
                    continue
            erased = image.copy()
            draw = ImageDraw.Draw(erased)
            draw.rectangle((left, top, left + erase_width - 1, top + erase_height - 1), fill=(0, 0, 0))
            return erased
        return image

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
