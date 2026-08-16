from __future__ import annotations

from typing import Tuple

import torch


def clip_boxes_to_image(boxes: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Clip boxes to an image of shape (height, width)."""
    if boxes.numel() == 0:
        return boxes

    height, width = size
    boxes = boxes.clone()
    boxes[:, 0::2].clamp_(min=0, max=width)
    boxes[:, 1::2].clamp_(min=0, max=height)
    return boxes


def remove_small_boxes(boxes: torch.Tensor, min_size: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    keep = (widths >= min_size) & (heights >= min_size)
    return torch.where(keep)[0]


def resize_boxes(
    boxes: torch.Tensor,
    original_size: Tuple[int, int],
    new_size: Tuple[int, int],
) -> torch.Tensor:
    """Resize [xmin, ymin, xmax, ymax] boxes from original_size to new_size."""
    if boxes.numel() == 0:
        return boxes

    original_height, original_width = original_size
    new_height, new_width = new_size
    ratio_width = new_width / original_width
    ratio_height = new_height / original_height

    boxes = boxes.clone()
    boxes[:, [0, 2]] *= ratio_width
    boxes[:, [1, 3]] *= ratio_height
    return boxes


def horizontal_flip_boxes(boxes: torch.Tensor, width: int) -> torch.Tensor:
    """Flip [xmin, ymin, xmax, ymax] boxes horizontally."""
    if boxes.numel() == 0:
        return boxes

    flipped = boxes.clone()
    flipped[:, 0] = width - boxes[:, 2]
    flipped[:, 2] = width - boxes[:, 0]
    return flipped


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    iou = box_iou(boxes1, boxes2)

    lt = torch.minimum(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.maximum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    enclosing = wh[:, :, 0] * wh[:, :, 1]

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    inter_lt = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    inter_rb = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    inter_wh = (inter_rb - inter_lt).clamp(min=0)
    inter = inter_wh[:, :, 0] * inter_wh[:, :, 1]
    union = area1[:, None] + area2 - inter

    return iou - (enclosing - union) / enclosing.clamp(min=1e-6)


def paired_generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """GIoU for aligned box pairs with shape [N, 4] and [N, 4]."""
    if boxes1.numel() == 0:
        return boxes1.new_zeros((0,))

    inter_lt = torch.maximum(boxes1[:, :2], boxes2[:, :2])
    inter_rb = torch.minimum(boxes1[:, 2:], boxes2[:, 2:])
    inter_wh = (inter_rb - inter_lt).clamp(min=0)
    inter = inter_wh[:, 0] * inter_wh[:, 1]

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    union = area1 + area2 - inter
    iou = inter / union.clamp(min=1e-6)

    enclosing_lt = torch.minimum(boxes1[:, :2], boxes2[:, :2])
    enclosing_rb = torch.maximum(boxes1[:, 2:], boxes2[:, 2:])
    enclosing_wh = (enclosing_rb - enclosing_lt).clamp(min=0)
    enclosing = enclosing_wh[:, 0] * enclosing_wh[:, 1]
    return iou - (enclosing - union) / enclosing.clamp(min=1e-6)


def distance_to_boxes(points: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
    x = points[:, 0]
    y = points[:, 1]
    return torch.stack(
        (
            x - distances[:, 0],
            y - distances[:, 1],
            x + distances[:, 2],
            y + distances[:, 3],
        ),
        dim=1,
    )
