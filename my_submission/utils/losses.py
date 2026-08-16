from __future__ import annotations

import torch
import torch.nn.functional as F

from .box_ops import distance_to_boxes, paired_generalized_box_iou
from .locations import flatten_level_values


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "sum",
) -> torch.Tensor:
    logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)
    targets = targets.float()
    prob = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss


class FCOSLoss:
    def __init__(
        self,
        num_classes: int = 5,
        box_weight: float = 2.0,
        centerness_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        chair_class_index: int = 2,
        chair_positive_weight: float = 1.0,
        chair_negative_weight: float = 1.0,
    ) -> None:
        self.num_classes = num_classes
        self.box_weight = box_weight
        self.centerness_weight = centerness_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        # PDF class order is bottle, cup, chair, laptop, backpack; keep this
        # argument only for backward-compatible configs that explicitly tune chair.
        self.chair_class_index = chair_class_index
        self.chair_positive_weight = chair_positive_weight
        self.chair_negative_weight = chair_negative_weight

    def __call__(self, outputs: dict, targets: dict, strides: dict[str, int]) -> dict[str, torch.Tensor]:
        cls_logits = flatten_level_values(outputs["cls_logits"]).float()
        bbox_raw = flatten_level_values(outputs["bbox_regression"]).float()
        centerness_logits = flatten_level_values(outputs["centerness"]).squeeze(-1).float()

        labels = targets["labels"]
        reg_targets = targets["reg_targets"].float()
        centerness_targets = targets["centerness"].float()
        locations = targets["locations"].to(cls_logits.device).float()

        positive = labels >= 0
        valid = labels >= -1
        num_positive = positive.sum().clamp(min=1).float()

        cls_targets = torch.zeros_like(cls_logits)
        if positive.any():
            batch_idx, loc_idx = torch.where(positive)
            cls_targets[batch_idx, loc_idx, labels[positive]] = 1.0

        loss_cls_raw = sigmoid_focal_loss(
            cls_logits[valid],
            cls_targets[valid],
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            reduction="none",
        )
        if (
            0 <= self.chair_class_index < self.num_classes
            and (self.chair_positive_weight != 1.0 or self.chair_negative_weight != 1.0)
        ):
            class_weights = torch.ones_like(loss_cls_raw)
            chair_targets = cls_targets[valid, self.chair_class_index]
            class_weights[:, self.chair_class_index] = torch.where(
                chair_targets > 0,
                torch.full_like(chair_targets, self.chair_positive_weight),
                torch.full_like(chair_targets, self.chair_negative_weight),
            )
            loss_cls_raw = loss_cls_raw * class_weights
        loss_cls = loss_cls_raw.sum() / num_positive

        if positive.any():
            distances = decode_raw_distances(bbox_raw, strides, outputs["bbox_regression"])
            pred_boxes = []
            target_boxes = []
            for batch_index in range(cls_logits.shape[0]):
                pos = positive[batch_index]
                if pos.any():
                    pred_boxes.append(distance_to_boxes(locations[pos], distances[batch_index, pos]))
                    target_boxes.append(distance_to_boxes(locations[pos], reg_targets[batch_index, pos]))
            pred_boxes = torch.cat(pred_boxes, dim=0)
            target_boxes = torch.cat(target_boxes, dim=0)
            giou = paired_generalized_box_iou(pred_boxes, target_boxes)
            loss_box = (1 - giou).sum() / num_positive
            loss_centerness = F.binary_cross_entropy_with_logits(
                centerness_logits[positive],
                centerness_targets[positive],
                reduction="sum",
            ) / num_positive
        else:
            loss_box = bbox_raw.sum() * 0
            loss_centerness = centerness_logits.sum() * 0

        total = loss_cls + self.box_weight * loss_box + self.centerness_weight * loss_centerness
        return {
            "loss": total,
            "loss_cls": loss_cls.detach(),
            "loss_box": loss_box.detach(),
            "loss_centerness": loss_centerness.detach(),
            "num_positive": num_positive.detach(),
        }


def decode_raw_distances(
    bbox_raw_flat: torch.Tensor,
    strides: dict[str, int],
    bbox_raw_by_level: dict[str, torch.Tensor],
) -> torch.Tensor:
    decoded = []
    offset = 0
    for level, raw in bbox_raw_by_level.items():
        _, _, height, width = raw.shape
        count = height * width
        level_raw = bbox_raw_flat[:, offset : offset + count].float()
        level_distances = F.softplus(level_raw).clamp(max=128.0) * float(strides[level])
        decoded.append(level_distances.clamp(max=4096.0))
        offset += count
    return torch.cat(decoded, dim=1)
