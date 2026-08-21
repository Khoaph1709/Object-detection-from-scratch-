
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


def quality_focal_loss(
    logits: torch.Tensor,
    quality_targets: torch.Tensor,
    gamma: float = 2.0,
    reduction: str = "sum",
) -> torch.Tensor:
    """Quality Focal Loss for class scores whose positive targets are in [0, 1].

    Negative locations use target 0. Positive locations use a detached quality
    target derived from predicted IoU and FCOS centerness. This makes the
    classification score reflect localization quality instead of only class
    identity, while preserving the same tensor shape as ordinary focal loss.
    """
    logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)
    targets = quality_targets.float().clamp(0.0, 1.0)
    probabilities = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    modulation = (probabilities - targets).abs().pow(gamma)
    loss = ce_loss * modulation

    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss


def paired_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """IoU for aligned [N, 4] box pairs without constructing an NxN matrix."""
    if boxes1.numel() == 0:
        return boxes1.new_zeros((0,))
    inter_lt = torch.maximum(boxes1[:, :2], boxes2[:, :2])
    inter_rb = torch.minimum(boxes1[:, 2:], boxes2[:, 2:])
    inter_wh = (inter_rb - inter_lt).clamp(min=0)
    intersection = inter_wh[:, 0] * inter_wh[:, 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1 + area2 - intersection
    return intersection / union.clamp(min=1e-6)


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
        backpack_class_index: int = 4,
        backpack_positive_weight: float = 1.0,
        backpack_negative_weight: float = 1.0,
        quality_aware_cls: bool = False,
        quality_target_floor: float = 0.20,
        quality_blend_start: float = 1.0,
        quality_blend_ramp_epochs: int = 0,
    ) -> None:
        self.num_classes = num_classes
        self.box_weight = box_weight
        self.centerness_weight = centerness_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        # PDF class order is bottle, cup, chair, laptop, backpack; keep these
        # arguments backward-compatible with existing experiment configs.
        self.chair_class_index = chair_class_index
        self.chair_positive_weight = chair_positive_weight
        self.chair_negative_weight = chair_negative_weight
        self.backpack_class_index = backpack_class_index
        self.backpack_positive_weight = backpack_positive_weight
        self.backpack_negative_weight = backpack_negative_weight
        self.quality_aware_cls = quality_aware_cls
        self.quality_target_floor = float(quality_target_floor)
        self.quality_blend_start = float(quality_blend_start)

    def __call__(
        self,
        outputs: dict,
        targets: dict,
        strides: dict[str, int],
        quality_blend_weight: float | None = None,
    ) -> dict[str, torch.Tensor]:
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

        pred_boxes = None
        target_boxes = None
        quality_targets = centerness_targets.clone()
        if positive.any():
            distances = decode_raw_distances(bbox_raw, strides, outputs["bbox_regression"])
            pred_boxes_per_image = []
            target_boxes_per_image = []
            for batch_index in range(cls_logits.shape[0]):
                pos = positive[batch_index]
                if pos.any():
                    pred_boxes_per_image.append(
                        distance_to_boxes(locations[pos], distances[batch_index, pos])
                    )
                    target_boxes_per_image.append(
                        distance_to_boxes(locations[pos], reg_targets[batch_index, pos])
                    )
            pred_boxes = torch.cat(pred_boxes_per_image, dim=0)
            target_boxes = torch.cat(target_boxes_per_image, dim=0)
            if self.quality_aware_cls:
                aligned_iou = paired_box_iou(pred_boxes.detach(), target_boxes.detach()).clamp(0.0, 1.0)
                aligned_centerness = centerness_targets[positive].detach().clamp(0.0, 1.0)
                quality = torch.sqrt(aligned_iou * aligned_centerness)
                quality = self.quality_target_floor + (1.0 - self.quality_target_floor) * quality
                quality_targets[positive] = quality

        cls_targets = torch.zeros_like(cls_logits)
        if positive.any():
            batch_idx, loc_idx = torch.where(positive)
            positive_targets = quality_targets[positive] if self.quality_aware_cls else torch.ones_like(quality_targets[positive])
            cls_targets[batch_idx, loc_idx, labels[positive]] = positive_targets

        binary_cls_targets = torch.zeros_like(cls_logits)
        if positive.any():
            batch_idx, loc_idx = torch.where(positive)
            binary_cls_targets[batch_idx, loc_idx, labels[positive]] = 1.0
        binary_cls_raw = sigmoid_focal_loss(
            cls_logits[valid],
            binary_cls_targets[valid],
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            reduction="none",
        )
        if self.quality_aware_cls:
            quality_cls_raw = quality_focal_loss(
                cls_logits[valid],
                cls_targets[valid],
                gamma=self.focal_gamma,
                reduction="none",
            )
            if quality_blend_weight is None:
                quality_blend_weight = self.quality_blend_start
            quality_blend_weight = float(max(0.0, min(1.0, quality_blend_weight)))
            loss_cls_raw = (1.0 - quality_blend_weight) * binary_cls_raw + quality_blend_weight * quality_cls_raw
        else:
            quality_blend_weight = 0.0
            loss_cls_raw = binary_cls_raw

        class_weight_specs = [
            (
                self.chair_class_index,
                self.chair_positive_weight,
                self.chair_negative_weight,
            ),
            (
                self.backpack_class_index,
                self.backpack_positive_weight,
                self.backpack_negative_weight,
            ),
        ]
        if any(
            0 <= class_index < self.num_classes
            and (positive_weight != 1.0 or negative_weight != 1.0)
            for class_index, positive_weight, negative_weight in class_weight_specs
        ):
            class_weights = torch.ones_like(loss_cls_raw)
            for class_index, positive_weight, negative_weight in class_weight_specs:
                if not (0 <= class_index < self.num_classes):
                    continue
                targets_for_class = cls_targets[valid, class_index]
                class_weights[:, class_index] = torch.where(
                    targets_for_class > 0,
                    torch.full_like(targets_for_class, positive_weight),
                    torch.full_like(targets_for_class, negative_weight),
                )
            loss_cls_raw = loss_cls_raw * class_weights
        loss_cls = loss_cls_raw.sum() / num_positive

        if positive.any():
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
        result = {
            "loss": total,
            "loss_cls": loss_cls.detach(),
            "loss_box": loss_box.detach(),
            "loss_centerness": loss_centerness.detach(),
            "num_positive": num_positive.detach(),
            "quality_blend_weight": cls_logits.new_tensor(float(quality_blend_weight)),
        }
        if self.quality_aware_cls and positive.any():
            result["quality_target_mean"] = quality_targets[positive].mean().detach()
        return result


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
