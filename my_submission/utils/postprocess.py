from __future__ import annotations

from collections import OrderedDict

import torch

try:
    from torchvision.ops import nms as torchvision_nms
except (ImportError, RuntimeError):
    torchvision_nms = None

from .box_ops import box_iou, clip_boxes_to_image, distance_to_boxes
from .classes import IDX_TO_CLASS
from .locations import flatten_level_values, generate_locations
from .losses import decode_raw_distances


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    if torchvision_nms is not None:
        try:
            return torchvision_nms(boxes, scores, iou_threshold)
        except (RuntimeError, NotImplementedError):
            pass

    x1, y1, x2, y2 = boxes.unbind(dim=1)
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        index = order[0]
        keep.append(index)
        if order.numel() == 1:
            break

        rest = order[1:]
        xx1 = torch.maximum(x1[index], x1[rest])
        yy1 = torch.maximum(y1[index], y1[rest])
        xx2 = torch.minimum(x2[index], x2[rest])
        yy2 = torch.minimum(y2[index], y2[rest])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        union = areas[index] + areas[rest] - inter
        iou = inter / union.clamp(min=1e-6)
        order = rest[iou <= iou_threshold]

    return torch.stack(keep)


def decode_detections(
    outputs: dict,
    image_sizes: list[tuple[int, int]],
    strides: dict[str, int],
    score_threshold: float = 0.05,
    nms_threshold: float = 0.55,
    max_detections_per_image: int = 100,
    pre_nms_topk: int = 1000,
    score_cls_power: float = 0.5,
    score_centerness_power: float = 0.5,
    class_score_thresholds: dict[str, float] | None = None,
) -> list[list[dict]]:
    features = outputs["features"]
    locations_by_level = generate_locations(features, strides)
    locations = torch.cat(list(locations_by_level.values()), dim=0)

    cls_logits = flatten_level_values(outputs["cls_logits"])
    bbox_raw = flatten_level_values(outputs["bbox_regression"])
    centerness_logits = flatten_level_values(outputs["centerness"]).squeeze(-1)
    distances = decode_raw_distances(bbox_raw, strides, outputs["bbox_regression"])

    class_scores = torch.sigmoid(cls_logits)
    center_scores = torch.sigmoid(centerness_logits).unsqueeze(-1)
    scores = class_scores.clamp(min=1e-8).pow(score_cls_power) * center_scores.clamp(min=1e-8).pow(
        score_centerness_power
    )

    class_score_thresholds = class_score_thresholds or {}
    results = []
    for batch_index, image_size in enumerate(image_sizes):
        boxes = distance_to_boxes(locations, distances[batch_index])
        boxes = clip_boxes_to_image(boxes, image_size)

        image_results = []
        for class_index in range(scores.shape[-1]):
            class_scores_i = scores[batch_index, :, class_index]
            class_name = IDX_TO_CLASS[int(class_index)]
            class_threshold = float(class_score_thresholds.get(class_name, score_threshold))
            keep = class_scores_i >= class_threshold
            if keep.sum() == 0:
                continue

            class_boxes = boxes[keep]
            class_scores_kept = class_scores_i[keep]
            valid_boxes = (class_boxes[:, 2] > class_boxes[:, 0] + 1) & (
                class_boxes[:, 3] > class_boxes[:, 1] + 1
            )
            class_boxes = class_boxes[valid_boxes]
            class_scores_kept = class_scores_kept[valid_boxes]
            if class_boxes.numel() == 0:
                continue
            if pre_nms_topk > 0 and class_scores_kept.numel() > pre_nms_topk:
                topk_scores, topk_idx = class_scores_kept.topk(pre_nms_topk)
                class_boxes = class_boxes[topk_idx]
                class_scores_kept = topk_scores
            keep_nms = nms(class_boxes, class_scores_kept, nms_threshold)
            for item in keep_nms:
                image_results.append(
                    {
                        "class_index": int(class_index),
                        "class": IDX_TO_CLASS[int(class_index)],
                        "confidence": float(class_scores_kept[item].clamp(0, 1).item()),
                        "bbox": [float(v) for v in class_boxes[item].tolist()],
                    }
                )

        image_results.sort(key=lambda item: item["confidence"], reverse=True)
        results.append(image_results[:max_detections_per_image])

    return results


def scale_detections_to_original(
    detections: list[dict],
    resized_size: tuple[int, int],
    original_size: tuple[int, int],
) -> list[dict]:
    resized_h, resized_w = resized_size
    original_h, original_w = original_size
    scale_x = original_w / resized_w
    scale_y = original_h / resized_h
    scaled = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        bbox = [
            max(0.0, min(original_w, x1 * scale_x)),
            max(0.0, min(original_h, y1 * scale_y)),
            max(0.0, min(original_w, x2 * scale_x)),
            max(0.0, min(original_h, y2 * scale_y)),
        ]
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        scaled.append(
            {
                "class": det["class"],
                "confidence": det["confidence"],
                "bbox": bbox,
            }
        )
    return scaled


def flip_detections_horizontal(detections: list[dict], width: int) -> list[dict]:
    flipped = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        updated = dict(det)
        updated["bbox"] = [width - x2, y1, width - x1, y2]
        flipped.append(updated)
    return flipped


def merge_detections(
    detections: list[dict],
    image_size: tuple[int, int],
    nms_threshold: float = 0.55,
    max_detections_per_image: int = 100,
) -> list[dict]:
    if not detections:
        return []

    merged = []
    class_names = sorted({det["class"] for det in detections})
    for class_name in class_names:
        class_detections = [det for det in detections if det["class"] == class_name]
        boxes = torch.tensor([det["bbox"] for det in class_detections], dtype=torch.float32)
        scores = torch.tensor([det["confidence"] for det in class_detections], dtype=torch.float32)
        boxes = clip_boxes_to_image(boxes, image_size)
        valid = (boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
        boxes = boxes[valid]
        scores = scores[valid]
        if boxes.numel() == 0:
            continue

        keep = nms(boxes, scores, nms_threshold)
        for index in keep:
            merged.append(
                {
                    "class": class_name,
                    "confidence": float(scores[index].clamp(0, 1).item()),
                    "bbox": [float(v) for v in boxes[index].tolist()],
                }
            )

    merged.sort(key=lambda item: item["confidence"], reverse=True)
    return merged[:max_detections_per_image]


def merge_detections_consensus(
    primary_detections: list[dict],
    secondary_detections: list[dict],
    image_size: tuple[int, int],
    match_iou_threshold: float = 0.6,
    nms_threshold: float = 0.55,
    max_detections_per_image: int = 70,
    single_view_score_factor: float = 0.6,
    matched_score_factor: float = 1.0,
    class_score_thresholds: dict[str, float] | None = None,
    class_topk: dict[str, int] | None = None,
) -> list[dict]:
    """Fuse two TTA views, rewarding cross-view agreement and downweighting one-view boxes."""
    if not primary_detections and not secondary_detections:
        return []

    class_score_thresholds = class_score_thresholds or {}
    class_topk = class_topk or {}
    fused = []
    class_names = sorted(
        {det["class"] for det in primary_detections}
        | {det["class"] for det in secondary_detections}
    )
    for class_name in class_names:
        primary = [det for det in primary_detections if det["class"] == class_name]
        secondary = [det for det in secondary_detections if det["class"] == class_name]
        fused.extend(
            _fuse_class_detections(
                class_name=class_name,
                primary=primary,
                secondary=secondary,
                image_size=image_size,
                match_iou_threshold=match_iou_threshold,
                single_view_score_factor=single_view_score_factor,
                matched_score_factor=matched_score_factor,
            )
        )

    if not fused:
        return []

    merged = []
    for class_name in sorted({det["class"] for det in fused}):
        class_detections = [det for det in fused if det["class"] == class_name]
        threshold = class_score_thresholds.get(class_name, 0.0)
        class_detections = [
            det for det in class_detections if float(det["confidence"]) >= threshold
        ]
        if not class_detections:
            continue

        boxes = torch.tensor([det["bbox"] for det in class_detections], dtype=torch.float32)
        scores = torch.tensor([det["confidence"] for det in class_detections], dtype=torch.float32)
        boxes = clip_boxes_to_image(boxes, image_size)
        valid = (boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
        boxes = boxes[valid]
        scores = scores[valid]
        if boxes.numel() == 0:
            continue

        keep = nms(boxes, scores, nms_threshold)
        class_limit = class_topk.get(class_name, 0)
        if class_limit > 0:
            keep = keep[:class_limit]
        for index in keep:
            merged.append(
                {
                    "class": class_name,
                    "confidence": float(scores[index].clamp(0, 1).item()),
                    "bbox": [float(v) for v in boxes[index].tolist()],
                }
            )

    merged.sort(key=lambda item: item["confidence"], reverse=True)
    return merged[:max_detections_per_image]


def _fuse_class_detections(
    class_name: str,
    primary: list[dict],
    secondary: list[dict],
    image_size: tuple[int, int],
    match_iou_threshold: float,
    single_view_score_factor: float,
    matched_score_factor: float,
) -> list[dict]:
    if not primary:
        return [_penalize_single_view(det, single_view_score_factor) for det in secondary]
    if not secondary:
        return [_penalize_single_view(det, single_view_score_factor) for det in primary]

    primary_boxes = clip_boxes_to_image(
        torch.tensor([det["bbox"] for det in primary], dtype=torch.float32),
        image_size,
    )
    secondary_boxes = clip_boxes_to_image(
        torch.tensor([det["bbox"] for det in secondary], dtype=torch.float32),
        image_size,
    )
    primary_scores = torch.tensor([det["confidence"] for det in primary], dtype=torch.float32)
    secondary_scores = torch.tensor([det["confidence"] for det in secondary], dtype=torch.float32)
    ious = box_iou(primary_boxes, secondary_boxes)

    primary_order = primary_scores.argsort(descending=True).tolist()
    used_secondary: set[int] = set()
    fused = []
    for primary_index in primary_order:
        available = [
            secondary_index
            for secondary_index in range(len(secondary))
            if secondary_index not in used_secondary
        ]
        best_secondary_index = -1
        best_iou = -1.0
        for secondary_index in available:
            iou = float(ious[primary_index, secondary_index].item())
            if iou > best_iou:
                best_iou = iou
                best_secondary_index = secondary_index

        if best_secondary_index >= 0 and best_iou >= match_iou_threshold:
            used_secondary.add(best_secondary_index)
            score_a = primary_scores[primary_index]
            score_b = secondary_scores[best_secondary_index]
            total_score = (score_a + score_b).clamp(min=1e-6)
            box = (
                primary_boxes[primary_index] * score_a
                + secondary_boxes[best_secondary_index] * score_b
            ) / total_score
            score = ((score_a + score_b) * 0.5 * matched_score_factor).clamp(0, 1)
            fused.append(
                {
                    "class": class_name,
                    "confidence": float(score.item()),
                    "bbox": [float(v) for v in box.tolist()],
                }
            )
        else:
            fused.append(_penalize_single_view(primary[primary_index], single_view_score_factor))

    for secondary_index, det in enumerate(secondary):
        if secondary_index not in used_secondary:
            fused.append(_penalize_single_view(det, single_view_score_factor))

    return fused


def _penalize_single_view(det: dict, factor: float) -> dict:
    updated = dict(det)
    updated["confidence"] = float(max(0.0, min(1.0, float(det["confidence"]) * factor)))
    return updated
