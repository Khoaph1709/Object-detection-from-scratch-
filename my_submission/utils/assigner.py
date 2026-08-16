from __future__ import annotations

from collections import OrderedDict

import torch


REGRESSION_RANGES = {
    "p3": (0, 64),
    "p4": (64, 128),
    "p5": (128, 256),
    "p6": (256, 512),
    "p7": (512, 1_000_000),
}


class FCOSTargetAssigner:
    def __init__(
        self,
        strides: dict[str, int],
        regression_ranges: dict[str, tuple[int, int]] | None = None,
        center_sampling_radius: float = 1.5,
    ) -> None:
        self.strides = strides
        self.regression_ranges = regression_ranges or REGRESSION_RANGES
        self.center_sampling_radius = center_sampling_radius

    def __call__(
        self,
        locations_by_level: OrderedDict[str, torch.Tensor],
        targets: list[dict],
    ) -> dict[str, torch.Tensor]:
        locations = torch.cat(list(locations_by_level.values()), dim=0)
        ranges = self._cat_ranges(locations_by_level, locations.device)
        strides = self._cat_strides(locations_by_level, locations.device)

        labels = []
        reg_targets = []
        centerness = []
        matched_boxes = []

        for target in targets:
            item = self.assign_single_image(locations, ranges, strides, target)
            labels.append(item["labels"])
            reg_targets.append(item["reg_targets"])
            centerness.append(item["centerness"])
            matched_boxes.append(item["matched_boxes"])

        return {
            "labels": torch.stack(labels, dim=0),
            "reg_targets": torch.stack(reg_targets, dim=0),
            "centerness": torch.stack(centerness, dim=0),
            "matched_boxes": torch.stack(matched_boxes, dim=0),
            "locations": locations,
        }

    def assign_single_image(
        self,
        locations: torch.Tensor,
        ranges: torch.Tensor,
        strides: torch.Tensor,
        target: dict,
    ) -> dict[str, torch.Tensor]:
        boxes = target["boxes"].to(locations.device)
        gt_labels = target["labels"].to(locations.device)
        num_locations = locations.shape[0]

        labels = torch.full((num_locations,), -1, dtype=torch.long, device=locations.device)
        reg_targets = torch.zeros((num_locations, 4), dtype=torch.float32, device=locations.device)
        centerness = torch.zeros((num_locations,), dtype=torch.float32, device=locations.device)
        matched_boxes = torch.zeros((num_locations, 4), dtype=torch.float32, device=locations.device)
        resized_h, resized_w = target["resized_size"].to(locations.device).float()
        outside_image = (locations[:, 0] >= resized_w) | (locations[:, 1] >= resized_h)
        labels[outside_image] = -2

        if boxes.numel() == 0:
            return {
                "labels": labels,
                "reg_targets": reg_targets,
                "centerness": centerness,
                "matched_boxes": matched_boxes,
            }

        xs, ys = locations[:, 0], locations[:, 1]
        left = xs[:, None] - boxes[:, 0]
        top = ys[:, None] - boxes[:, 1]
        right = boxes[:, 2] - xs[:, None]
        bottom = boxes[:, 3] - ys[:, None]
        reg = torch.stack((left, top, right, bottom), dim=2)

        inside_box = reg.min(dim=2).values > 0
        max_reg = reg.max(dim=2).values
        inside_range = (max_reg >= ranges[:, [0]]) & (max_reg <= ranges[:, [1]])
        inside_center = self._inside_center(locations, boxes, strides)

        candidate = inside_box & inside_range & inside_center & ~outside_image[:, None]
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        areas = areas[None].repeat(num_locations, 1)
        areas[~candidate] = float("inf")
        min_area, matched_idx = areas.min(dim=1)

        positive = torch.isfinite(min_area)
        if positive.any():
            assigned_reg = reg[torch.arange(num_locations, device=locations.device), matched_idx]
            labels[positive] = gt_labels[matched_idx[positive]]
            reg_targets[positive] = assigned_reg[positive]
            matched_boxes[positive] = boxes[matched_idx[positive]]
            centerness[positive] = compute_centerness_targets(reg_targets[positive])

        return {
            "labels": labels,
            "reg_targets": reg_targets,
            "centerness": centerness,
            "matched_boxes": matched_boxes,
        }

    def _cat_ranges(
        self,
        locations_by_level: OrderedDict[str, torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        ranges = []
        for level, locations in locations_by_level.items():
            min_value, max_value = self.regression_ranges[level]
            ranges.append(
                torch.tensor([min_value, max_value], dtype=torch.float32, device=device)
                .view(1, 2)
                .repeat(locations.shape[0], 1)
            )
        return torch.cat(ranges, dim=0)

    def _cat_strides(
        self,
        locations_by_level: OrderedDict[str, torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        strides = []
        for level, locations in locations_by_level.items():
            strides.append(
                torch.full(
                    (locations.shape[0],),
                    float(self.strides[level]),
                    dtype=torch.float32,
                    device=device,
                )
            )
        return torch.cat(strides, dim=0)

    def _inside_center(
        self,
        locations: torch.Tensor,
        boxes: torch.Tensor,
        strides: torch.Tensor,
    ) -> torch.Tensor:
        xs, ys = locations[:, 0], locations[:, 1]
        centers_x = (boxes[:, 0] + boxes[:, 2]) / 2
        centers_y = (boxes[:, 1] + boxes[:, 3]) / 2
        center_radius = strides[:, None] * self.center_sampling_radius
        x_min = torch.maximum(boxes[:, 0], centers_x - center_radius)
        y_min = torch.maximum(boxes[:, 1], centers_y - center_radius)
        x_max = torch.minimum(boxes[:, 2], centers_x + center_radius)
        y_max = torch.minimum(boxes[:, 3], centers_y + center_radius)
        return (
            (xs[:, None] >= x_min)
            & (xs[:, None] <= x_max)
            & (ys[:, None] >= y_min)
            & (ys[:, None] <= y_max)
        )


def compute_centerness_targets(reg_targets: torch.Tensor) -> torch.Tensor:
    left_right = reg_targets[:, [0, 2]]
    top_bottom = reg_targets[:, [1, 3]]
    centerness = (
        left_right.min(dim=1).values
        / left_right.max(dim=1).values.clamp(min=1e-6)
        * top_bottom.min(dim=1).values
        / top_bottom.max(dim=1).values.clamp(min=1e-6)
    )
    return torch.sqrt(centerness.clamp(min=0))
