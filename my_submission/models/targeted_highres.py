from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn


class ConvGNAct(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualUpdate(nn.Module):
    """Small residual branch whose final projection starts at zero."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.features = ConvGNAct(channels)
        self.output = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.features(x))


class TargetedHighResNeck(nn.Module):
    """Gated bidirectional P2/P3 refinement with identity initialization.

    The module preserves P4-P7 and starts very close to identity so a baseline
    FCOS checkpoint can be warm-started without a sudden prediction shift.
    """

    def __init__(self, channels: int = 256) -> None:
        super().__init__()
        self.p3_to_p2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(32, channels),
            nn.SiLU(inplace=True),
        )
        self.p2_to_p3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, channels),
            nn.SiLU(inplace=True),
        )
        self.p2_update = ResidualUpdate(channels)
        self.p3_update = ResidualUpdate(channels)
        self.p2_gate = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.p3_gate = nn.Conv2d(channels * 2, channels, kernel_size=1)
        nn.init.zeros_(self.p2_gate.weight)
        nn.init.constant_(self.p2_gate.bias, -4.0)
        nn.init.zeros_(self.p3_gate.weight)
        nn.init.constant_(self.p3_gate.bias, -4.0)

    def forward(self, features: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
        if "p2" not in features or "p3" not in features:
            raise KeyError("TargetedHighResNeck requires p2 and p3 features")

        p2 = features["p2"]
        p3 = features["p3"]
        p2_context = F.interpolate(
            self.p3_to_p2(p3), size=p2.shape[-2:], mode="bilinear", align_corners=False
        )
        p2_gate = torch.sigmoid(self.p2_gate(torch.cat([p2, p2_context], dim=1)))
        p2_refined = p2 + p2_gate * self.p2_update(p2_context)

        p3_context = self.p2_to_p3(p2_refined)
        if p3_context.shape[-2:] != p3.shape[-2:]:
            p3_context = F.interpolate(p3_context, size=p3.shape[-2:], mode="bilinear", align_corners=False)
        p3_gate = torch.sigmoid(self.p3_gate(torch.cat([p3, p3_context], dim=1)))
        p3_refined = p3 + p3_gate * self.p3_update(p3_context)

        output = OrderedDict(features)
        output["p2"] = p2_refined
        output["p3"] = p3_refined
        return output


class SmallObjectResidualHead(nn.Module):
    """Zero-initialized P2/P3 residual FCOS head.

    Its outputs are added to the existing FCOS head logits/distances, so a
    baseline checkpoint remains functionally unchanged at initialization.
    """

    def __init__(self, channels: int = 256, num_classes: int = 5, num_convs: int = 2) -> None:
        super().__init__()
        cls_layers: list[nn.Module] = []
        reg_layers: list[nn.Module] = []
        for _ in range(num_convs):
            cls_layers.extend(
                [
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                    nn.GroupNorm(32, channels),
                    nn.SiLU(inplace=True),
                ]
            )
            reg_layers.extend(
                [
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                    nn.GroupNorm(32, channels),
                    nn.SiLU(inplace=True),
                ]
            )
        self.cls_tower = nn.Sequential(*cls_layers)
        self.reg_tower = nn.Sequential(*reg_layers)
        self.cls_delta = nn.Conv2d(channels, num_classes, kernel_size=3, padding=1)
        self.bbox_delta = nn.Conv2d(channels, 4, kernel_size=3, padding=1)
        self.centerness_delta = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        for module in (self.cls_delta, self.bbox_delta, self.centerness_delta):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self, features: OrderedDict[str, torch.Tensor]
    ) -> dict[str, OrderedDict[str, torch.Tensor]]:
        cls_delta = OrderedDict()
        bbox_delta = OrderedDict()
        centerness_delta = OrderedDict()
        for level, feature in features.items():
            cls_delta[level] = self.cls_delta(self.cls_tower(feature))
            reg_feature = self.reg_tower(feature)
            bbox_delta[level] = self.bbox_delta(reg_feature)
            centerness_delta[level] = self.centerness_delta(reg_feature)
        return {
            "cls_logits": cls_delta,
            "bbox_regression": bbox_delta,
            "centerness": centerness_delta,
        }
