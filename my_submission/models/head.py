from __future__ import annotations

import math
from collections import OrderedDict

import torch
from torch import nn


class FCOSHead(nn.Module):
    """Dense FCOS-style detection head for every FPN level."""

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 5,
        num_convs: int = 4,
        prior_probability: float = 0.01,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        cls_layers = []
        reg_layers = []
        for _ in range(num_convs):
            cls_layers.extend(self._tower_block(in_channels, in_channels))
            reg_layers.extend(self._tower_block(in_channels, in_channels))

        self.cls_tower = nn.Sequential(*cls_layers)
        self.reg_tower = nn.Sequential(*reg_layers)
        self.cls_logits = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

        self._init_weights(prior_probability)

    @staticmethod
    def _tower_block(in_channels: int, out_channels: int) -> list[nn.Module]:
        return [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(inplace=True),
        ]

    def _init_weights(self, prior_probability: float) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.constant_(module.bias, 0)

        bias = -math.log((1 - prior_probability) / prior_probability)
        nn.init.constant_(self.cls_logits.bias, bias)

    def forward(
        self,
        features: OrderedDict[str, torch.Tensor],
    ) -> dict[str, OrderedDict[str, torch.Tensor]]:
        cls_logits = OrderedDict()
        bbox_regression = OrderedDict()
        centerness = OrderedDict()

        for level, feature in features.items():
            cls_feature = self.cls_tower(feature)
            reg_feature = self.reg_tower(feature)
            cls_logits[level] = self.cls_logits(cls_feature)
            bbox_regression[level] = self.bbox_pred(reg_feature)
            centerness[level] = self.centerness(reg_feature)

        return {
            "cls_logits": cls_logits,
            "bbox_regression": bbox_regression,
            "centerness": centerness,
        }

