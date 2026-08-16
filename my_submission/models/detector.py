from __future__ import annotations

import torch
from torch import nn

from .backbone import ConvNeXtTinyBackbone
from .fpn import FPN
from .head import FCOSHead


class FCOSDetector(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        backbone_name: str = "convnext_tiny",
        pretrained_backbone: bool = True,
        fpn_channels: int = 256,
    ) -> None:
        super().__init__()
        if backbone_name != "convnext_tiny":
            raise ValueError("Only convnext_tiny is implemented in this checkpoint.")

        self.backbone = ConvNeXtTinyBackbone(pretrained=pretrained_backbone)
        self.fpn = FPN(in_channels=self.backbone.out_channels, out_channels=fpn_channels)
        self.head = FCOSHead(in_channels=fpn_channels, num_classes=num_classes)
        self.strides = {"p3": 8, "p4": 16, "p5": 32, "p6": 64, "p7": 128}

    def forward(self, images: torch.Tensor) -> dict:
        c_features = self.backbone(images)
        p_features = self.fpn(c_features)
        head_outputs = self.head(p_features)
        return {
            "features": p_features,
            **head_outputs,
        }


def build_detector(
    num_classes: int = 5,
    pretrained_backbone: bool = True,
) -> FCOSDetector:
    return FCOSDetector(num_classes=num_classes, pretrained_backbone=pretrained_backbone)

