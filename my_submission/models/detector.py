import torch
from torch import nn

from .backbone import ConvNeXtBackbone
from .bifpn import BiFPN
from .fpn import FPN
from .head import FCOSHead


class FCOSDetector(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        backbone_name: str = "convnext_tiny",
        pretrained_backbone: bool = True,
        fpn_channels: int = 256,
        fpn_type: str = "fpn",
        bifpn_layers: int = 1,
    ) -> None:
        super().__init__()
        if fpn_type not in {"fpn", "bifpn"}:
            raise ValueError(f"Unsupported pyramid type: {fpn_type}")

        self.backbone_name = backbone_name
        self.fpn_type = fpn_type
        self.bifpn_layers = bifpn_layers
        self.backbone = ConvNeXtBackbone(model_name=backbone_name, pretrained=pretrained_backbone)
        if fpn_type == "bifpn":
            self.fpn = BiFPN(
                in_channels=self.backbone.out_channels,
                out_channels=fpn_channels,
                num_layers=bifpn_layers,
            )
        else:
            self.fpn = FPN(in_channels=self.backbone.out_channels, out_channels=fpn_channels)
        self.head = FCOSHead(in_channels=fpn_channels, num_classes=num_classes)
        self.strides = {"p2": 4, "p3": 8, "p4": 16, "p5": 32, "p6": 64, "p7": 128}

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
    backbone_name: str = "convnext_tiny",
    fpn_type: str = "fpn",
    bifpn_layers: int = 1,
) -> FCOSDetector:
    return FCOSDetector(
        num_classes=num_classes,
        backbone_name=backbone_name,
        pretrained_backbone=pretrained_backbone,
        fpn_type=fpn_type,
        bifpn_layers=bifpn_layers,
    )
