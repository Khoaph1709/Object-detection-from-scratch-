from __future__ import annotations

from collections import OrderedDict

from torch import nn

from .backbone import ConvNeXtBackbone
from .bifpn import BiFPN
from .fpn import FPN
from .head import FCOSHead


class FCOSDetector(nn.Module):
    """Custom anchor-free FCOS detector used by the final HEM pipeline.

    The final submission uses P2-P7 features, a ConvNeXt backbone, BiFPN and a
    custom FCOS head. The legacy P1 and targeted high-resolution experiment
    switches are rejected so a stale experiment cannot be trained accidentally.
    """

    def __init__(
        self,
        num_classes: int = 5,
        backbone_name: str = "convnext_tiny",
        pretrained_backbone: bool = True,
        fpn_channels: int = 256,
        fpn_type: str = "fpn",
        bifpn_layers: int = 1,
        use_p1: bool = False,
        targeted_highres: bool = False,
    ) -> None:
        super().__init__()
        if use_p1 or targeted_highres:
            raise ValueError(
                "This cleaned submission is HEM-only; P1 and targeted_highres are disabled."
            )
        if fpn_type not in {"fpn", "bifpn"}:
            raise ValueError(f"Unsupported pyramid type: {fpn_type}")

        self.backbone_name = backbone_name
        self.fpn_type = fpn_type
        self.bifpn_layers = bifpn_layers
        self.use_p1 = False
        self.targeted_highres = False
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

    def forward(self, images):
        features = self.fpn(self.backbone(images))
        return {"features": features, **self.head(features)}


def build_detector(
    num_classes: int = 5,
    pretrained_backbone: bool = True,
    backbone_name: str = "convnext_tiny",
    fpn_type: str = "fpn",
    bifpn_layers: int = 1,
    use_p1: bool = False,
    targeted_highres: bool = False,
) -> FCOSDetector:
    return FCOSDetector(
        num_classes=num_classes,
        backbone_name=backbone_name,
        pretrained_backbone=pretrained_backbone,
        fpn_type=fpn_type,
        bifpn_layers=bifpn_layers,
        use_p1=use_p1,
        targeted_highres=targeted_highres,
    )
