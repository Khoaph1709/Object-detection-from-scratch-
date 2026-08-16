from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FPN(nn.Module):
    """Feature Pyramid Network producing P3-P7 from C3-C5."""

    def __init__(self, in_channels: list[int], out_channels: int = 256) -> None:
        super().__init__()
        if len(in_channels) != 3:
            raise ValueError("FPN expects exactly three input channel counts for C3, C4, C5.")

        self.lateral_c3 = nn.Conv2d(in_channels[0], out_channels, kernel_size=1)
        self.lateral_c4 = nn.Conv2d(in_channels[1], out_channels, kernel_size=1)
        self.lateral_c5 = nn.Conv2d(in_channels[2], out_channels, kernel_size=1)

        self.output_p3 = ConvBlock(out_channels, out_channels, kernel_size=3)
        self.output_p4 = ConvBlock(out_channels, out_channels, kernel_size=3)
        self.output_p5 = ConvBlock(out_channels, out_channels, kernel_size=3)
        self.output_p6 = ConvBlock(out_channels, out_channels, kernel_size=3)
        self.output_p7 = ConvBlock(out_channels, out_channels, kernel_size=3)

        self.p6_downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.p7_downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, features: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
        c3, c4, c5 = features["c3"], features["c4"], features["c5"]

        p5 = self.lateral_c5(c5)
        p4 = self.lateral_c4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lateral_c3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")

        p3 = self.output_p3(p3)
        p4 = self.output_p4(p4)
        p5 = self.output_p5(p5)
        p6 = self.output_p6(self.p6_downsample(p5))
        p7 = self.output_p7(self.p7_downsample(p6))

        return OrderedDict(
            {
                "p3": p3,
                "p4": p4,
                "p5": p5,
                "p6": p6,
                "p7": p7,
            }
        )


def print_fpn_feature_shapes(image_size: int = 640, batch_size: int = 2) -> None:
    from .backbone import ConvNeXtTinyBackbone

    backbone = ConvNeXtTinyBackbone(pretrained=False)
    fpn = FPN(in_channels=backbone.out_channels, out_channels=256)
    backbone.eval()
    fpn.eval()

    x = torch.randn(batch_size, 3, image_size, image_size)
    with torch.no_grad():
        c_features = backbone(x)
        p_features = fpn(c_features)

    print("Backbone features:")
    for name, tensor in c_features.items():
        print(f"{name.upper()}: {tuple(tensor.shape)}")

    print("FPN features:")
    for name, tensor in p_features.items():
        print(f"{name.upper()}: {tuple(tensor.shape)}")

