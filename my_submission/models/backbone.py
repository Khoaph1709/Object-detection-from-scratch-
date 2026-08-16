from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn


class ConvNeXtTinyBackbone(nn.Module):
    """ConvNeXt-Tiny feature extractor returning C3, C4, and C5."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("ConvNeXtTinyBackbone requires timm. Install requirements.txt first.") from exc

        self.body = timm.create_model(
            "convnext_tiny",
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3),
        )
        self.out_channels = list(self.body.feature_info.channels())
        self.out_strides = list(self.body.feature_info.reduction())

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        features = self.body(x)
        return OrderedDict(
            {
                "c3": features[0],
                "c4": features[1],
                "c5": features[2],
            }
        )


def print_convnext_tiny_feature_shapes(
    image_size: int = 640,
    batch_size: int = 2,
    pretrained: bool = False,
) -> None:
    model = ConvNeXtTinyBackbone(pretrained=pretrained)
    model.eval()
    x = torch.randn(batch_size, 3, image_size, image_size)
    with torch.no_grad():
        features = model(x)
    for name, tensor in features.items():
        print(f"{name.upper()}: {tuple(tensor.shape)}")

