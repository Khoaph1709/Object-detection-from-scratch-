from collections import OrderedDict

import torch
from torch import nn


class ConvNeXtBackbone(nn.Module):
    """ConvNeXt feature extractor returning C2-C5 features."""

    def __init__(self, model_name: str = "convnext_tiny", pretrained: bool = True) -> None:
        super().__init__()
        if model_name not in {"convnext_tiny", "convnext_small"}:
            raise ValueError(f"Unsupported ConvNeXt backbone: {model_name}")
        try:
            import timm
        except ImportError as exc:
            raise ImportError("ConvNeXtBackbone requires timm. Install requirements.txt first.") from exc

        self.model_name = model_name
        self.body = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        self.out_channels = list(self.body.feature_info.channels())
        self.out_strides = list(self.body.feature_info.reduction())

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        features = self.body(x)
        return OrderedDict(
            {
                "c2": features[0],
                "c3": features[1],
                "c4": features[2],
                "c5": features[3],
            }
        )


class ConvNeXtTinyBackbone(ConvNeXtBackbone):
    """Backward-compatible ConvNeXt-Tiny wrapper."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__(model_name="convnext_tiny", pretrained=pretrained)


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
