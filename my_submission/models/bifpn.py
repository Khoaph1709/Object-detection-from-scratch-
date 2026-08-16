from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn


class WeightedFusion(nn.Module):
    """Fast normalized positive weighted feature fusion used by BiFPN."""

    def __init__(self, num_inputs: int) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        if len(inputs) != self.weights.numel():
            raise ValueError(f"Expected {self.weights.numel()} inputs, got {len(inputs)}")
        weights = F.relu(self.weights)
        return sum(weight * value for weight, value in zip(weights, inputs)) / (weights.sum() + 1e-4)


class SeparableConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm = nn.GroupNorm(32, channels)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.pointwise(self.depthwise(x))))


class BiFPNLayer(nn.Module):
    """One P2-P7 bidirectional weighted feature-fusion layer."""

    def __init__(self, channels: int = 256) -> None:
        super().__init__()
        self.top_fusions = nn.ModuleList([WeightedFusion(2) for _ in range(5)])
        self.bottom_fusions = nn.ModuleList([WeightedFusion(3) for _ in range(4)])
        self.top_convs = nn.ModuleList([SeparableConvBlock(channels) for _ in range(5)])
        self.bottom_convs = nn.ModuleList([SeparableConvBlock(channels) for _ in range(5)])

    @staticmethod
    def _upsample(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.interpolate(source, size=target.shape[-2:], mode="nearest")

    @staticmethod
    def _downsample(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(source, output_size=target.shape[-2:])

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(features) != 6:
            raise ValueError("BiFPN expects P2-P7 features.")

        p2, p3, p4, p5, p6, p7 = features
        top = [None] * 6
        top[5] = self.top_convs[4](p7)
        top[4] = self.top_convs[3](self.top_fusions[4]([p6, self._upsample(top[5], p6)]))
        top[3] = self.top_convs[2](self.top_fusions[3]([p5, self._upsample(top[4], p5)]))
        top[2] = self.top_convs[1](self.top_fusions[2]([p4, self._upsample(top[3], p4)]))
        top[1] = self.top_convs[0](self.top_fusions[1]([p3, self._upsample(top[2], p3)]))
        top[0] = self.bottom_convs[0](self.top_fusions[0]([p2, self._upsample(top[1], p2)]))

        bottom = [None] * 6
        bottom[0] = top[0]
        bottom[1] = self.bottom_convs[1](self.bottom_fusions[0]([p3, top[1], self._downsample(bottom[0], p3)]))
        bottom[2] = self.bottom_convs[2](self.bottom_fusions[1]([p4, top[2], self._downsample(bottom[1], p4)]))
        bottom[3] = self.bottom_convs[3](self.bottom_fusions[2]([p5, top[3], self._downsample(bottom[2], p5)]))
        bottom[4] = self.bottom_convs[4](self.bottom_fusions[3]([p6, top[4], self._downsample(bottom[3], p6)]))
        bottom[5] = top[5]
        return bottom


class BiFPN(nn.Module):
    """Repeated weighted bidirectional pyramid over P2-P7."""

    def __init__(self, in_channels: list[int], out_channels: int = 256, num_layers: int = 1) -> None:
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError("BiFPN expects C2-C5 input channel counts.")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.lateral = nn.ModuleList(nn.Conv2d(channels, out_channels, 1) for channels in in_channels)
        self.initial_p6 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        self.initial_p7 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        self.layers = nn.ModuleList(BiFPNLayer(out_channels) for _ in range(num_layers))

    def forward(self, features: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
        c2, c3, c4, c5 = features["c2"], features["c3"], features["c4"], features["c5"]
        p2, p3, p4, p5 = [layer(value) for layer, value in zip(self.lateral, [c2, c3, c4, c5])]
        p6 = self.initial_p6(p5)
        p7 = self.initial_p7(p6)
        pyramid = [p2, p3, p4, p5, p6, p7]
        for layer in self.layers:
            pyramid = layer(pyramid)
        return OrderedDict((f"p{index + 2}", value) for index, value in enumerate(pyramid))
