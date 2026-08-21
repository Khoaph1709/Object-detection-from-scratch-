from __future__ import annotations

import torch.nn.functional as F
from torch import nn


class P1Refinement(nn.Module):
    """Learned stride-2 refinement synthesized from the existing P2 feature."""

    def __init__(self, channels: int = 256) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, p2):
        refined = F.interpolate(p2, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.block(refined)
