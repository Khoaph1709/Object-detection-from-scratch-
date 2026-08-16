from __future__ import annotations

from collections import OrderedDict

import torch


def generate_locations(
    features: OrderedDict[str, torch.Tensor],
    strides: dict[str, int],
) -> OrderedDict[str, torch.Tensor]:
    """Generate center locations in padded image coordinates for each FPN level."""
    locations = OrderedDict()
    for level, feature in features.items():
        _, _, height, width = feature.shape
        stride = strides[level]
        device = feature.device
        shifts_x = (torch.arange(width, dtype=torch.float32, device=device) + 0.5) * stride
        shifts_y = (torch.arange(height, dtype=torch.float32, device=device) + 0.5) * stride
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        locations[level] = torch.stack((shift_x.reshape(-1), shift_y.reshape(-1)), dim=1)
    return locations


def flatten_locations(locations: OrderedDict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat(list(locations.values()), dim=0)


def flatten_level_values(
    values: OrderedDict[str, torch.Tensor],
    channels_last: bool = True,
) -> torch.Tensor:
    flattened = []
    for tensor in values.values():
        if channels_last:
            tensor = tensor.permute(0, 2, 3, 1).reshape(tensor.shape[0], -1, tensor.shape[1])
        else:
            tensor = tensor.reshape(tensor.shape[0], tensor.shape[1], -1)
        flattened.append(tensor)
    return torch.cat(flattened, dim=1)

