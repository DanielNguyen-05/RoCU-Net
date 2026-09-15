"""Feature upsampling controls for the shared RoCU refinement scaffold.

CARAFE is a pure-PyTorch implementation of the reassembly equation and the
CARAFEPack defaults: https://mmcv.readthedocs.io/en/latest/_modules/mmcv/ops/carafe.html
DySample-LP follows https://github.com/tiny-smart/dysample (MIT, Wenze Liu 2023).
See THIRD_PARTY_NOTICES.md. These operators do not constrain occupancy.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class BilinearUpsampling(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


class CARAFE(nn.Module):
    """2x CARAFE, one reassembly group, zero-padded 5x5 support by default.

    Contract coarse patches with four child kernels before pixel shuffle to
    avoid explicitly expanding the C*k*k neighborhood tensor to full resolution.
    This is an equation-level reference, not the optimized MMCV CUDA kernel.
    """

    def __init__(self, channels: int, kernel_size: int = 5, compressed_channels: int = 64):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 != 1 or compressed_channels < 1:
            raise ValueError("CARAFE requires an odd positive kernel and positive compression width")
        self.kernel_size = int(kernel_size)
        self.channel_compressor = nn.Conv2d(channels, compressed_channels, 1)
        self.content_encoder = nn.Conv2d(compressed_channels, kernel_size**2 * 4, 3, padding=1)
        nn.init.xavier_uniform_(self.channel_compressor.weight)
        nn.init.zeros_(self.channel_compressor.bias)
        nn.init.normal_(self.content_encoder.weight, std=.001)
        nn.init.zeros_(self.content_encoder.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        k = self.kernel_size
        logits = self.content_encoder(self.channel_compressor(x))
        weights = logits.reshape(b, k * k, 4, h, w).softmax(dim=1)
        patches = F.unfold(x, kernel_size=k, padding=k // 2).reshape(b, c, k * k, h, w)
        children = torch.einsum("bckhw,bkshw->bcshw", patches, weights.to(x.dtype))
        return F.pixel_shuffle(children.reshape(b, c * 4, h, w), 2)


class DySampleLP(nn.Module):
    """2x DySample, LP ordering, four groups, static offset scope (not DySample+).

    Pixel-center coordinates, border padding, offset scale .25 and small-normal
    offset initialization match the authors' LP implementation. Sampling uses
    FP32 under mixed precision to avoid rounding the coordinate grid.
    """

    def __init__(self, channels: int, groups: int = 4):
        super().__init__()
        if groups < 1 or channels % groups:
            raise ValueError("DySample channels must be divisible by its positive group count")
        self.groups = groups
        self.offset = nn.Conv2d(channels, 2 * groups * 4, 1)
        nn.init.normal_(self.offset.weight, std=.001)
        nn.init.zeros_(self.offset.bias)
        positions = torch.tensor([-.25, .25])
        yy, xx = torch.meshgrid(positions, positions, indexing="ij")
        initial = torch.stack((xx, yy)).reshape(2, 1, 4).repeat(1, groups, 1)
        self.register_buffer("init_pos", initial.reshape(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        predicted = self.offset(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            offset = (predicted.float() * .25 + self.init_pos.float()).reshape(b, 2, -1, h, w)
            yy, xx = torch.meshgrid(torch.arange(h, device=x.device, dtype=torch.float32) + .5,
                                    torch.arange(w, device=x.device, dtype=torch.float32) + .5,
                                    indexing="ij")
            centers = torch.stack((xx, yy)).reshape(1, 2, 1, h, w)
            normalizer = x.new_tensor([w, h], dtype=torch.float32).reshape(1, 2, 1, 1, 1)
            coords = 2 * (centers + offset) / normalizer - 1
            coords = F.pixel_shuffle(coords.reshape(b, -1, h, w), 2)
            grid = coords.reshape(b, 2, self.groups, 2 * h, 2 * w).permute(0, 2, 3, 4, 1)
            result = F.grid_sample(x.float().reshape(b * self.groups, c // self.groups, h, w),
                                   grid.reshape(b * self.groups, 2 * h, 2 * w, 2),
                                   mode="bilinear", padding_mode="border", align_corners=False)
        return result.reshape(b, c, 2 * h, 2 * w).to(x.dtype)


def build_carrier_upsampler(name: str, channels: int) -> nn.Module:
    if name == "bilinear":
        return BilinearUpsampling()
    if name == "carafe":
        return CARAFE(channels)
    if name == "dysample":
        return DySampleLP(channels)
    raise ValueError(f"Unsupported carrier upsampling operator: {name}")
