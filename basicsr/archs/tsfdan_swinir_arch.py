"""TSFDAN dual-branch SwinIR (Wasserdawn/TSFDAN new_model.SwinIR).

Pretrain: SwinTransformer_288800iter.ckpt — not compatible with official SwinIR-M.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from basicsr.archs.swinir_arch import SwinIR
from basicsr.utils.registry import ARCH_REGISTRY


class TSFDANSwinIRBranch(SwinIR):
    """Feature-space SwinIR branch used inside TSFDAN (skip branch conv_first/last)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TSFDAN networks.py denoising path: features in, features out.
        return self.conv_after_body(self.forward_features(x)) + x


@ARCH_REGISTRY.register()
class TSFDANSwinIR(nn.Module):
    """Dual-branch LDCT denoiser matching TSFDAN ``new_model.SwinIR`` + ckpt keys."""

    def __init__(
        self,
        in_chans: int = 1,
        embed_dim: int = 60,
        img_size: int = 64,
        window_size: int = 8,
        img_range: float = 1.0,
        depths: Sequence[int] = (4, 4, 4),
        num_heads: Sequence[int] = (6, 6, 6),
        mlp_ratio: float = 2.0,
        **kwargs: object,
    ) -> None:
        super().__init__()
        _ = kwargs  # absorb unused YAML keys (e.g. upscale) without failing build_network
        self.window_size = window_size
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        self.conv_last = nn.Conv2d(embed_dim * 2, in_chans, 3, 1, 1)

        branch_kw = {
            "upscale": 1,
            "img_size": img_size,
            "in_chans": 1,
            "window_size": window_size,
            "img_range": img_range,
            "depths": list(depths),
            "embed_dim": embed_dim,
            "num_heads": list(num_heads),
            "mlp_ratio": mlp_ratio,
            "upsampler": "",
        }
        self.swinIR_s = TSFDANSwinIRBranch(patch_size=1, **branch_kw)
        self.swinIR_l = TSFDANSwinIRBranch(patch_size=4, **branch_kw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_first = self.conv_first(x)
        x_s = self.swinIR_s(x_first)
        x_l = self.swinIR_l(x_first)
        return x + self.conv_last(torch.cat((x_s, x_l), dim=1))
