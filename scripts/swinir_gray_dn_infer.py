"""Shared SwinIR gray_dn inference helpers for smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from basicsr.archs.swinir_arch import SwinIR

GRAY_DN_MODELS: dict[int, str] = {
    15: "004_grayDN_DFWB_s128w8_SwinIR-M_noise15.pth",
    25: "004_grayDN_DFWB_s128w8_SwinIR-M_noise25.pth",
    50: "004_grayDN_DFWB_s128w8_SwinIR-M_noise50.pth",
}


def build_gray_dn_model() -> SwinIR:
    return SwinIR(
        upscale=1,
        in_chans=1,
        img_size=128,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="",
        resi_connection="1conv",
    )


def load_gray_dn_model(model_path: Path, device: torch.device) -> SwinIR:
    model = build_gray_dn_model()
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    key = "params_ema" if "params_ema" in checkpoint else "params"
    model.load_state_dict(checkpoint[key], strict=True)
    model.eval()
    return model.to(device)


def infer_gray(
    model: SwinIR,
    img_hw: np.ndarray,
    device: torch.device,
    to_model_tensor,
) -> np.ndarray:
    window_size = 8
    img = to_model_tensor(img_hw).to(device)

    with torch.no_grad():
        _, _, h, w = img.size()
        mod_pad_h = (window_size - h % window_size) % window_size
        mod_pad_w = (window_size - w % window_size) % window_size
        img = F.pad(img, (0, mod_pad_w, 0, mod_pad_h), "reflect")
        output = model(img)
        output = output[:, :, :h, :w]

    return output.squeeze().float().cpu().clamp_(0, 1).numpy()
