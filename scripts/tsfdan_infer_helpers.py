"""Shared TSFDAN dual-branch inference helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from basicsr.archs.tsfdan_swinir_arch import TSFDANSwinIR

DEFAULT_PRETRAIN = Path("experiments/pretrained_models/TSFDAN/SwinTransformer_288800iter.ckpt")


def build_tsfdan_model() -> TSFDANSwinIR:
    return TSFDANSwinIR(
        in_chans=1,
        embed_dim=60,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=[4, 4, 4],
        num_heads=[6, 6, 6],
        mlp_ratio=2.0,
    )


def load_tsfdan_model(model_path: Path, device: torch.device) -> TSFDANSwinIR:
    model = build_tsfdan_model()
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint format: {model_path}")
    if "params" in checkpoint and isinstance(checkpoint["params"], dict):
        state = checkpoint["params"]
    elif "params_ema" in checkpoint and isinstance(checkpoint["params_ema"], dict):
        state = checkpoint["params_ema"]
    else:
        # Raw OrderedDict from TSFDAN release (tensor values at top level).
        state = checkpoint
    model.load_state_dict(state, strict=True)
    model.eval()
    return model.to(device)


def infer_tiled(
    model: torch.nn.Module,
    img_hw: np.ndarray,
    device: torch.device,
    *,
    tile: int = 256,
    overlap: int = 32,
    window_size: int = 8,
) -> np.ndarray:
    img = torch.from_numpy(img_hw[None, None, ...]).float().to(device)
    with torch.no_grad():
        if tile <= 0:
            _, _, h, w = img.size()
            mod_h = (window_size - h % window_size) % window_size
            mod_w = (window_size - w % window_size) % window_size
            padded = F.pad(img, (0, mod_w, 0, mod_h), "reflect")
            out = model(padded)
            return out[:, :, :h, :w].squeeze().float().cpu().clamp_(0, 1).numpy()

        _, _, h, w = img.size()
        tile_sz = min(tile, h, w)
        tile_sz = tile_sz - (tile_sz % window_size) or window_size
        stride = max(tile_sz - overlap, window_size)
        output = torch.zeros((1, 1, h, w), device=device)
        weight = torch.zeros_like(output)
        h_idxs = list(range(0, max(h - tile_sz, 0) + 1, stride))
        w_idxs = list(range(0, max(w - tile_sz, 0) + 1, stride))
        if not h_idxs or h_idxs[-1] != h - tile_sz:
            h_idxs.append(max(h - tile_sz, 0))
        if not w_idxs or w_idxs[-1] != w - tile_sz:
            w_idxs.append(max(w - tile_sz, 0))
        for hi in h_idxs:
            for wi in w_idxs:
                patch = img[:, :, hi : hi + tile_sz, wi : wi + tile_sz]
                ph, pw = patch.shape[-2:]
                mod_h = (window_size - ph % window_size) % window_size
                mod_w = (window_size - pw % window_size) % window_size
                patch = F.pad(patch, (0, mod_w, 0, mod_h), "reflect")
                out = model(patch)[:, :, :ph, :pw]
                output[:, :, hi : hi + ph, wi : wi + pw].add_(out)
                weight[:, :, hi : hi + ph, wi : wi + pw].add_(1.0)
        return (output / weight).squeeze().float().cpu().clamp_(0, 1).numpy()
