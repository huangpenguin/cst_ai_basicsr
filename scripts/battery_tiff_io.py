"""TIFF I/O and preprocessing for 1_battery SEM images."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def read_battery_tiff(
    path: str | Path,
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> np.ndarray:
    """Read uint16 TIFF and percentile-normalize to float32 in [0, 1].

    Returns a single-channel array with shape (H, W).
    """
    img = np.array(Image.open(path))
    if img.ndim == 3:
        img = img[..., 0]
    if img.ndim != 2:
        raise ValueError(f"Expected 2D grayscale TIFF, got shape {img.shape}")

    lo, hi = np.percentile(img, [p_low, p_high])
    if hi <= lo:
        hi = float(lo) + 1.0
    normalized = np.clip((img.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    return normalized


def to_model_tensor(img_hw: np.ndarray):
    """Convert (H, W) float32 image to NCHW torch tensor for gray_dn."""
    import torch

    chw = np.expand_dims(img_hw, axis=0)
    return torch.from_numpy(chw).float().unsqueeze(0)


def save_gray_png(img_hw: np.ndarray, path: str | Path) -> None:
    """Save float32 [0, 1] grayscale image as 8-bit PNG."""
    out = (np.clip(img_hw, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    Image.fromarray(out, mode="L").save(path)
