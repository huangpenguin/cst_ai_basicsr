"""Selectable output encodings for denoised [0, 1] float images (CLI and demo app share this)."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

FloatHW = NDArray[np.float32]


class OutputFormat(StrEnum):
    """How a denoised image is written to disk."""

    PNG8 = "png8"  # 8-bit PNG, for quick viewing
    TIFF16 = "tiff16"  # uint16 TIFF, de-normalized back to the input raw range
    TIFF32 = "tiff32"  # float32 TIFF, normalized [0, 1] values (lossless model output)

    @property
    def suffix(self) -> str:
        return ".png" if self is OutputFormat.PNG8 else ".tif"


def encode_denoised(image_01: FloatHW, fmt: OutputFormat, *, raw_min: float, raw_max: float) -> Image.Image:
    """Convert a [0, 1] HW float image to a PIL image in the requested encoding."""
    if image_01.ndim != 2:
        raise ValueError(f"expected HW image, got shape {image_01.shape}")
    clipped = np.clip(image_01, 0.0, 1.0)
    if fmt is OutputFormat.PNG8:
        return Image.fromarray((clipped * 255.0).round().astype(np.uint8))
    if fmt is OutputFormat.TIFF16:
        if raw_max <= raw_min:
            raise ValueError("raw_max must be greater than raw_min")
        scaled = clipped.astype(np.float64) * (raw_max - raw_min) + raw_min
        return Image.fromarray(np.clip(np.rint(scaled), 0, 65535).astype(np.uint16))
    if fmt is OutputFormat.TIFF32:
        return Image.fromarray(clipped.astype(np.float32), mode="F")
    raise ValueError(f"unsupported output format: {fmt!r}")


def save_denoised(
    image_01: FloatHW,
    path_without_suffix: Path,
    fmt: OutputFormat,
    *,
    raw_min: float = 0.0,
    raw_max: float = 65535.0,
) -> Path:
    """Write the image; the file suffix is chosen from ``fmt``. Returns the written path."""
    out_path = path_without_suffix.with_name(path_without_suffix.name + fmt.suffix)
    encode_denoised(image_01, fmt, raw_min=raw_min, raw_max=raw_max).save(out_path)
    return out_path
