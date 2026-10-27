"""Batched tiled inference for the Gradio demo (AMP on CUDA). Eval scripts stay serial fp32."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from typing import Final

import numpy as np
import torch
from numpy.typing import NDArray
from torch.nn import functional as F

from basicsr.archs.swinir_arch import RSTB

FloatHW = NDArray[np.float32]

TILE: Final[int] = 256
OVERLAP: Final[int] = 32
WINDOW_SIZE: Final[int] = 8
TILE_BATCH: Final[int] = 8


def _tile_starts(length: int, tile_sz: int, stride: int) -> list[int]:
    idxs: list[int] = list(range(0, max(length - tile_sz, 0) + 1, stride))
    last = max(length - tile_sz, 0)
    if not idxs or idxs[-1] != last:
        idxs.append(last)
    return idxs


def _forward_batch(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    return model(batch)


def prepare_model_for_fast_infer(model: torch.nn.Module) -> None:
    """Keep RSTB (attention + residual conv) in fp32 under CUDA autocast.

    SwinIR-M activations overflow float16 around mid-depth RSTB convs, which
    yields all-NaN outputs. Autocast still applies to stem/head convs.
    """
    for module in model.modules():
        if not isinstance(module, RSTB):
            continue
        if getattr(module, "_demo_amp_fp32_rstb", False):
            continue
        orig_rstb = module.forward

        def rstb_fp32(
            x: torch.Tensor,
            x_size: tuple[int, int],
            *,
            _orig: Callable[..., torch.Tensor] = orig_rstb,
        ) -> torch.Tensor:
            if x.device.type != "cuda":
                return _orig(x, x_size)
            with torch.autocast(device_type="cuda", enabled=False):
                return _orig(x.float(), x_size)

        module.forward = rstb_fp32  # type: ignore[method-assign]
        module._demo_amp_fp32_rstb = True


def infer_tiled_fast(
    model: torch.nn.Module,
    img_hw: FloatHW,
    device: torch.device,
    *,
    tile: int = TILE,
    overlap: int = OVERLAP,
    window_size: int = WINDOW_SIZE,
    tile_batch: int = TILE_BATCH,
) -> FloatHW:
    """Tiled denoise with optional CUDA AMP and batched tile forwards.

    Geometry matches `scripts/infer/infer_swinir_graydn_tiff.py::infer_tiled` and
    `scripts/infer/tsfdan_infer_helpers.py::infer_tiled` (reflect pad, overlap-average).
    Output is float32 HW in [0, 1]. CUDA uses inference_mode + fp16 autocast;
    CPU uses inference_mode only.
    """
    if img_hw.ndim != 2:
        raise ValueError(f"expected HW float image, got shape {img_hw.shape}")
    if tile_batch < 1:
        raise ValueError("tile_batch must be >= 1")

    img = torch.from_numpy(np.ascontiguousarray(img_hw)[None, None, ...]).to(device=device, dtype=torch.float32)
    use_amp = device.type == "cuda"
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()

    with torch.inference_mode(), autocast:
        _, _, height, width = img.size()
        if tile <= 0:
            mod_h = (window_size - height % window_size) % window_size
            mod_w = (window_size - width % window_size) % window_size
            padded = F.pad(img, (0, mod_w, 0, mod_h), "reflect")
            out = _forward_batch(model, padded)
            result = out[:, :, :height, :width].squeeze().float().cpu().clamp_(0, 1).numpy()
            return np.asarray(result, dtype=np.float32)

        tile_sz = min(tile, height, width)
        tile_sz = tile_sz - (tile_sz % window_size) or window_size
        stride = max(tile_sz - overlap, window_size)
        output = torch.zeros((1, 1, height, width), device=device, dtype=torch.float32)
        weight = torch.zeros_like(output)
        h_idxs = _tile_starts(height, tile_sz, stride)
        w_idxs = _tile_starts(width, tile_sz, stride)

        patches: list[torch.Tensor] = []
        coords: list[tuple[int, int, int, int]] = []
        for hi in h_idxs:
            for wi in w_idxs:
                patch = img[:, :, hi : hi + tile_sz, wi : wi + tile_sz]
                patch_h, patch_w = patch.shape[-2:]
                mod_h = (window_size - patch_h % window_size) % window_size
                mod_w = (window_size - patch_w % window_size) % window_size
                patch = F.pad(patch, (0, mod_w, 0, mod_h), "reflect")
                patches.append(patch)
                coords.append((hi, wi, patch_h, patch_w))

        groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, patch in enumerate(patches):
            groups[(int(patch.shape[-2]), int(patch.shape[-1]))].append(index)

        for indices in groups.values():
            for start in range(0, len(indices), tile_batch):
                chunk = indices[start : start + tile_batch]
                batch = torch.cat([patches[i] for i in chunk], dim=0)
                pred = _forward_batch(model, batch).float()
                for local, index in enumerate(chunk):
                    hi, wi, patch_h, patch_w = coords[index]
                    output[:, :, hi : hi + patch_h, wi : wi + patch_w].add_(
                        pred[local : local + 1, :, :patch_h, :patch_w]
                    )
                    weight[:, :, hi : hi + patch_h, wi : wi + patch_w].add_(1.0)

        merged = (output / weight).squeeze().float().cpu().clamp_(0, 1).numpy()
        return np.asarray(merged, dtype=np.float32)
