"""Singleton LRU cache for denoise models and tiled inference dispatch."""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from apps.denoise_demo.checkpoint_info import CacheInfo
from apps.denoise_demo.fast_infer import OVERLAP, TILE, infer_tiled_fast, prepare_model_for_fast_infer
from apps.denoise_demo.registry import (
    REPO_ROOT,
    Family,
    NormalizationSpec,
    parse_normalization,
    resolve_ckpt_path,
)

SCRIPTS_DIR: Path = REPO_ROOT / "scripts" / "infer"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from output_formats import OutputFormat, save_denoised  # noqa: E402
from swinir_gray_dn_infer import load_gray_dn_model  # noqa: E402
from tsfdan_infer_helpers import load_tsfdan_model  # noqa: E402

CACHE_CAP: Final[int] = 2
ENV_IDLE_SECONDS: Final[str] = "DENOISE_DEMO_IDLE_SECONDS"
DEFAULT_IDLE_SECONDS: Final[float] = 600.0
IDLE_CHECK_INTERVAL: Final[float] = 15.0
CacheKey = tuple[Family, str]
FloatHW = NDArray[np.float32]
UInt8HW = NDArray[np.uint8]


@dataclass
class CacheEntry:
    """One loaded network kept in the process-wide LRU cache."""

    model: torch.nn.Module
    device: torch.device
    load_seconds: float
    family: Family
    ckpt: Path
    last_used: float = 0.0


class ModelEngine:
    """Load by (family, resolved ckpt path); infer with demo batched tiled AMP."""

    def __init__(self, max_entries: int = CACHE_CAP) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self._max_entries: int = max_entries
        self._cache: OrderedDict[CacheKey, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._reaper: threading.Thread | None = None
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def cache_info(self, family: Family, ckpt_resolved: Path) -> CacheInfo:
        """Report whether this key is currently in the GPU/CPU cache."""
        resolved = resolve_ckpt_path(str(ckpt_resolved), REPO_ROOT)
        key: CacheKey = (family, str(resolved))
        entry = self._cache.get(key)
        if entry is None:
            return CacheInfo(loaded=False)
        return CacheInfo(
            loaded=True,
            device=str(entry.device),
            last_load_seconds=entry.load_seconds,
        )

    def _evict_oldest(self) -> None:
        _key, entry = self._cache.popitem(last=False)
        entry.model.to("cpu")
        del entry.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def release_idle(self, max_idle_seconds: float) -> int:
        """Unload every model idle longer than ``max_idle_seconds``; return how many were freed."""
        now = time.monotonic()
        freed = 0
        with self._lock:
            for key in [k for k, e in self._cache.items() if now - e.last_used > max_idle_seconds]:
                entry = self._cache.pop(key)
                entry.model.to("cpu")
                del entry.model
                freed += 1
            if freed and torch.cuda.is_available():
                torch.cuda.empty_cache()
        return freed

    def start_idle_reaper(self, max_idle_seconds: float, interval: float = IDLE_CHECK_INTERVAL) -> None:
        """Start a daemon thread that frees idle models. Independent of open browser tabs."""
        if max_idle_seconds <= 0:
            return  # explicitly disabled
        if self._reaper is not None and self._reaper.is_alive():
            return

        def _loop() -> None:
            while True:
                time.sleep(interval)
                if self.release_idle(max_idle_seconds):
                    print(f"[denoise_demo] released models idle > {max_idle_seconds:.0f}s", flush=True)

        self._reaper = threading.Thread(target=_loop, name="denoise-idle-reaper", daemon=True)
        self._reaper.start()

    def get_or_load(self, family: Family, ckpt: Path) -> CacheEntry:
        """Load once per cache key. Never substitutes another weight on failure."""
        with self._lock:
            return self._get_or_load_locked(family, ckpt)

    def _get_or_load_locked(self, family: Family, ckpt: Path) -> CacheEntry:
        resolved = resolve_ckpt_path(str(ckpt), REPO_ROOT)
        key: CacheKey = (family, str(resolved))
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            cached.last_used = time.monotonic()
            return cached
        if not resolved.is_file():
            raise FileNotFoundError(f"checkpoint not found: {resolved}")
        started = time.perf_counter()
        if family == "swinir_m":
            model: torch.nn.Module = load_gray_dn_model(resolved, self.device)
        elif family == "tsfdan":
            model = load_tsfdan_model(resolved, self.device)
        else:
            raise ValueError(f"unsupported family: {family!r}")
        prepare_model_for_fast_infer(model)
        elapsed = time.perf_counter() - started
        entry = CacheEntry(
            model=model,
            device=self.device,
            load_seconds=elapsed,
            family=family,
            ckpt=resolved,
            last_used=time.monotonic(),
        )
        while len(self._cache) >= self._max_entries:
            self._evict_oldest()
        self._cache[key] = entry
        return entry

    def infer(self, family: Family, ckpt: Path, image_hw: FloatHW) -> FloatHW:
        """Run tiled denoise. Loads the model if this key is not cached."""
        if image_hw.ndim != 2:
            raise ValueError(f"expected HW float image, got shape {image_hw.shape}")
        if family not in ("swinir_m", "tsfdan"):
            raise ValueError(f"unsupported family: {family!r}")
        # Hold the lock for the whole inference so the idle reaper never unloads a model in use.
        with self._lock:
            entry = self._get_or_load_locked(family, ckpt)
            try:
                output = infer_tiled_fast(
                    entry.model,
                    image_hw,
                    entry.device,
                    tile=TILE,
                    overlap=OVERLAP,
                )
            finally:
                entry.last_used = time.monotonic()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # drop activation buffers; weights stay cached
        return np.clip(output.astype(np.float32), 0.0, 1.0)


_ENGINE: ModelEngine | None = None


def idle_seconds_from_env() -> float:
    """Idle timeout from ``DENOISE_DEMO_IDLE_SECONDS`` (default 600; 0 disables)."""
    raw = os.environ.get(ENV_IDLE_SECONDS)
    if raw is None or raw.strip() == "":
        return DEFAULT_IDLE_SECONDS
    return float(raw)


def get_engine() -> ModelEngine:
    """Process-wide singleton engine (LRU cap 2)."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = ModelEngine()
    return _ENGINE


def to_display_u8(image_01: FloatHW) -> UInt8HW:
    """Map [0, 1] to 8-bit PNG pixels for Gradio."""
    return (np.clip(image_01, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def load_input_image(path: Path, normalization: NormalizationSpec) -> FloatHW:
    """Load PNG/JPEG/TIFF to [0, 1] HW. RGB uses channel 0 (matches TIFF infer)."""
    array = np.array(Image.open(path))
    if array.ndim == 3:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"expected 2D image after channel-0, got shape {array.shape}: {path}")
    if array.dtype == np.uint16:
        scale = normalization.raw_max - normalization.raw_min
        if scale <= 0:
            raise ValueError("raw_max must be greater than raw_min")
        return np.clip(
            (array.astype(np.float32) - np.float32(normalization.raw_min)) / np.float32(scale),
            0.0,
            1.0,
        )
    if array.dtype == np.uint8:
        return array.astype(np.float32) / 255.0
    raise TypeError(f"unsupported image dtype {array.dtype} for {path}; expected uint8 PNG/JPEG or uint16 TIFF")


def load_input_image_from_preset_json(path: Path, normalization_json: str | None) -> FloatHW:
    """Load an image using optional preset `normalization` JSON path or inline object."""
    return load_input_image(path, parse_normalization(normalization_json))


def write_download_file(
    output_01: FloatHW,
    source_path: Path,
    fmt: OutputFormat,
    normalization: NormalizationSpec,
) -> Path:
    """Write the denoised image to a fresh temp dir as ``<stem>_denoised.<ext>`` for download."""
    out_dir = Path(tempfile.mkdtemp(prefix="denoise_demo_"))
    return save_denoised(
        output_01,
        out_dir / f"{source_path.stem}_denoised",
        fmt,
        raw_min=normalization.raw_min,
        raw_max=normalization.raw_max,
    )
