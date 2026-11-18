"""Run inside the deployed container to verify weights and GPU inference."""

import sys
from pathlib import Path

sys.path.insert(0, "/workspace")
import numpy as np
import torch
from PIL import Image

from apps.denoise_demo.app import denoise_and_debug
from apps.denoise_demo.registry import load_registry, resolve_ckpt_path

assert torch.cuda.is_available(), "CUDA unavailable"
print("GPU:", torch.cuda.get_device_name(), "torch:", torch.__version__, flush=True)
registry = load_registry()
for preset in registry.presets:
    path = resolve_ckpt_path(preset.ckpt)
    print(preset.id, "exists:", path.is_file(), flush=True)
    assert path.is_file(), path

image = np.random.default_rng(0).integers(0, 65536, (64, 64), dtype=np.uint16)
input_path = Path("/tmp/basicsr-smoke.tif")
Image.fromarray(image).save(input_path)
for preset_id in (registry.default_id, "tsfdan_preft", "tsfdan_lora_mix"):
    preset = registry.get(preset_id)
    original, denoised, debug, download = denoise_and_debug(registry, preset_id, "", preset.family, input_path)
    assert download is not None, debug
    assert original is not None and denoised is not None, debug
    assert original.shape == denoised.shape == image.shape
    assert Path(download).is_file()
    with Image.open(download) as result:
        pixels = np.asarray(result)
        assert pixels.shape == image.shape and np.isfinite(pixels).all()
    print(preset_id, "GPU inference + TIFF export: OK", flush=True)
