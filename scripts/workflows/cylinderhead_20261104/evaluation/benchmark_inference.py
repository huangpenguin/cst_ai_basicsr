"""Measure serving implementation on validation data, without touching the test set."""

import gc
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/workspace")
from apps.denoise_demo.fast_infer import infer_tiled_fast, prepare_model_for_fast_infer
from scripts.infer.swinir_gray_dn_infer import load_gray_dn_model

OUT = Path("/benchmark_out")
selection = json.loads((OUT / "demo_model_status.json").read_text())
checkpoint = Path(selection["entry"]["ckpt"])
with Image.open("/benchmark_input/input_validation.tif") as im:
    full = np.asarray(im).astype(np.float32) / np.float32(65535)
device = torch.device("cuda:0")
torch.cuda.synchronize()
started = time.perf_counter()
model = load_gray_dn_model(checkpoint, device)
prepare_model_for_fast_infer(model)
torch.cuda.synchronize()
load_seconds = time.perf_counter() - started
results = []
for size, batch in [(128, 1), (512, 1), (1518, 1), (1518, 2), (1518, 8)]:
    image = full.copy() if size == 1518 else full[400 : 400 + size, 400 : 400 + size].copy()
    try:
        output = infer_tiled_fast(model, image, device, tile=256, overlap=32, tile_batch=batch)
        assert np.isfinite(output).all()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(3):
            started = time.perf_counter()
            output = infer_tiled_fast(model, image, device, tile=256, overlap=32, tile_batch=batch)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - started)
            assert output.shape == image.shape and np.isfinite(output).all()
        row = {
            "height": int(image.shape[0]),
            "width": int(image.shape[1]),
            "tile": 256,
            "tile_batch": batch,
            "warm_seconds": times,
            "median_warm_seconds": statistics.median(times),
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            "finite_output": True,
        }
    except torch.cuda.OutOfMemoryError:
        row = {"height": int(image.shape[0]), "width": int(image.shape[1]), "tile_batch": batch, "error": "CUDA OOM"}
        gc.collect()
        torch.cuda.empty_cache()
    results.append(row)
    print(json.dumps(row), flush=True)
report = {
    "gpu": torch.cuda.get_device_name(),
    "torch": torch.__version__,
    "checkpoint": str(checkpoint),
    "checkpoint_load_seconds": load_seconds,
    "model_parameters": sum(p.numel() for p in model.parameters()),
    "input_source": "validation slice 755, 20uA; not test",
    "method": (
        "Same fast_infer implementation as Demo; one warm-up then 3 timed repeats; includes "
        "CPU-to-GPU/GPU-to-CPU, excludes HTTP/upload/file decode and model load"
    ),
    "memory_note": (
        "PyTorch process peak allocations/reservations; CUDA context and other serving processes add overhead"
    ),
    "results": results,
}
(OUT / "inference_benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
