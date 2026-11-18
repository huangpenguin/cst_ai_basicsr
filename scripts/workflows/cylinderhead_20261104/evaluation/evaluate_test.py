"""Observe the final checkpoint on a fixed reserved test list, after training only."""

import json
import sys
from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

sys.path.insert(0, "/workspace")
from apps.denoise_demo.fast_infer import infer_tiled_fast, prepare_model_for_fast_infer
from scripts.infer.swinir_gray_dn_infer import load_gray_dn_model

data = Path("/data")
run = Path("/training_run")
name = "train_SwinIR_cylinderhead_mean3_raw_P128_bs4x4_i30k"
checkpoint = run / "artifacts" / name / "models/net_g_latest.pth"
model = load_gray_dn_model(checkpoint, torch.device("cuda:0"))
prepare_model_for_fast_infer(model)
out = run / "test_preview"
out.mkdir(exist_ok=True)
records = []
font = ImageFont.load_default(size=20)
for filename in (data / "test_preview.txt").read_text().splitlines():
    with Image.open(data / "LQ" / filename) as im:
        lq = np.asarray(im).astype(np.float32) / np.float32(65535)
    with Image.open(data / "GT" / filename) as im:
        gt = np.asarray(im).astype(np.float32) / np.float32(65535)
    prediction = infer_tiled_fast(model, lq, torch.device("cuda:0"), tile_batch=2)
    assert np.isfinite(prediction).all()
    row = {
        "filename": filename,
        "input_psnr": float(peak_signal_noise_ratio(gt, lq, data_range=1)),
        "output_psnr": float(peak_signal_noise_ratio(gt, prediction, data_range=1)),
        "input_ssim": float(cast(float, structural_similarity(gt, lq, data_range=1))),
        "output_ssim": float(cast(float, structural_similarity(gt, prediction, data_range=1))),
    }
    records.append(row)
    Image.fromarray(np.rint(np.clip(prediction, 0, 1) * 65535).astype(np.uint16)).save(
        out / (Path(filename).stem + "_denoised.tif")
    )
    panel = Image.new("RGB", (1536, 560), "#f6f8fa")
    draw = ImageDraw.Draw(panel)
    for i, (label, array) in enumerate([("LQ", lq), ("Final model", prediction), ("Mean-3 GT", gt)]):
        draw.text((i * 512 + 12, 10), label, font=font, fill="#182431")
        image = Image.fromarray(np.rint(np.clip((array * 65535 - 5000) / 35000, 0, 1) * 255).astype(np.uint8))
        panel.paste(image.convert("RGB").resize((512, 512)), (i * 512, 40))
    panel.save(out / (Path(filename).stem + "_comparison.png"))
    print("TEST", json.dumps(row), flush=True)
summary = {
    "checkpoint": str(checkpoint),
    "selection": "Fixed before training; not used for model selection",
    "images": len(records),
    "records": records,
    "mean_input_psnr": float(np.mean([r["input_psnr"] for r in records])),
    "mean_output_psnr": float(np.mean([r["output_psnr"] for r in records])),
}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
