"""Make reproducible same-window image panels without changing source TIFFs."""

import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path("/analysis/comparison")
OUT.mkdir(exist_ok=True)
ROOT = Path("/dataset/archives")
FONT = ImageFont.load_default(size=22)
SMALL = ImageFont.load_default(size=18)
SCANS = [zipfile.ZipFile(ROOT / f"Cylinderhead_300kV200uA_data{i}.zip") for i in (1, 2, 3)]
TIMES = ["07:51:25", "07:58:42", "08:02:18"]


def read(index):
    return [np.asarray(Image.open(io.BytesIO(z.read(f"XY/data_{index:04d}.tif")))).astype(np.float32) for z in SCANS]


def gray(a, size, window=(5000, 40000)):
    v = np.clip((a - window[0]) / (window[1] - window[0]), 0, 1)
    im = Image.fromarray(np.rint(v * 255).astype(np.uint8)).convert("RGB")
    return im.resize(size, Image.Resampling.NEAREST)


def text(canvas, xy, value, font=FONT):
    ImageDraw.Draw(canvas).text(xy, value, font=font, fill="#17202a")


try:
    indices = [384, 768, 1152]
    panel = 420
    canvas = Image.new("RGB", (1356, 1510), "#f6f8fa")
    text(canvas, (28, 15), "Three separate 200 uA scans | 300 kV | same display window")
    text(canvas, (28, 48), "All panels: raw values 5000-40000; identical scaling; 1518 x 1518 uint16.", SMALL)
    for col, t in enumerate(TIMES):
        text(canvas, (28 + col * 446, 84), f"data{col + 1} | acquired {t}")
    loaded = {}
    for row, index in enumerate(indices):
        arrays = read(index)
        loaded[index] = arrays
        top = 145 + row * 450
        text(canvas, (28, top - 31), f"XY/data_{index:04d}.tif")
        for col, a in enumerate(arrays):
            canvas.paste(gray(a, (panel, panel)), (28 + col * 446, top))
    canvas.save(OUT / "three_scans_overview.png")

    # Select a textured, non-saturated ROI for a reproducible detailed comparison.
    arrays = loaded[768]
    mean = np.mean(arrays, axis=0)
    smooth = cv2.GaussianBlur(mean, (0, 0), 2)
    roi_size = 192
    best = (-1, None)
    for y in range(220, 1100, 96):
        for x in range(220, 1100, 96):
            region = smooth[y : y + roi_size, x : x + roi_size]
            if np.mean((region > 10000) & (region < 38000)) < 0.7:
                continue
            score = float(region.std())
            if score > best[0]:
                best = (score, (x, y))
    x, y = best[1] or (664, 664)
    crops = [a[y : y + roi_size, x : x + roi_size] for a in arrays]
    detail = Image.new("RGB", (1356, 1120), "#f6f8fa")
    text(detail, (28, 15), f"Slice 0768 | same ROI x={x}:{x + roi_size}, y={y}:{y + roi_size}")
    text(detail, (28, 48), "Top: identical window 5000-40000. Bottom: signed raw differences, +/-3000.", SMALL)
    for col, a in enumerate(crops):
        text(detail, (28 + col * 446, 87), f"data{col + 1} | no registration applied")
        detail.paste(gray(a, (420, 420)), (28 + col * 446, 120))
    stats = []
    for col, (i, j) in enumerate([(0, 1), (0, 2), (1, 2)]):
        diff = crops[i] - crops[j]
        t = np.clip(diff / 3000, -1, 1)
        rgb = np.ones((*t.shape, 3), dtype=np.float32)
        rgb[:, :, 0] = 1 + np.minimum(t, 0)
        rgb[:, :, 1] = 1 - np.abs(t)
        rgb[:, :, 2] = 1 - np.maximum(t, 0)
        im = Image.fromarray(np.rint(rgb * 255).astype(np.uint8)).resize((420, 420), Image.Resampling.NEAREST)
        text(detail, (28 + col * 446, 575), f"data{i + 1} - data{j + 1}")
        detail.paste(im, (28 + col * 446, 610))
        mae = float(np.abs(diff).mean())
        text(detail, (28 + col * 446, 1040), f"ROI mean abs difference: {mae:.1f}", SMALL)
        stats.append({"pair": [i + 1, j + 1], "roi_mae_raw": mae, "roi_diff_std_raw": float(diff.std())})
    text(detail, (28, 1080), "Blue: negative | white: zero | red: positive. Difference colors are amplified.", SMALL)
    detail.save(OUT / "three_scans_detail.png")
    (OUT / "display_settings.json").write_text(
        json.dumps(
            {
                "slice": 768,
                "roi_xywh": [x, y, roi_size, roi_size],
                "display_window": [5000, 40000],
                "difference_limits": [-3000, 3000],
                "roi_statistics": stats,
            },
            indent=2,
        )
        + "\n"
    )
    print("saved", str(OUT), json.dumps(stats), flush=True)
finally:
    for z in SCANS:
        z.close()
