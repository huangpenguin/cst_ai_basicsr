"""Sample same-index images across currents; visualize original pixel positions."""

import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

cv2.setNumThreads(2)
ROOT = Path("/input/archives")
OUT = Path("/analysis")
CURRENTS = [10, 20, 40, 60, 80, 100, 120, 140, 160, 180]
INDICES = sorted({0, 64, *range(128, 1401, 64), 384, 768, 1152, 1400, 1472, 1523})
FONT = ImageFont.load_default(size=20)
SMALL = ImageFont.load_default(size=15)


def text(canvas, xy, message, font=FONT):
    ImageDraw.Draw(canvas).text(xy, message, font=font, fill="#182431")


def load(z, index):
    with Image.open(io.BytesIO(z.read(f"XY/data_{index:04d}.tif"))) as im:
        array = np.asarray(im)
        assert array.shape == (1518, 1518) and array.dtype == np.uint16
        return array


def shifted(reference, moving):
    a = cv2.GaussianBlur(reference.astype(np.float32), (0, 0), 3)[128:-128, 128:-128].copy()
    b = cv2.GaussianBlur(moving.astype(np.float32), (0, 0), 3)[128:-128, 128:-128].copy()
    window = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    delta, response = cv2.phaseCorrelate(a.copy(), b.copy(), window)
    a0 = a - a.mean()
    b0 = b - b.mean()
    denominator = np.linalg.norm(a0) * np.linalg.norm(b0)
    ncc = float(np.sum(a0 * b0) / denominator) if denominator else None
    material_fraction = float(np.mean(a > 10000))
    magnitude = float(np.hypot(delta[0], delta[1]))
    reliable = bool(response >= 0.15 and material_fraction >= 0.02 and np.isfinite(magnitude))
    return {
        "dx_pixels": float(delta[0]),
        "dy_pixels": float(delta[1]),
        "translation_magnitude_pixels": magnitude,
        "phase_response": float(response),
        "blurred_ncc_before": ncc,
        "reference_material_fraction": material_fraction,
        "reliable": reliable,
    }


def render(index, arrays, reference, rows, detail=False):
    images = [arrays[c] for c in CURRENTS] + [reference]
    labels = [f"{c} uA" for c in CURRENTS] + ["200 uA | mean of 3 (GT)"]
    canvas = Image.new("RGB", (1640, 1190), "#f6f8fa")
    text(canvas, (20, 15), f"Cylinderhead | 300 kV | XY/data_{index:04d}.tif | original pixel positions")
    text(
        canvas, (20, 45), "Shared display window: raw 5000-40000. No alignment or per-image contrast adjustment.", SMALL
    )
    if detail:
        text(canvas, (20, 68), "Same ROI in every panel: x=508:700, y=412:604 (192 x 192).", SMALL)
    else:
        text(canvas, (20, 68), "Full frame: 1518 x 1518 uint16. dx/dy are estimated shifts relative to mean GT.", SMALL)
    for i, (array, label) in enumerate(zip(images, labels, strict=True)):
        x = 20 + (i % 5) * 324
        y = 110 + (i // 5) * 350
        text(canvas, (x, y), label)
        source = array[412:604, 508:700] if detail else array
        image = Image.fromarray(
            np.rint(np.clip((source.astype(np.float32) - 5000) / 35000, 0, 1) * 255).astype(np.uint8)
        )
        canvas.paste(image.convert("RGB").resize((304, 304), Image.Resampling.NEAREST), (x, y + 24))
        if i < len(rows):
            row = rows[i]
            label = (
                f"dx {row['dx_pixels']:+.2f}, dy {row['dy_pixels']:+.2f} px"
                if row["reliable"]
                else "Weak structure: shift unreliable"
            )
        else:
            label = "round((data1 + data2 + data3) / 3)"
        text(canvas, (x, y + 329), label, SMALL)
    path = OUT / f"slice_{index:04d}_{'detail' if detail else 'all_currents'}.png"
    canvas.save(path)


def main():
    low = {c: zipfile.ZipFile(ROOT / f"Cylinderhead_300kV{c}uA_data.zip") for c in CURRENTS}
    high = [zipfile.ZipFile(ROOT / f"Cylinderhead_300kV200uA_data{i}.zip") for i in (1, 2, 3)]
    try:
        results = []
        for index in INDICES:
            refs = [load(z, index) for z in high]
            mean = ((refs[0].astype(np.uint32) + refs[1] + refs[2] + 1) // 3).astype(np.uint16)
            arrays = {c: load(z, index) for c, z in low.items()}
            rows = [{"slice": index, "current_ua": c, **shifted(mean, arrays[c])} for c in CURRENTS]
            results.extend(rows)
            if index in (384, 768, 1152):
                render(index, arrays, mean, rows)
            if index == 768:
                render(index, arrays, mean, rows, detail=True)
            print("Compared slice", index, "reliable", sum(r["reliable"] for r in rows), "of 10", flush=True)
        summaries = []
        for current in CURRENTS:
            rows = [r for r in results if r["current_ua"] == current and r["reliable"]]
            values = np.array([r["translation_magnitude_pixels"] for r in rows])
            summaries.append(
                {
                    "current_ua": current,
                    "reliable_samples": len(rows),
                    "median_translation_pixels": float(np.median(values)) if len(values) else None,
                    "max_translation_pixels": float(values.max()) if len(values) else None,
                    "worst_sample": max(rows, key=lambda r: r["translation_magnitude_pixels"]) if rows else None,
                }
            )
        report = {
            "source": str(ROOT),
            "sampled_slices": INDICES,
            "comparisons": results,
            "summary_by_current": summaries,
            "method": (
                "Translation-only phase correlation on Gaussian sigma=3 images, 128px border "
                "excluded; same-index uint16 rounded mean of 3 is reference; original images are not "
                "warped"
            ),
            "confidence_rule": "response >= 0.15 and reference fraction above raw 10000 >= 0.02",
            "limitations": (
                "Sampled 2D translation diagnostic; not a full-stack, depth, rotation or nonrigid "
                "registration validation"
            ),
            "display_window": [5000, 40000],
        }
        (OUT / "alignment_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(summaries, indent=2), flush=True)
    finally:
        for z in [*low.values(), *high]:
            z.close()


if __name__ == "__main__":
    main()
