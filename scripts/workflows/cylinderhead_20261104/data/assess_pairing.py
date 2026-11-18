"""Read ZIPs without extracting; summarize acquisition and sample alignment."""

import io
import itertools
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path("/dataset/archives")
OUT = Path("/analysis/pairing_assessment.json")
SLICES = [128, 384, 768, 1152, 1400]


def alignment(reference, moving):
    a = cv2.GaussianBlur(reference, (0, 0), 3)[128:-128, 128:-128].copy()
    b = cv2.GaussianBlur(moving, (0, 0), 3)[128:-128, 128:-128].copy()
    window = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(a, b, window)
    return {"dx_pixels": shift[0], "dy_pixels": shift[1], "response": response}


archives = {p.stem: zipfile.ZipFile(p) for p in sorted(ROOT.glob("*.zip"))}
try:
    scans = []
    names = {}
    for name, archive in archives.items():
        tiffs = [i for i in archive.infolist() if i.filename.lower().endswith((".tif", ".tiff"))]
        names[name] = {i.filename for i in tiffs}
        meta = {}
        for line in archive.read("data.tsv").decode("utf-8-sig").splitlines():
            row = line.split("\t")
            if len(row) >= 2:
                meta[row[0]] = row[1]
        scans.append({"scan": name, "tiff_count": len(tiffs), "metadata": meta})
    report = {
        "all_slice_names_identical": all(s == next(iter(names.values())) for s in names.values()),
        "scans": scans,
        "sample_slices": SLICES,
        "samples": [],
        "method": (
            "OpenCV phase correlation after Gaussian sigma=3 and 128px border removal; shifts are "
            "moving relative to reference; diagnostic only, no registration applied"
        ),
    }
    high = [n for n in archives if "200uA" in n]
    for index in SLICES:
        arrays = {}
        formats = {}
        for name, archive in archives.items():
            with Image.open(io.BytesIO(archive.read(f"XY/data_{index:04d}.tif"))) as image:
                arr = np.asarray(image)
                formats[name] = {
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "min": int(arr.min()),
                    "max": int(arr.max()),
                }
                arrays[name] = arr.astype(np.float32)
        comparisons = []
        for a, b in itertools.combinations(high, 2):
            diff = arrays[a] - arrays[b]
            comparisons.append(
                {
                    "reference": a,
                    "moving": b,
                    "mae_raw_units": float(np.abs(diff).mean()),
                    "difference_std_raw_units": float(diff.std()),
                    "identical_pixels_fraction": float(np.mean(diff == 0)),
                    **alignment(arrays[a], arrays[b]),
                }
            )
        low = []
        for name in arrays:
            if name in high:
                continue
            low.append({"reference": high[0], "moving": name, **alignment(arrays[high[0]], arrays[name])})
        report["samples"].append(
            {
                "slice": index,
                "formats": formats,
                "high_current_comparisons": comparisons,
                "lower_current_comparisons": low,
            }
        )
        print(
            json.dumps(
                {
                    "slice": index,
                    "high_shifts": [
                        {k: row[k] for k in ("moving", "dx_pixels", "dy_pixels", "response")} for row in comparisons
                    ],
                    "largest_low_shift_pixels": max(max(abs(r["dx_pixels"]), abs(r["dy_pixels"])) for r in low),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("report:", OUT, flush=True)
finally:
    for archive in archives.values():
        archive.close()
