"""Profile mean-GT content without consuming any LQ information."""

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

root = Path("/mean/GT")
rows = []
for i, p in enumerate(sorted(root.glob("*.tif"))):
    with Image.open(p) as im:
        a = np.asarray(im)[::8, ::8].astype(np.float32)
    rows.append(
        {
            "slice": int(p.stem.split("_")[-1]),
            "material_fraction_gt10000": float(np.mean(a > 10000)),
            "material_fraction_gt15000": float(np.mean(a > 15000)),
            "p01": float(np.percentile(a, 1)),
            "p99": float(np.percentile(a, 99)),
            "saturated_fraction": float(np.mean(a >= 65535)),
        }
    )
    if i % 200 == 0:
        print("Profiled", i + 1, "GT slices", flush=True)
out = Path("/analysis/gt_content_profile.json")
out.write_text(json.dumps(rows, indent=2) + "\n")
with out.with_suffix(".csv").open("w") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
for threshold in (0.005, 0.01, 0.02):
    ids = [r["slice"] for r in rows if r["material_fraction_gt10000"] >= threshold]
    blocks = []
    for x in ids:
        if not blocks or x > blocks[-1][-1] + 1:
            blocks.append([x])
        else:
            blocks[-1].append(x)
    print("Threshold", threshold, "count", len(ids), "ranges", [(b[0], b[-1]) for b in blocks], flush=True)
