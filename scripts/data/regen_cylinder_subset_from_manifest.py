#!/usr/bin/env python3
"""Rebuild a small cylinder LQ/GT subset from archive manifest + raw XY slices.

Uses the same geometry recorded in the archived cylinder experiment
(rotation + bilinear resize to 1530). Does not recreate the full dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/mnt/home/huang/cst_ai/archive/cylinderblock_2673001900/manifest.csv"),
    )
    parser.add_argument(
        "--normalization",
        type=Path,
        default=Path("/mnt/home/huang/cst_ai/archive/cylinderblock_2673001900/normalization.json"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--n-per-batch", type=int, default=10, help="Train samples per batch (260520/260527)")
    parser.add_argument("--data-prefix", type=str, default="/mnt/data", help="Host mount replacing /data")
    return parser.parse_args()


def host_path(container_path: str, data_prefix: str) -> Path:
    if container_path.startswith("/data/"):
        return Path(data_prefix) / container_path[len("/data/") :]
    return Path(container_path)


def transform_lq(arr: np.ndarray, *, rotation_deg_cw: float, out_size: int) -> np.ndarray:
    """Apply CW rotation (nearest) then bilinear resize to out_size x out_size."""
    img = Image.fromarray(arr)
    # PIL rotate is CCW-positive; CW theta => angle -theta.
    rotated = img.rotate(-float(rotation_deg_cw), resample=Image.Resampling.NEAREST, expand=False, fillcolor=0)
    resized = rotated.resize((out_size, out_size), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized)


def pick_rows(rows: list[dict[str, str]], n_per_batch: int) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for batch in ("260520", "260527"):
        batch_rows = [r for r in rows if r["split"] == "train" and r["batch"] == batch]
        if len(batch_rows) < n_per_batch:
            raise SystemExit(f"Not enough train rows for batch {batch}: {len(batch_rows)}")
        # Spread across the train list for diversity (current / slice).
        step = len(batch_rows) / n_per_batch
        idxs = [min(len(batch_rows) - 1, int(i * step)) for i in range(n_per_batch)]
        # Deduplicate while preserving order.
        seen: set[int] = set()
        for i in idxs:
            if i in seen:
                continue
            seen.add(i)
            selected.append(batch_rows[i])
        while len([r for r in selected if r["batch"] == batch]) < n_per_batch:
            for i, row in enumerate(batch_rows):
                if i not in seen:
                    seen.add(i)
                    selected.append(row)
                    break
    return selected


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8")))
    norm = json.loads(args.normalization.read_text(encoding="utf-8"))
    geom = norm["geometry"]["lq_by_batch"]
    out_size = int(norm["geometry"]["gt"]["width"])

    selected = pick_rows(rows, args.n_per_batch)
    lq_dir = args.output_root / "LQ"
    gt_dir = args.output_root / "GT"
    lq_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    meta_lines: list[str] = []
    for row in selected:
        pair_id = row["pair_id"]
        batch = row["batch"]
        rot = float(geom[batch]["rotation_deg_cw"])
        lq_src = host_path(row["source_lq_path"], args.data_prefix)
        gt_src = host_path(row["source_gt_path"], args.data_prefix)
        if not lq_src.is_file() or not gt_src.is_file():
            raise SystemExit(f"Missing source for {pair_id}: LQ={lq_src} GT={gt_src}")

        lq = np.array(Image.open(lq_src))
        gt = np.array(Image.open(gt_src))
        if lq.ndim == 3:
            lq = lq[..., 0]
        if gt.ndim == 3:
            gt = gt[..., 0]
        lq_out = transform_lq(lq, rotation_deg_cw=rot, out_size=out_size)
        if gt.shape != (out_size, out_size):
            raise SystemExit(f"Unexpected GT shape {gt.shape} for {pair_id}, expected {(out_size, out_size)}")

        name = f"{pair_id}.tif"
        Image.fromarray(lq_out.astype(np.uint16)).save(lq_dir / name)
        Image.fromarray(gt.astype(np.uint16)).save(gt_dir / name)
        meta_lines.append(name)
        print(f"wrote {name} rot_cw={rot} from {lq_src.name}", flush=True)

    (args.output_root / "pick20.txt").write_text("".join(f"{n}\n" for n in meta_lines), encoding="utf-8")
    # Reuse archived normalization for inference.
    (args.output_root / "normalization.json").write_text(
        json.dumps(
            {
                "method": "fixed_range",
                "raw_min": float(norm.get("raw_min", 0.0)),
                "raw_max": float(norm.get("raw_max", 65535.0)),
                "output_range": [0.0, 1.0],
                "apply_to_lq": True,
                "apply_to_gt": True,
                "source_dtype": "uint16",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Done: {len(meta_lines)} pairs -> {args.output_root}")


if __name__ == "__main__":
    main()
