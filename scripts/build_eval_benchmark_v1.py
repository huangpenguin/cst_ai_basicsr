#!/usr/bin/env python3
"""Build fixed eval_benchmark_v1 manifests and hardlinked LQ/GT subsets.

Five columns (separate, never merged):
  - battery_260520 / battery_260527 (all available LQ currents vs max-current GT)
  - cylinder_260520_val20 (from cylinder_260520_gt90ua val.txt)
  - cylinder_260527_holdout20 (from cylinder_260527_eval_gt60ua)
  - phantom_aligned_drive_val20 (from aligned_drive val.txt)
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/home/huang/cst_ai/outputs/eval_benchmark_v1"),
    )
    p.add_argument("--n-cylinder", type=int, default=20)
    p.add_argument("--n-phantom", type=int, default=20)
    return p.parse_args()


def hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def pick_spread(names: list[str], n: int) -> list[str]:
    if not names:
        return []
    if len(names) <= n:
        return list(names)
    step = len(names) / n
    idxs = sorted({min(len(names) - 1, int(i * step)) for i in range(n)})
    while len(idxs) < n:
        for i in range(len(names)):
            if i not in idxs:
                idxs.append(i)
            if len(idxs) >= n:
                break
    return [names[i] for i in sorted(idxs)[:n]]


def write_list(path: Path, names: list[str]) -> None:
    path.write_text("".join(f"{n}\n" for n in names), encoding="utf-8")


def link_pairs(
    names: list[str],
    *,
    lq_src: Path,
    gt_src: Path,
    lq_dst: Path,
    gt_dst: Path,
) -> None:
    for name in names:
        hardlink_or_copy(lq_src / name, lq_dst / name)
        hardlink_or_copy(gt_src / name, gt_dst / name)


_UA_RE = re.compile(r"(\d+)uA", re.IGNORECASE)


def battery_ua(path: Path) -> int:
    match = _UA_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse µA from {path.name}")
    return int(match.group(1))


def build_battery_column(
    tiff_dir: Path,
    *,
    column: str,
    output_root: Path,
) -> list[str]:
    import numpy as np
    from PIL import Image

    files = sorted(p for p in tiff_dir.iterdir() if p.suffix.lower() in {".tif", ".tiff"})
    if not files:
        raise FileNotFoundError(f"No TIFF in {tiff_dir}")
    by_ua = {battery_ua(p): p for p in files}
    gt_ua = max(by_ua)
    gt_src = by_ua[gt_ua]
    gt_arr = np.asarray(Image.open(gt_src))
    if gt_arr.ndim == 3:
        gt_arr = gt_arr[..., 0]
    lq_items = sorted(((ua, p) for ua, p in by_ua.items() if ua != gt_ua), key=lambda x: x[0])
    col_root = output_root / "pairs" / column
    lq_dst = col_root / "LQ"
    gt_dst = col_root / "GT"
    lq_dst.mkdir(parents=True, exist_ok=True)
    gt_dst.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for ua, lq_path in lq_items:
        lq_arr = np.asarray(Image.open(lq_path))
        if lq_arr.ndim == 3:
            lq_arr = lq_arr[..., 0]
        h = min(lq_arr.shape[0], gt_arr.shape[0])
        w = min(lq_arr.shape[1], gt_arr.shape[1])
        name = f"battery_{column}_{ua}uA_vs_{gt_ua}uA.tif"
        # Crop to common size (battery TIFF heights can differ by 1px).
        Image.fromarray(lq_arr[:h, :w].astype(np.uint16)).save(lq_dst / name)
        Image.fromarray(gt_arr[:h, :w].astype(np.uint16)).save(gt_dst / name)
        names.append(name)
    write_list(output_root / "manifests" / f"{column}.txt", names)
    (col_root / "normalization.json").write_text(
        json.dumps(
            {
                "method": "fixed_range",
                "raw_min": 0.0,
                "raw_max": 65535.0,
                "output_range": [0.0, 1.0],
                "apply_to_lq": True,
                "apply_to_gt": True,
                "source_dtype": "uint16",
                "gt_ua": gt_ua,
                "note": "Battery flat TIFF pairs; highest µA as GT; cropped to common HxW.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return names


def build_meta_column(
    *,
    column: str,
    meta_list: Path,
    lq_src: Path,
    gt_src: Path,
    norm_src: Path,
    output_root: Path,
    n: int,
) -> list[str]:
    names = [line.strip() for line in meta_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    picked = pick_spread(names, n)
    col_root = output_root / "pairs" / column
    link_pairs(picked, lq_src=lq_src, gt_src=gt_src, lq_dst=col_root / "LQ", gt_dst=col_root / "GT")
    write_list(output_root / "manifests" / f"{column}.txt", picked)
    shutil.copy2(norm_src, col_root / "normalization.json")
    return picked


def main() -> None:
    args = parse_args()
    root = args.output_root
    if root.exists():
        shutil.rmtree(root)
    (root / "manifests").mkdir(parents=True)
    (root / "pairs").mkdir(parents=True)
    (root / "panels").mkdir(parents=True)
    (root / "metrics").mkdir(parents=True)

    summary: dict[str, object] = {}

    b520 = build_battery_column(
        Path("/mnt/data/cst_ai/data_0607/260520_1-battery_data/260520_battery image tiff data"),
        column="battery_260520",
        output_root=root,
    )
    summary["battery_260520"] = {"count": len(b520), "files": b520}

    b527 = build_battery_column(
        Path("/mnt/data/cst_ai/data_0607/260527_3-battery_data/260527_battery image tiff data"),
        column="battery_260527",
        output_root=root,
    )
    summary["battery_260527"] = {"count": len(b527), "files": b527}

    c520 = build_meta_column(
        column="cylinder_260520_val20",
        meta_list=Path("/mnt/home/huang/cst_ai/datasets/cylinder_260520_gt90ua/val.txt"),
        lq_src=Path("/mnt/data/cst_ai/datasets/cylinder_260520_gt90ua/LQ"),
        gt_src=Path("/mnt/data/cst_ai/datasets/cylinder_260520_gt90ua/GT"),
        norm_src=Path("/mnt/home/huang/cst_ai/datasets/cylinder_260520_gt90ua/normalization.json"),
        output_root=root,
        n=args.n_cylinder,
    )
    summary["cylinder_260520_val20"] = {"count": len(c520), "files": c520}

    # Prefer val.txt; if shorter than n, fall back to full sorted LQ names.
    meta527_val = Path("/mnt/home/huang/cst_ai/datasets/cylinder_260527_eval_gt60ua/val.txt")
    meta527_all = Path("/mnt/data/cst_ai/datasets/cylinder_260527_eval_gt60ua/LQ")
    if meta527_val.is_file():
        pool = [line.strip() for line in meta527_val.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        pool = sorted(p.name for p in meta527_all.glob("*.tif"))
    if len(pool) < args.n_cylinder:
        pool = sorted(p.name for p in meta527_all.glob("*.tif"))
    c527 = pick_spread(pool, args.n_cylinder)
    col = "cylinder_260527_holdout20"
    link_pairs(
        c527,
        lq_src=Path("/mnt/data/cst_ai/datasets/cylinder_260527_eval_gt60ua/LQ"),
        gt_src=Path("/mnt/data/cst_ai/datasets/cylinder_260527_eval_gt60ua/GT"),
        lq_dst=root / "pairs" / col / "LQ",
        gt_dst=root / "pairs" / col / "GT",
    )
    write_list(root / "manifests" / f"{col}.txt", c527)
    shutil.copy2(
        Path("/mnt/home/huang/cst_ai/datasets/cylinder_260527_eval_gt60ua/normalization.json"),
        root / "pairs" / col / "normalization.json",
    )
    summary[col] = {"count": len(c527), "files": c527}

    ph = build_meta_column(
        column="phantom_aligned_drive_val20",
        meta_list=Path("/mnt/home/huang/cst_ai/datasets/aligned_drive/val.txt"),
        lq_src=Path("/mnt/data/cst_ai/datasets/aligned_drive/LQ"),
        gt_src=Path("/mnt/data/cst_ai/datasets/aligned_drive/GT"),
        norm_src=Path("/mnt/home/huang/cst_ai/datasets/aligned_drive/normalization.json"),
        output_root=root,
        n=args.n_phantom,
    )
    summary["phantom_aligned_drive_val20"] = {"count": len(ph), "files": ph}

    (root / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v["count"] for k, v in summary.items()}, indent=2))
    print(f"Wrote {root}")


if __name__ == "__main__":
    main()
