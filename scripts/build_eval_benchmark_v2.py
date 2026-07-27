#!/usr/bin/env python3
"""Build fixed eval_benchmark_v2 (leakage-first; multi-µA where safe).

Keeps the same five columns as v1. Does not delete eval_benchmark_v1.

Sampling rules:
  - battery_*: same as v1 (all LQ currents vs max-current GT)
  - cylinder_260527_holdout20: stratified-by-µA from full holdout (never trained)
  - cylinder_260520_val20: leakage-safe pool = cylinder_260520_gt90ua/val.txt
    minus mix train originals (cyl__*). Multi-µA not forced if pool is single-µA.
  - phantom_aligned_drive_val20: from aligned_drive/val.txt minus mix train (ph__*)
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

_UA_RE = re.compile(r"(\d+)uA", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/home/huang/cst_ai/outputs/eval_benchmark_v2"),
    )
    p.add_argument("--n-cylinder", type=int, default=20)
    p.add_argument("--n-phantom", type=int, default=20)
    p.add_argument(
        "--mix-train",
        type=Path,
        default=Path("/mnt/home/huang/cst_ai/datasets/cylinder_phantom_mix/train.txt"),
    )
    p.add_argument(
        "--cyl90-train",
        type=Path,
        default=Path("/mnt/home/huang/cst_ai/datasets/cylinder_260520_gt90ua/train.txt"),
    )
    p.add_argument(
        "--ph-train",
        type=Path,
        default=Path("/mnt/home/huang/cst_ai/datasets/aligned_drive/train.txt"),
    )
    return p.parse_args()


def hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def write_list(path: Path, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{n}\n" for n in names), encoding="utf-8")


def read_list(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ua_of(name: str) -> str:
    match = _UA_RE.search(name)
    return match.group(1) if match else "unknown"


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


def pick_stratified_ua(names: list[str], n: int) -> list[str]:
    """Round-robin across µA buckets with spread within each bucket."""
    if not names:
        return []
    if len(names) <= n:
        return sorted(names)
    by_ua: dict[str, list[str]] = defaultdict(list)
    for name in sorted(names):
        by_ua[ua_of(name)].append(name)
    buckets = [by_ua[k] for k in sorted(by_ua)]
    picked: list[str] = []
    positions = [0] * len(buckets)
    # Pre-spread indices per bucket for even slice coverage
    spreads = [pick_spread(b, min(len(b), n)) for b in buckets]
    while len(picked) < n:
        progressed = False
        for i, spread in enumerate(spreads):
            if positions[i] < len(spread) and len(picked) < n:
                picked.append(spread[positions[i]])
                positions[i] += 1
                progressed = True
        if not progressed:
            break
    return sorted(picked)


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


def battery_ua(path: Path) -> int:
    match = _UA_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse µA from {path.name}")
    return int(match.group(1))


def build_battery_column(tiff_dir: Path, *, column: str, output_root: Path) -> list[str]:
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


def mix_train_originals(mix_train: Path) -> tuple[set[str], set[str]]:
    cyl: set[str] = set()
    ph: set[str] = set()
    for name in read_list(mix_train):
        if name.startswith("cyl__"):
            cyl.add(name.removeprefix("cyl__"))
        elif name.startswith("ph__"):
            ph.add(name.removeprefix("ph__"))
    return cyl, ph


def main() -> None:
    args = parse_args()
    root = args.output_root
    if root.exists():
        shutil.rmtree(root)
    (root / "manifests").mkdir(parents=True)
    (root / "pairs").mkdir(parents=True)
    (root / "panels").mkdir(parents=True)
    (root / "metrics").mkdir(parents=True)

    mix_cyl_train, mix_ph_train = mix_train_originals(args.mix_train)
    cyl90_train = set(read_list(args.cyl90_train))
    ph_train = set(read_list(args.ph_train))
    # Safe for all compared models: not in Cylinder90 train, phantom train, or mix train.
    forbidden_cyl = cyl90_train | mix_cyl_train
    forbidden_ph = ph_train | mix_ph_train

    summary: dict[str, object] = {}

    b520 = build_battery_column(
        Path("/mnt/data/cst_ai/data_0607/260520_1-battery_data/260520_battery image tiff data"),
        column="battery_260520",
        output_root=root,
    )
    ua_b520: dict[str, int] = defaultdict(int)
    for n in b520:
        ua_b520[ua_of(n)] += 1
    summary["battery_260520"] = {"count": len(b520), "ua_counts": dict(sorted(ua_b520.items())), "files": b520}

    b527 = build_battery_column(
        Path("/mnt/data/cst_ai/data_0607/260527_3-battery_data/260527_battery image tiff data"),
        column="battery_260527",
        output_root=root,
    )
    summary["battery_260527"] = {"count": len(b527), "files": b527}

    # cylinder 260520: start from official val, drop anything that appears in any train.
    cyl_val_pool = [
        n
        for n in read_list(Path("/mnt/home/huang/cst_ai/datasets/cylinder_260520_gt90ua/val.txt"))
        if n not in forbidden_cyl
    ]
    c520 = pick_stratified_ua(cyl_val_pool, args.n_cylinder)
    col520 = "cylinder_260520_val20"
    link_pairs(
        c520,
        lq_src=Path("/mnt/data/cst_ai/datasets/cylinder_260520_gt90ua/LQ"),
        gt_src=Path("/mnt/data/cst_ai/datasets/cylinder_260520_gt90ua/GT"),
        lq_dst=root / "pairs" / col520 / "LQ",
        gt_dst=root / "pairs" / col520 / "GT",
    )
    write_list(root / "manifests" / f"{col520}.txt", c520)
    shutil.copy2(
        Path("/mnt/home/huang/cst_ai/datasets/cylinder_260520_gt90ua/normalization.json"),
        root / "pairs" / col520 / "normalization.json",
    )
    ua520: dict[str, int] = defaultdict(int)
    for n in c520:
        ua520[ua_of(n)] += 1
    summary[col520] = {
        "count": len(c520),
        "pool_size": len(cyl_val_pool),
        "ua_counts": dict(sorted(ua520.items())),
        "note": "Leakage-safe pool from cyl90 val minus mix/cyl90 train; multi-µA not forced.",
        "files": c520,
    }

    # cylinder 260527 holdout: never trained — stratified multi-µA from full LQ.
    holdout_lq = Path("/mnt/data/cst_ai/datasets/cylinder_260527_eval_gt60ua/LQ")
    holdout_pool = sorted(p.name for p in holdout_lq.glob("*.tif"))
    c527 = pick_stratified_ua(holdout_pool, args.n_cylinder)
    col527 = "cylinder_260527_holdout20"
    link_pairs(
        c527,
        lq_src=holdout_lq,
        gt_src=Path("/mnt/data/cst_ai/datasets/cylinder_260527_eval_gt60ua/GT"),
        lq_dst=root / "pairs" / col527 / "LQ",
        gt_dst=root / "pairs" / col527 / "GT",
    )
    write_list(root / "manifests" / f"{col527}.txt", c527)
    shutil.copy2(
        Path("/mnt/home/huang/cst_ai/datasets/cylinder_260527_eval_gt60ua/normalization.json"),
        root / "pairs" / col527 / "normalization.json",
    )
    ua527: dict[str, int] = defaultdict(int)
    for n in c527:
        ua527[ua_of(n)] += 1
    summary[col527] = {
        "count": len(c527),
        "pool_size": len(holdout_pool),
        "ua_counts": dict(sorted(ua527.items())),
        "note": "Never used in training; stratified by µA.",
        "files": c527,
    }

    ph_val_all = read_list(Path("/mnt/home/huang/cst_ai/datasets/aligned_drive/val.txt"))
    ph_val_pool = [n for n in ph_val_all if n not in forbidden_ph]
    ph = pick_stratified_ua(ph_val_pool, args.n_phantom)
    col_ph = "phantom_aligned_drive_val20"
    link_pairs(
        ph,
        lq_src=Path("/mnt/data/cst_ai/datasets/aligned_drive/LQ"),
        gt_src=Path("/mnt/data/cst_ai/datasets/aligned_drive/GT"),
        lq_dst=root / "pairs" / col_ph / "LQ",
        gt_dst=root / "pairs" / col_ph / "GT",
    )
    write_list(root / "manifests" / f"{col_ph}.txt", ph)
    shutil.copy2(
        Path("/mnt/home/huang/cst_ai/datasets/aligned_drive/normalization.json"),
        root / "pairs" / col_ph / "normalization.json",
    )
    ua_ph: dict[str, int] = defaultdict(int)
    for n in ph:
        ua_ph[ua_of(n)] += 1
    summary[col_ph] = {
        "count": len(ph),
        "pool_size": len(ph_val_pool),
        "ua_counts": dict(sorted(ua_ph.items())),
        "files": ph,
    }

    (root / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: {
                    "count": v["count"] if isinstance(v, dict) else v,
                    "ua_counts": v.get("ua_counts") if isinstance(v, dict) else None,
                }
                for k, v in summary.items()
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"Wrote {root}")


if __name__ == "__main__":
    main()
