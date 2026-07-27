#!/usr/bin/env python3
"""Hardlink-merge cylinder_260520_gt90ua + aligned_drive into cylinder_phantom_mix.

Prefixes:
  cyl__*  — from cylinder_260520_gt90ua
  ph__*   — from aligned_drive

Images are hardlinked (no extra TIFF bytes when same filesystem). Metadata uses
domain×µA stratified train/val (~10% val per stratum). Does not include 260527.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

_UA_RE = re.compile(r"(\d+)uA", re.IGNORECASE)
TIFF_SUFFIXES = {".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cyl-root",
        type=Path,
        default=Path("/mnt/data/cst_ai/datasets/cylinder_260520_gt90ua"),
    )
    p.add_argument(
        "--ph-root",
        type=Path,
        default=Path("/mnt/data/cst_ai/datasets/aligned_drive"),
    )
    p.add_argument(
        "--flat-root",
        type=Path,
        default=Path("/mnt/data/cst_ai/datasets/cylinder_phantom_mix"),
    )
    p.add_argument(
        "--meta-root",
        type=Path,
        default=Path("/mnt/home/huang/cst_ai/datasets/cylinder_phantom_mix"),
    )
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.hardlink_to(src)
    except OSError:
        import shutil

        shutil.copy2(src, dst)


def list_tiffs(folder: Path) -> dict[str, Path]:
    files = {path.name: path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in TIFF_SUFFIXES}
    if not files:
        raise ValueError(f"No TIFF in {folder}")
    return files


def ua_of(name: str) -> str:
    match = _UA_RE.search(name)
    return match.group(1) if match else "unknown"


def stratum_key(prefixed: str) -> tuple[str, str]:
    if prefixed.startswith("cyl__"):
        return ("cyl", ua_of(prefixed.removeprefix("cyl__")))
    if prefixed.startswith("ph__"):
        return ("ph", ua_of(prefixed.removeprefix("ph__")))
    return ("other", ua_of(prefixed))


def stratified_splits(
    names: list[str],
    *,
    val_fraction: float,
    seed: int,
) -> dict[str, str]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    rng = random.Random(seed)
    by_stratum: dict[tuple[str, str], list[str]] = defaultdict(list)
    for name in names:
        by_stratum[stratum_key(name)].append(name)

    splits: dict[str, str] = {}
    for _key, group in sorted(by_stratum.items()):
        ordered = list(group)
        rng.shuffle(ordered)
        if len(ordered) == 1:
            splits[ordered[0]] = "train"
            continue
        val_count = min(len(ordered) - 1, max(1, round(len(ordered) * val_fraction)))
        val_set = set(ordered[:val_count])
        for name in ordered:
            splits[name] = "val" if name in val_set else "train"
    return splits


def link_source(src_root: Path, prefix: str, flat_root: Path) -> list[str]:
    lq = list_tiffs(src_root / "LQ")
    gt = list_tiffs(src_root / "GT")
    if set(lq) != set(gt):
        raise ValueError(f"LQ/GT name mismatch under {src_root}")
    names: list[str] = []
    for original in sorted(lq):
        dest_name = f"{prefix}{original}"
        hardlink_or_copy(lq[original], flat_root / "LQ" / dest_name)
        hardlink_or_copy(gt[original], flat_root / "GT" / dest_name)
        names.append(dest_name)
    return names


def main() -> None:
    args = parse_args()
    flat = args.flat_root
    meta = args.meta_root
    if flat.exists() and any(flat.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{flat} exists; pass --overwrite")
    if meta.exists() and any(meta.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{meta} exists; pass --overwrite")

    (flat / "LQ").mkdir(parents=True, exist_ok=True)
    (flat / "GT").mkdir(parents=True, exist_ok=True)
    meta.mkdir(parents=True, exist_ok=True)

    cyl_names = link_source(args.cyl_root, "cyl__", flat)
    ph_names = link_source(args.ph_root, "ph__", flat)
    all_names = sorted(cyl_names + ph_names)
    splits = stratified_splits(all_names, val_fraction=args.val_fraction, seed=args.seed)

    train = [n for n in all_names if splits[n] == "train"]
    val = [n for n in all_names if splits[n] == "val"]
    (meta / "train.txt").write_text("".join(f"{n}\n" for n in train), encoding="utf-8")
    (meta / "val.txt").write_text("".join(f"{n}\n" for n in val), encoding="utf-8")

    with (meta / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("split", "filename", "domain", "ua"))
        for name in all_names:
            domain, ua = stratum_key(name)
            writer.writerow((splits[name], name, domain, ua))

    norm = {
        "method": "fixed_range",
        "raw_min": 0.0,
        "raw_max": 65535.0,
        "output_range": [0.0, 1.0],
        "apply_to_lq": True,
        "apply_to_gt": True,
        "source_dtype": "uint16",
        "note": "Merged cylinder_260520_gt90ua + aligned_drive; hardlink; stratified domain×µA.",
    }
    (meta / "normalization.json").write_text(json.dumps(norm, indent=2) + "\n", encoding="utf-8")

    stratum_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "val": 0})
    for name in all_names:
        domain, ua = stratum_key(name)
        stratum_counts[f"{domain}:{ua}uA"][splits[name]] += 1

    dataset = {
        "name": "cylinder_phantom_mix",
        "flat_root": str(flat),
        "cyl_root": str(args.cyl_root),
        "ph_root": str(args.ph_root),
        "pair_count": len(all_names),
        "train_pair_count": len(train),
        "val_pair_count": len(val),
        "cyl_count": len(cyl_names),
        "ph_count": len(ph_names),
        "split_mode": "stratified_domain_ua",
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "prefixes": {"cylinder": "cyl__", "phantom": "ph__"},
        "stratum_counts": dict(sorted(stratum_counts.items())),
        "excluded": ["cylinder_260527_eval_gt60ua"],
    }
    (meta / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")

    now = datetime.now(timezone.utc).isoformat()
    record = f"""# SPLIT_RECORD — cylinder_phantom_mix

- created_utc: {now}
- flat_root: {flat} (hardlink)
- meta_root: {meta}
- sources: {args.cyl_root} (prefix cyl__), {args.ph_root} (prefix ph__)
- excluded: cylinder_260527_eval_gt60ua
- split_mode: stratified domain×µA
- val_fraction: {args.val_fraction}
- seed: {args.seed}
- pair_count: {len(all_names)} (cyl={len(cyl_names)}, ph={len(ph_names)})
- train: {len(train)}
- val: {len(val)}
- note: Does NOT overwrite cylinder_260520_gt90ua or aligned_drive.
"""
    (meta / "SPLIT_RECORD.md").write_text(record, encoding="utf-8")
    print(
        json.dumps(
            {
                k: dataset[k]
                for k in (
                    "pair_count",
                    "train_pair_count",
                    "val_pair_count",
                    "cyl_count",
                    "ph_count",
                )
            },
            indent=2,
        )
    )
    print(f"Wrote flat={flat} meta={meta}")


if __name__ == "__main__":
    main()
