#!/usr/bin/env python3
"""Validate aligned TIFF pairs and create deterministic train/val metadata.

This script never modifies image pixels and does not copy the source images.
The generated metadata lets BasicSR train directly from shared read-only data.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

TIFF_SUFFIXES = {".tif", ".tiff"}


@dataclass(frozen=True)
class PairInfo:
    filename: str
    width: int
    height: int
    dtype: str
    split: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="Directory containing flat LQ/ and GT/.")
    parser.add_argument("--output-root", type=Path, required=True, help="Directory for metadata files only.")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-mode", choices=("tail", "random"), default="tail")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--raw-min", type=float, default=0.0)
    parser.add_argument("--raw-max", type=float, default=65535.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def list_tiffs(folder: Path) -> dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing TIFF directory: {folder}")
    files = {path.name: path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in TIFF_SUFFIXES}
    if not files:
        raise ValueError(f"No TIFF files found in {folder}")
    return files


def validate_pair_names(lq_files: dict[str, Path], gt_files: dict[str, Path]) -> list[str]:
    lq_names = set(lq_files)
    gt_names = set(gt_files)
    if lq_names != gt_names:
        lq_only = sorted(lq_names - gt_names)
        gt_only = sorted(gt_names - lq_names)
        raise ValueError(
            "LQ/GT filenames do not match. "
            f"LQ-only ({len(lq_only)}): {lq_only[:10]}; "
            f"GT-only ({len(gt_only)}): {gt_only[:10]}"
        )
    return sorted(lq_names)


def inspect_tiff(path: Path) -> tuple[int, int, str]:
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale TIFF, got shape {array.shape}: {path}")
    if array.dtype != np.uint16:
        raise ValueError(f"Expected uint16 TIFF, got {array.dtype}: {path}")
    height, width = array.shape
    return width, height, str(array.dtype)


def assign_splits(
    names: list[str],
    *,
    val_fraction: float,
    split_mode: str,
    seed: int,
) -> dict[str, str]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1")
    if len(names) < 2:
        raise ValueError("At least two image pairs are required for train/val splitting")

    ordered = list(names)
    if split_mode == "random":
        random.Random(seed).shuffle(ordered)
    val_count = min(len(ordered) - 1, max(1, round(len(ordered) * val_fraction)))
    val_names = set(ordered[-val_count:])
    return {name: ("val" if name in val_names else "train") for name in names}


def validate_pairs(
    lq_files: dict[str, Path],
    gt_files: dict[str, Path],
    splits: dict[str, str],
) -> list[PairInfo]:
    pairs: list[PairInfo] = []
    for name in sorted(splits):
        lq_info = inspect_tiff(lq_files[name])
        gt_info = inspect_tiff(gt_files[name])
        if lq_info != gt_info:
            raise ValueError(f"LQ/GT shape or dtype mismatch for {name}: LQ={lq_info}, GT={gt_info}")
        width, height, dtype = lq_info
        pairs.append(PairInfo(name, width, height, dtype, splits[name]))
    return pairs


def ensure_output_root(output_root: Path, *, overwrite: bool) -> None:
    generated = ("train.txt", "val.txt", "manifest.csv", "normalization.json", "dataset.json")
    existing = [output_root / name for name in generated if (output_root / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Metadata already exists; pass --overwrite: {existing}")
    output_root.mkdir(parents=True, exist_ok=True)


def write_metadata(
    output_root: Path,
    source_root: Path,
    pairs: list[PairInfo],
    *,
    split_mode: str,
    seed: int,
    raw_min: float,
    raw_max: float,
) -> None:
    if raw_max <= raw_min:
        raise ValueError("--raw-max must be greater than --raw-min")

    for split in ("train", "val"):
        names = [pair.filename for pair in pairs if pair.split == split]
        (output_root / f"{split}.txt").write_text("".join(f"{name}\n" for name in names), encoding="utf-8")

    with (output_root / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("split", "filename", "width", "height", "dtype"))
        writer.writerows((pair.split, pair.filename, pair.width, pair.height, pair.dtype) for pair in pairs)

    normalization = {
        "method": "fixed_range",
        "raw_min": raw_min,
        "raw_max": raw_max,
        "output_range": [0.0, 1.0],
        "apply_to_lq": True,
        "apply_to_gt": True,
        "source_dtype": "uint16",
    }
    (output_root / "normalization.json").write_text(
        json.dumps(normalization, indent=2) + "\n",
        encoding="utf-8",
    )

    widths = sorted({pair.width for pair in pairs})
    heights = sorted({pair.height for pair in pairs})
    dataset = {
        "source_root": str(source_root),
        "pixel_transform": "none",
        "pair_count": len(pairs),
        "train_pair_count": sum(pair.split == "train" for pair in pairs),
        "val_pair_count": sum(pair.split == "val" for pair in pairs),
        "split_mode": split_mode,
        "seed": seed if split_mode == "random" else None,
        "widths": widths,
        "heights": heights,
    }
    (output_root / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    lq_files = list_tiffs(source_root / "LQ")
    gt_files = list_tiffs(source_root / "GT")
    names = validate_pair_names(lq_files, gt_files)
    splits = assign_splits(
        names,
        val_fraction=args.val_fraction,
        split_mode=args.split_mode,
        seed=args.seed,
    )
    pairs = validate_pairs(lq_files, gt_files, splits)

    train_count = sum(pair.split == "train" for pair in pairs)
    val_count = len(pairs) - train_count
    print(f"Validated {len(pairs)} aligned uint16 TIFF pairs: train={train_count}, val={val_count}")
    print("Pixel transforms: none")
    if args.dry_run:
        print("Dry run: no metadata written")
        return

    ensure_output_root(output_root, overwrite=args.overwrite)
    write_metadata(
        output_root,
        source_root,
        pairs,
        split_mode=args.split_mode,
        seed=args.seed,
        raw_min=args.raw_min,
        raw_max=args.raw_max,
    )
    print(f"Wrote metadata: {output_root}")


if __name__ == "__main__":
    main()
