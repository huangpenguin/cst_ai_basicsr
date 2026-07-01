#!/usr/bin/env python3
"""Prepare paired Cylinderblock TIFF data for SwinIR grayscale denoising.

Geometry (per batch):
  - Teacher GT (300 kV / ``0-300kV_100uA``): full 1530x1530, no rotation.
  - Noisy LQ scans: batch-specific clockwise rotation (nearest-edge fill), then bilinear
    resize to 1530x1530 so LQ/GT share the same canvas without black wedges.
    - ``260520`` (Cylinderblock 2): 193° CW
    - ``260527`` (Cylinderblock 4): 91° CW

Outputs uint16 TIFF pairs plus ``manifest.csv`` and ``normalization.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from cylinderblock_tiff_io import (  # noqa: E402
    TiffImage,
    array_to_u16_bytes,
    read_tiff,
    resize_u16,
    rotate_u16_cw,
    u16_bytes_to_array,
    u16_stats,
    write_u16_tiff,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("/mnt/data/cst_ai/data_0607")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "datasets" / "swinir_cylinderblock_paired"
INDEX_START = 130
INDEX_END = 750
GT_DIR = "0-300kV_100uA"
TARGET_PLANE = "XY"
GT_SIZE = 1530
LQ_SIZE = 1280
LQ_FIT_MODE = "resize"


@dataclass(frozen=True)
class ScanSpec:
    batch: str
    root_rel: Path
    noisy_dirs: tuple[str, ...]
    lq_rotation_deg_cw: float


@dataclass(frozen=True)
class PairRecord:
    split: str
    batch: str
    noisy_dir: str
    voltage_kv: int
    current_ua: int
    slice_index: int
    lq_src: Path
    gt_src: Path
    lq_out: Path
    gt_out: Path
    gt_width: int
    gt_height: int
    lq_width: int
    lq_height: int
    lq_rotation_deg_cw: float
    lq_fit_mode: str
    lq_scale: float
    output_width: int
    output_height: int

    @property
    def pair_id(self) -> str:
        return self.lq_out.stem


SCAN_SPECS = (
    ScanSpec(
        batch="260520",
        root_rel=Path("260520_2-Cylinderblock_data") / "260520_cylinderblock image data",
        noisy_dirs=(
            "4-110kV_30uA",
            "6-110kV_50uA",
            "7-110kV_60uA",
            "8-110kV_70uA",
            "9-110kV_80uA",
            "1-110kV_90uA",
        ),
        lq_rotation_deg_cw=193.0,
    ),
    ScanSpec(
        batch="260527",
        root_rel=Path("260527_4-Cylinderblock_data") / "260527_cylinderblock image data",
        noisy_dirs=(
            "4-150kV_30uA",
            "3-150kV_40uA",
            "2-150kV_50uA",
            "1-150kV_60uA",
        ),
        lq_rotation_deg_cw=91.0,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--index-start", type=int, default=INDEX_START)
    parser.add_argument("--index-end", type=int, default=INDEX_END)
    parser.add_argument(
        "--lq-rotation-deg",
        type=float,
        default=None,
        help="Optional override: apply the same clockwise LQ rotation to all batches.",
    )
    parser.add_argument(
        "--norm-mode",
        choices=("uint16_full", "global_dataset"),
        default="uint16_full",
    )
    parser.add_argument("--split-mode", choices=("tail", "modulo"), default="tail")
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--val-count", type=int, default=None)
    parser.add_argument("--val-modulo", type=int, default=10)
    parser.add_argument("--val-remainder", type=int, default=0)
    parser.add_argument("--max-slices", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stats-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--gt-duplicate-mode",
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def parse_scan_condition(scan_dir: str) -> tuple[int, int]:
    try:
        condition = scan_dir.split("-", 1)[1]
        voltage, current = condition.split("_", 1)
        return int(voltage.removesuffix("kV")), int(current.removesuffix("uA"))
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Cannot parse scan condition from directory name: {scan_dir}") from exc


def choose_indices(args: argparse.Namespace) -> list[int]:
    if args.index_start > args.index_end:
        raise ValueError("--index-start must be <= --index-end")
    indices = list(range(args.index_start, args.index_end + 1))
    if args.max_slices is not None:
        if args.max_slices <= 0:
            raise ValueError("--max-slices must be positive when provided")
        indices = indices[: args.max_slices]
    return indices


def assign_split(index: int, indices: list[int], args: argparse.Namespace) -> str:
    if args.split_mode == "modulo":
        return "val" if index % args.val_modulo == args.val_remainder else "train"
    if len(indices) <= 1:
        return "train"
    val_count = args.val_count
    if val_count is None:
        val_count = max(1, math.ceil(len(indices) * args.val_fraction))
    val_count = min(max(val_count, 0), len(indices) - 1)
    val_indices = set(indices[-val_count:])
    return "val" if index in val_indices else "train"


def build_records(args: argparse.Namespace) -> list[PairRecord]:
    indices = choose_indices(args)
    records: list[PairRecord] = []
    missing: list[Path] = []

    for spec in SCAN_SPECS:
        scan_root = args.data_root / spec.root_rel
        gt_xy = scan_root / GT_DIR / TARGET_PLANE
        if not gt_xy.is_dir():
            missing.append(gt_xy)
            continue

        for noisy_dir in spec.noisy_dirs:
            voltage_kv, current_ua = parse_scan_condition(noisy_dir)
            lq_xy = scan_root / noisy_dir / TARGET_PLANE
            if not lq_xy.is_dir():
                missing.append(lq_xy)
                continue

            for index in indices:
                lq_src = lq_xy / f"data_{index:04d}.tif"
                gt_src = gt_xy / f"data_{index:04d}.tif"
                if not lq_src.is_file():
                    missing.append(lq_src)
                if not gt_src.is_file():
                    missing.append(gt_src)
                filename = f"{spec.batch}_{voltage_kv}kV_{current_ua}uA_slice{index:04d}.tif"
                split = assign_split(index, indices, args)
                records.append(
                    PairRecord(
                        split=split,
                        batch=spec.batch,
                        noisy_dir=noisy_dir,
                        voltage_kv=voltage_kv,
                        current_ua=current_ua,
                        slice_index=index,
                        lq_src=lq_src,
                        gt_src=gt_src,
                        lq_out=args.output_root / split / "LQ" / filename,
                        gt_out=args.output_root / split / "GT" / filename,
                        gt_width=-1,
                        gt_height=-1,
                        lq_width=-1,
                        lq_height=-1,
                        lq_rotation_deg_cw=(
                            args.lq_rotation_deg if args.lq_rotation_deg is not None else spec.lq_rotation_deg_cw
                        ),
                        lq_fit_mode=LQ_FIT_MODE,
                        lq_scale=-1.0,
                        output_width=-1,
                        output_height=-1,
                    )
                )

    if missing:
        sample = "\n".join(f"  - {path}" for path in missing[:20])
        suffix = "" if len(missing) <= 20 else f"\n  ... and {len(missing) - 20} more"
        raise FileNotFoundError(f"Missing required input files/directories:\n{sample}{suffix}")
    return records


def validate_and_enrich_records(records: Iterable[PairRecord]) -> list[PairRecord]:
    enriched: list[PairRecord] = []
    meta_cache: dict[Path, TiffImage] = {}
    for record in records:
        lq = meta_cache.setdefault(record.lq_src, read_tiff(record.lq_src, load_pixels=False))
        gt = meta_cache.setdefault(record.gt_src, read_tiff(record.gt_src, load_pixels=False))
        if lq.width != LQ_SIZE or lq.height != LQ_SIZE:
            raise ValueError(f"Expected LQ {LQ_SIZE}x{LQ_SIZE}, got {lq.width}x{lq.height}: {record.lq_src}")
        if gt.width != GT_SIZE or gt.height != GT_SIZE:
            raise ValueError(f"Expected GT {GT_SIZE}x{GT_SIZE}, got {gt.width}x{gt.height}: {record.gt_src}")
        lq_scale = gt.width / lq.width
        enriched.append(
            PairRecord(
                split=record.split,
                batch=record.batch,
                noisy_dir=record.noisy_dir,
                voltage_kv=record.voltage_kv,
                current_ua=record.current_ua,
                slice_index=record.slice_index,
                lq_src=record.lq_src,
                gt_src=record.gt_src,
                lq_out=record.lq_out,
                gt_out=record.gt_out,
                gt_width=gt.width,
                gt_height=gt.height,
                lq_width=lq.width,
                lq_height=lq.height,
                lq_rotation_deg_cw=record.lq_rotation_deg_cw,
                lq_fit_mode=LQ_FIT_MODE,
                lq_scale=lq_scale,
                output_width=gt.width,
                output_height=gt.height,
            )
        )
    return enriched


def process_lq_pixels(record: PairRecord) -> bytes:
    lq = read_tiff(record.lq_src)
    arr = u16_bytes_to_array(lq)
    rotated = rotate_u16_cw(arr, record.lq_rotation_deg_cw)
    fitted = resize_u16(rotated, record.output_height, record.output_width)
    return array_to_u16_bytes(fitted)


def process_gt_pixels(record: PairRecord) -> tuple[bytes, int]:
    gt = read_tiff(record.gt_src)
    return array_to_u16_bytes(u16_bytes_to_array(gt)), gt.photometric


def prepare_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output root already exists. Use --overwrite to replace it: {output_root}")
        shutil.rmtree(output_root)
    for split in ("train", "val"):
        (output_root / split / "LQ").mkdir(parents=True, exist_ok=True)
        (output_root / split / "GT").mkdir(parents=True, exist_ok=True)


def materialize_duplicate(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        from cylinderblock_tiff_io import relative_symlink

        relative_symlink(src, dst)
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported GT duplicate mode: {mode}")


def compute_dataset_stats(records: Iterable[PairRecord]) -> dict[str, int]:
    stats = {
        "raw_min": 65535,
        "raw_max": 0,
        "zero_pixels": 0,
        "saturated_pixels": 0,
        "images_scanned": 0,
    }
    gt_cache: set[tuple[str, str, int]] = set()
    for record in records:
        lq_bytes = process_lq_pixels(record)
        mn, mx, zeros, saturated = u16_stats(lq_bytes)
        stats["raw_min"] = min(stats["raw_min"], mn)
        stats["raw_max"] = max(stats["raw_max"], mx)
        stats["zero_pixels"] += zeros
        stats["saturated_pixels"] += saturated
        stats["images_scanned"] += 1

        gt_key = (record.split, record.batch, record.slice_index)
        if gt_key in gt_cache:
            continue
        gt_cache.add(gt_key)
        gt_bytes, _ = process_gt_pixels(record)
        mn, mx, zeros, saturated = u16_stats(gt_bytes)
        stats["raw_min"] = min(stats["raw_min"], mn)
        stats["raw_max"] = max(stats["raw_max"], mx)
        stats["zero_pixels"] += zeros
        stats["saturated_pixels"] += saturated
        stats["images_scanned"] += 1
    if stats["images_scanned"] == 0:
        stats["raw_min"] = 0
        stats["raw_max"] = 65535
    return stats


def write_manifest(records: list[PairRecord], output_root: Path) -> None:
    fieldnames = [
        "split",
        "pair_id",
        "batch",
        "noisy_dir",
        "voltage_kv",
        "current_ua",
        "slice_index",
        "lq_path",
        "gt_path",
        "source_lq_path",
        "source_gt_path",
        "lq_rotation_deg_cw",
        "lq_fit_mode",
        "lq_scale",
        "lq_width",
        "lq_height",
        "gt_width",
        "gt_height",
        "output_width",
        "output_height",
    ]
    with (output_root / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "split": record.split,
                    "pair_id": record.pair_id,
                    "batch": record.batch,
                    "noisy_dir": record.noisy_dir,
                    "voltage_kv": record.voltage_kv,
                    "current_ua": record.current_ua,
                    "slice_index": record.slice_index,
                    "lq_path": record.lq_out.relative_to(output_root),
                    "gt_path": record.gt_out.relative_to(output_root),
                    "source_lq_path": record.lq_src,
                    "source_gt_path": record.gt_src,
                    "lq_rotation_deg_cw": record.lq_rotation_deg_cw,
                    "lq_fit_mode": record.lq_fit_mode,
                    "lq_scale": record.lq_scale,
                    "lq_width": record.lq_width,
                    "lq_height": record.lq_height,
                    "gt_width": record.gt_width,
                    "gt_height": record.gt_height,
                    "output_width": record.output_width,
                    "output_height": record.output_height,
                }
            )


def write_normalization(
    records: list[PairRecord], output_root: Path, args: argparse.Namespace, stats: dict[str, int]
) -> None:
    if args.norm_mode == "uint16_full":
        raw_min, raw_max = 0.0, 65535.0
    else:
        raw_min, raw_max = float(stats["raw_min"]), float(stats["raw_max"])
        if raw_max <= raw_min:
            raw_min, raw_max = 0.0, 65535.0

    payload = {
        "method": args.norm_mode,
        "raw_min": raw_min,
        "raw_max": raw_max,
        "output_range": [0.0, 1.0],
        "apply_to_lq": True,
        "apply_to_gt": True,
        "source_dtype": "uint16",
        "index_start": args.index_start,
        "index_end": args.index_end,
        "excluded_currents_ua": [10, 20],
        "plane": TARGET_PLANE,
        "gt_dir": GT_DIR,
        "pair_count": len(records),
        "train_pair_count": sum(1 for record in records if record.split == "train"),
        "val_pair_count": sum(1 for record in records if record.split == "val"),
        "geometry": {
            "gt": {"width": GT_SIZE, "height": GT_SIZE, "rotation_deg_cw": 0, "crop": None},
            "lq_by_batch": {
                spec.batch: {
                    "native_width": LQ_SIZE,
                    "native_height": LQ_SIZE,
                    "rotation_deg_cw": spec.lq_rotation_deg_cw,
                    "fit_mode": LQ_FIT_MODE,
                    "edge_mode": "nearest",
                    "scale": GT_SIZE / LQ_SIZE,
                    "output_width": GT_SIZE,
                    "output_height": GT_SIZE,
                }
                for spec in SCAN_SPECS
            },
        },
        "stats": stats,
    }
    with (output_root / "normalization.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def create_dataset(records: list[PairRecord], args: argparse.Namespace) -> dict[str, int]:
    prepare_output_root(args.output_root, args.overwrite)

    gt_cache: dict[tuple[str, str, int], Path] = {}
    for idx, record in enumerate(records, start=1):
        lq_bytes = process_lq_pixels(record)
        write_u16_tiff(record.lq_out, lq_bytes, record.output_width, record.output_height)

        gt_key = (record.split, record.batch, record.slice_index)
        cached_gt = gt_cache.get(gt_key)
        if cached_gt is None:
            gt_bytes, photometric = process_gt_pixels(record)
            write_u16_tiff(record.gt_out, gt_bytes, record.gt_width, record.gt_height, photometric)
            gt_cache[gt_key] = record.gt_out
        else:
            materialize_duplicate(cached_gt, record.gt_out, args.gt_duplicate_mode)

        if not args.no_progress and (idx % 250 == 0 or idx == len(records)):
            print(f"Processed {idx}/{len(records)} pairs", flush=True)

    return {"unique_gt": len(gt_cache), "materialized_pairs": len(records)}


def summarize(records: list[PairRecord], stats: dict[str, int] | None = None) -> None:
    train_count = sum(1 for record in records if record.split == "train")
    val_count = sum(1 for record in records if record.split == "val")
    currents = sorted({f"{record.batch}:{record.voltage_kv}kV_{record.current_ua}uA" for record in records})
    print(f"Pairs: total={len(records)} train={train_count} val={val_count}")
    if records:
        print(
            f"Output size: {records[0].output_width}x{records[0].output_height} "
            f"(GT full {GT_SIZE}, LQ {LQ_SIZE} -> rotate + {LQ_FIT_MODE} to {GT_SIZE})"
        )
        rotations = {record.batch: record.lq_rotation_deg_cw for record in records}
        for batch in sorted(rotations):
            print(f"  batch {batch}: LQ rotate {rotations[batch]}° CW")
    print("Noisy conditions:")
    for current in currents:
        print(f"  - {current}")
    if stats is not None:
        print(
            "Stats: "
            f"raw_min={stats['raw_min']} raw_max={stats['raw_max']} "
            f"zero_pixels={stats['zero_pixels']} saturated_pixels={stats['saturated_pixels']} "
            f"images_scanned={stats['images_scanned']}"
        )


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()

    records = validate_and_enrich_records(build_records(args))
    need_stats = args.stats_only or args.norm_mode == "global_dataset"
    stats = (
        compute_dataset_stats(records)
        if need_stats
        else {
            "raw_min": 0,
            "raw_max": 65535,
            "zero_pixels": 0,
            "saturated_pixels": 0,
            "images_scanned": 0,
        }
    )
    summarize(records, stats if need_stats else None)

    if args.dry_run or args.stats_only:
        print("Dry run/statistics mode: no output files were written.")
        return

    create_result = create_dataset(records, args)
    write_manifest(records, args.output_root)
    write_normalization(records, args.output_root, args, stats)
    print(f"Wrote paired dataset to: {args.output_root}")
    print(f"Unique GT files: {create_result['unique_gt']}")


if __name__ == "__main__":
    main()
