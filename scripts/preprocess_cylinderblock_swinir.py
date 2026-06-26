#!/usr/bin/env python3
"""Prepare paired Cylinderblock TIFF data for SwinIR grayscale denoising.

The raw Cylinderblock scans are classic uncompressed uint16 TIFF files. This
script intentionally keeps the output in uint16 TIFF form and records the
normalization range separately, so the training dataset can map raw physical
intensity values to float tensors without percentile clipping.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import struct
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("/mnt/data/cst_ai/data_0607")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "datasets" / "swinir_cylinderblock_paired"
INDEX_START = 130
INDEX_END = 750
GT_DIR = "0-300kV_100uA"
TARGET_PLANE = "XY"

TYPE_SIZES = {
    1: 1,  # BYTE
    2: 1,  # ASCII
    3: 2,  # SHORT
    4: 4,  # LONG
    5: 8,  # RATIONAL
    6: 1,  # SBYTE
    7: 1,  # UNDEFINED
    8: 2,  # SSHORT
    9: 4,  # SLONG
    10: 8,  # SRATIONAL
    11: 4,  # FLOAT
    12: 8,  # DOUBLE
}


@dataclass(frozen=True)
class ScanSpec:
    batch: str
    root_rel: Path
    noisy_dirs: tuple[str, ...]


@dataclass(frozen=True)
class TiffImage:
    path: Path
    endian: str
    width: int
    height: int
    bits_per_sample: int
    compression: int
    sample_format: int
    photometric: int
    pixel_bytes: bytes


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
    crop_top: int
    crop_left: int
    width: int
    height: int

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
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create paired LQ/GT uint16 TIFF data for Cylinderblock SwinIR denoising."
    )
    parser.add_argument(
        "--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Root containing raw data_0607 scans."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Output paired dataset root.")
    parser.add_argument("--index-start", type=int, default=INDEX_START, help="First slice index, inclusive.")
    parser.add_argument("--index-end", type=int, default=INDEX_END, help="Last slice index, inclusive.")
    parser.add_argument(
        "--norm-mode",
        choices=("uint16_full", "global_dataset"),
        default="uint16_full",
        help="Fixed linear normalization recorded in normalization.json.",
    )
    parser.add_argument(
        "--split-mode",
        choices=("tail", "modulo"),
        default="tail",
        help="Validation split policy by slice index.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.10, help="Validation fraction for tail split.")
    parser.add_argument("--val-count", type=int, default=None, help="Explicit validation slice count for tail split.")
    parser.add_argument("--val-modulo", type=int, default=10, help="Modulo divisor for modulo split.")
    parser.add_argument("--val-remainder", type=int, default=0, help="Modulo remainder assigned to validation.")
    parser.add_argument("--max-slices", type=int, default=None, help="Limit slices for quick validation runs.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and summarize without writing output files.")
    parser.add_argument(
        "--stats-only", action="store_true", help="Only compute summary/statistics without writing output files."
    )
    parser.add_argument("--overwrite", action="store_true", help="Remove an existing output-root before writing.")
    parser.add_argument(
        "--lq-mode",
        choices=("symlink", "hardlink", "copy"),
        default="copy",
        help="How to materialize LQ TIFF files.",
    )
    parser.add_argument(
        "--gt-duplicate-mode",
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
        help="How to materialize duplicate GT names after the first cropped GT is written.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable periodic progress output.")
    return parser.parse_args()


def parse_scan_condition(scan_dir: str) -> tuple[int, int]:
    """Return voltage kV and current uA from names like '4-110kV_30uA'."""
    try:
        condition = scan_dir.split("-", 1)[1]
        voltage, current = condition.split("_", 1)
        return int(voltage.removesuffix("kV")), int(current.removesuffix("uA"))
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Cannot parse scan condition from directory name: {scan_dir}") from exc


def tiff_value(endian: str, typ: int, count: int, raw_value: bytes) -> int | tuple[int, ...] | bytes:
    size = TYPE_SIZES.get(typ)
    if size is None:
        return raw_value
    total_size = size * count
    value_bytes = raw_value[:total_size]
    if typ == 3:
        values = struct.unpack(endian + f"{count}H", value_bytes)
    elif typ == 4:
        values = struct.unpack(endian + f"{count}I", value_bytes)
    elif typ == 8:
        values = struct.unpack(endian + f"{count}h", value_bytes)
    elif typ == 9:
        values = struct.unpack(endian + f"{count}i", value_bytes)
    else:
        return value_bytes
    return values[0] if count == 1 else values


def read_tiff(path: Path, load_pixels: bool = True) -> TiffImage:
    with path.open("rb") as file:
        header = file.read(8)
        byte_order = header[:2]
        if byte_order == b"II":
            endian = "<"
        elif byte_order == b"MM":
            endian = ">"
        else:
            raise ValueError(f"Not a classic TIFF file: {path}")

        magic = struct.unpack(endian + "H", header[2:4])[0]
        if magic != 42:
            raise ValueError(f"Unsupported TIFF magic {magic}: {path}")

        ifd_offset = struct.unpack(endian + "I", header[4:8])[0]
        file.seek(ifd_offset)
        entry_count = struct.unpack(endian + "H", file.read(2))[0]
        tags: dict[int, int | tuple[int, ...] | bytes] = {}
        for _ in range(entry_count):
            entry = file.read(12)
            tag, typ, count = struct.unpack(endian + "HHI", entry[:8])
            value_or_offset = entry[8:12]
            total_size = TYPE_SIZES.get(typ, 1) * count
            if total_size <= 4:
                value_bytes = value_or_offset[:total_size]
            else:
                current_pos = file.tell()
                value_offset = struct.unpack(endian + "I", value_or_offset)[0]
                file.seek(value_offset)
                value_bytes = file.read(total_size)
                file.seek(current_pos)
            tags[tag] = tiff_value(endian, typ, count, value_bytes)

    width = int(tags[256])
    height = int(tags[257])
    bits = tags.get(258, 16)
    bits_per_sample = int(bits[0] if isinstance(bits, tuple) else bits)
    compression = int(tags.get(259, 1))
    photometric = int(tags.get(262, 1))
    sample_format = int(tags.get(339, 1))
    offsets = tags.get(273) or tags.get(324)
    byte_counts = tags.get(279) or tags.get(325)
    if offsets is None or byte_counts is None:
        raise ValueError(f"TIFF is missing strip/tile offsets or byte counts: {path}")
    if isinstance(offsets, int):
        offsets = (offsets,)
    if isinstance(byte_counts, int):
        byte_counts = (byte_counts,)

    if compression != 1 or bits_per_sample != 16 or sample_format != 1:
        raise ValueError(
            "Only uncompressed unsigned uint16 grayscale TIFF is supported: "
            f"{path} (compression={compression}, bits={bits_per_sample}, sample_format={sample_format})"
        )

    pixel_bytes = b""
    if load_pixels:
        with path.open("rb") as file:
            chunks = []
            for offset, count in zip(offsets, byte_counts, strict=True):
                file.seek(int(offset))
                chunks.append(file.read(int(count)))
            pixel_bytes = b"".join(chunks)
    expected_bytes = width * height * 2
    if load_pixels and len(pixel_bytes) < expected_bytes:
        raise ValueError(
            f"TIFF pixel payload is shorter than expected in {path}: {len(pixel_bytes)} < {expected_bytes}"
        )
    return TiffImage(
        path=path,
        endian=endian,
        width=width,
        height=height,
        bits_per_sample=bits_per_sample,
        compression=compression,
        sample_format=sample_format,
        photometric=photometric,
        pixel_bytes=pixel_bytes[:expected_bytes] if load_pixels else b"",
    )


def crop_u16_bytes(image: TiffImage, left: int, top: int, width: int, height: int) -> bytes:
    if left < 0 or top < 0 or left + width > image.width or top + height > image.height:
        raise ValueError(f"Crop is outside image bounds for {image.path}")

    row_bytes = image.width * 2
    crop_row_bytes = width * 2
    rows = []
    for row in range(top, top + height):
        start = row * row_bytes + left * 2
        rows.append(image.pixel_bytes[start : start + crop_row_bytes])
    cropped = b"".join(rows)
    if image.endian == ">":
        cropped = swap_u16_endian(cropped)
    return cropped


def swap_u16_endian(data: bytes) -> bytes:
    return b"".join(data[i + 1 : i + 2] + data[i : i + 1] for i in range(0, len(data), 2))


def pack_ifd_entry(tag: int, typ: int, count: int, value: int) -> bytes:
    value_bytes = struct.pack("<H", value) + b"\x00\x00" if typ == 3 and count == 1 else struct.pack("<I", value)
    return struct.pack("<HHI", tag, typ, count) + value_bytes


def write_u16_tiff(path: Path, pixel_bytes: bytes, width: int, height: int, photometric: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    expected_bytes = width * height * 2
    if len(pixel_bytes) != expected_bytes:
        raise ValueError(f"Pixel payload size mismatch for {path}: {len(pixel_bytes)} != {expected_bytes}")

    entries = [
        (256, 4, 1, width),
        (257, 4, 1, height),
        (258, 3, 1, 16),
        (259, 3, 1, 1),
        (262, 3, 1, photometric),
        (273, 4, 1, 0),  # filled after IFD size is known
        (277, 3, 1, 1),
        (278, 4, 1, height),
        (279, 4, 1, expected_bytes),
        (296, 3, 1, 1),
        (339, 3, 1, 1),
    ]
    ifd_size = 2 + len(entries) * 12 + 4
    pixel_offset = 8 + ifd_size
    entries = [(tag, typ, count, pixel_offset if tag == 273 else value) for tag, typ, count, value in entries]

    with path.open("wb") as file:
        file.write(b"II")
        file.write(struct.pack("<H", 42))
        file.write(struct.pack("<I", 8))
        file.write(struct.pack("<H", len(entries)))
        for entry in entries:
            file.write(pack_ifd_entry(*entry))
        file.write(struct.pack("<I", 0))
        file.write(pixel_bytes)


def u16_stats(pixel_bytes: bytes, endian: str = "<") -> tuple[int, int, int, int]:
    values = array("H")
    values.frombytes(pixel_bytes)
    host_little = struct.pack("=H", 1) == b"\x01\x00"
    file_little = endian == "<"
    if host_little != file_little:
        values.byteswap()
    if not values:
        return 0, 0, 0, 0
    min_value = min(values)
    max_value = max(values)
    zero_count = values.count(0)
    saturated_count = values.count(65535)
    return min_value, max_value, zero_count, saturated_count


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
                        crop_top=-1,
                        crop_left=-1,
                        width=-1,
                        height=-1,
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
        if lq.width > gt.width or lq.height > gt.height:
            raise ValueError(
                f"LQ is larger than GT for {record.pair_id}: LQ={lq.width}x{lq.height}, GT={gt.width}x{gt.height}"
            )
        crop_left = (gt.width - lq.width) // 2
        crop_top = (gt.height - lq.height) // 2
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
                crop_top=crop_top,
                crop_left=crop_left,
                width=lq.width,
                height=lq.height,
            )
        )
    return enriched


def prepare_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output root already exists. Use --overwrite to replace it: {output_root}")
        shutil.rmtree(output_root)
    for split in ("train", "val"):
        (output_root / split / "LQ").mkdir(parents=True, exist_ok=True)
        (output_root / split / "GT").mkdir(parents=True, exist_ok=True)


def relative_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    target = os.path.relpath(src.resolve(), dst.parent.resolve())
    dst.symlink_to(target)


def materialize_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        relative_symlink(src, dst)
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported materialization mode: {mode}")


def materialize_duplicate(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
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
    gt_cache: set[tuple[Path, int, int, int, int]] = set()
    for record in records:
        lq = read_tiff(record.lq_src)
        mn, mx, zeros, saturated = u16_stats(lq.pixel_bytes, lq.endian)
        stats["raw_min"] = min(stats["raw_min"], mn)
        stats["raw_max"] = max(stats["raw_max"], mx)
        stats["zero_pixels"] += zeros
        stats["saturated_pixels"] += saturated
        stats["images_scanned"] += 1

        gt_key = (record.gt_src, record.crop_left, record.crop_top, record.width, record.height)
        if gt_key in gt_cache:
            continue
        gt_cache.add(gt_key)
        gt = read_tiff(record.gt_src)
        cropped = crop_u16_bytes(gt, record.crop_left, record.crop_top, record.width, record.height)
        mn, mx, zeros, saturated = u16_stats(cropped, "<")
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
    manifest_path = output_root / "manifest.csv"
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
        "gt_crop_top",
        "gt_crop_left",
        "width",
        "height",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
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
                    "gt_crop_top": record.crop_top,
                    "gt_crop_left": record.crop_left,
                    "width": record.width,
                    "height": record.height,
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
        "gt_crop": {
            "mode": "center",
            "height": records[0].height if records else None,
            "width": records[0].width if records else None,
            "top": records[0].crop_top if records else None,
            "left": records[0].crop_left if records else None,
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
        materialize_file(record.lq_src, record.lq_out, args.lq_mode)

        gt_key = (record.split, record.batch, record.slice_index)
        cached_gt = gt_cache.get(gt_key)
        if cached_gt is None:
            gt = read_tiff(record.gt_src)
            cropped = crop_u16_bytes(gt, record.crop_left, record.crop_top, record.width, record.height)
            write_u16_tiff(record.gt_out, cropped, record.width, record.height, gt.photometric)
            gt_cache[gt_key] = record.gt_out
        else:
            materialize_duplicate(cached_gt, record.gt_out, args.gt_duplicate_mode)

        if not args.no_progress and (idx % 250 == 0 or idx == len(records)):
            print(f"Processed {idx}/{len(records)} pairs", flush=True)

    return {
        "unique_cropped_gt": len(gt_cache),
        "materialized_pairs": len(records),
    }


def summarize(records: list[PairRecord], stats: dict[str, int] | None = None) -> None:
    train_count = sum(1 for record in records if record.split == "train")
    val_count = sum(1 for record in records if record.split == "val")
    currents = sorted({f"{record.batch}:{record.voltage_kv}kV_{record.current_ua}uA" for record in records})
    print(f"Pairs: total={len(records)} train={train_count} val={val_count}")
    print(f"Image size after pairing: {records[0].width}x{records[0].height}" if records else "No records")
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
    print(f"Unique cropped GT files: {create_result['unique_cropped_gt']}")


if __name__ == "__main__":
    main()
