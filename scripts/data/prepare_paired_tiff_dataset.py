#!/usr/bin/env python3
"""Validate aligned TIFF pairs and create deterministic split metadata.

This script never modifies image pixels and does not copy the source images.
The generated metadata lets BasicSR train directly from shared read-only data.

Modes:
  - default: train/val split (optional stratified-by-current when --stratify-current)
  - candidates: stratified test holdout + candidates.txt (no train/val yet)
  - finalize: train/val from an edited candidates list (test.txt kept if present)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

TIFF_SUFFIXES = {".tif", ".tiff"}
CURRENT_RE = re.compile(r"(?P<ua>\d+)\s*uA|(?P<ma>\d+)\s*mA", re.IGNORECASE)


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
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.0,
        help="Per-current test holdout fraction (used with --mode candidates).",
    )
    parser.add_argument(
        "--mode",
        choices=("train_val", "candidates", "finalize"),
        default="train_val",
        help="train_val: write train/val; candidates: test+candidates only; "
        "finalize: train/val from candidates.txt (or --candidates-file).",
    )
    parser.add_argument(
        "--candidates-file",
        type=Path,
        default=None,
        help="For --mode finalize: path to edited candidates list (default: output-root/candidates.txt).",
    )
    parser.add_argument("--split-mode", choices=("tail", "random"), default="tail")
    parser.add_argument(
        "--stratify-current",
        action="store_true",
        help="Stratify train/val (and test holdout) by LQ current parsed from filename.",
    )
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


def current_key_from_filename(filename: str) -> str:
    """Parse LQ current label from pair id (e.g. ``...150kV4mA...`` / ``...80uA_...``)."""
    match = CURRENT_RE.search(filename)
    if match is None:
        return "unknown"
    if match.group("ma") is not None:
        return f"{int(match.group('ma'))}mA"
    return f"{int(match.group('ua'))}uA"


def group_by_current(names: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for name in names:
        groups[current_key_from_filename(name)].append(name)
    return dict(groups)


def fraction_count(n: int, fraction: float) -> int:
    if n <= 0:
        return 0
    return min(n - 1 if n > 1 else n, max(1 if n >= 1 and fraction > 0 else 0, round(n * fraction)))


def assign_splits(
    names: list[str],
    *,
    val_fraction: float,
    split_mode: str,
    seed: int,
    stratify_current: bool = False,
) -> dict[str, str]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1")
    if len(names) < 2:
        raise ValueError("At least two image pairs are required for train/val splitting")

    splits: dict[str, str] = {}
    if stratify_current:
        rng = random.Random(seed)
        for _key, group in sorted(group_by_current(names).items()):
            ordered = list(group)
            if split_mode == "random":
                rng.shuffle(ordered)
            else:
                ordered = sorted(ordered)
            val_count = fraction_count(len(ordered), val_fraction)
            # Keep at least one train when possible.
            if len(ordered) >= 2:
                val_count = min(val_count, len(ordered) - 1)
            val_names = set(ordered[-val_count:] if val_count else [])
            for name in group:
                splits[name] = "val" if name in val_names else "train"
        return splits

    ordered = list(names)
    if split_mode == "random":
        random.Random(seed).shuffle(ordered)
    else:
        ordered = sorted(ordered)
    val_count = min(len(ordered) - 1, max(1, round(len(ordered) * val_fraction)))
    val_names = set(ordered[-val_count:])
    return {name: ("val" if name in val_names else "train") for name in names}


def assign_test_holdout(
    names: list[str],
    *,
    test_fraction: float,
    seed: int,
) -> dict[str, str]:
    """Stratified-by-current test holdout; remainder labeled ``candidate``."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("--test-fraction must be between 0 and 1 for candidates mode")
    if not names:
        raise ValueError("No pairs for test holdout")

    rng = random.Random(seed)
    splits: dict[str, str] = {}
    for _key, group in sorted(group_by_current(names).items()):
        ordered = list(group)
        rng.shuffle(ordered)
        test_count = fraction_count(len(ordered), test_fraction)
        if len(ordered) >= 2:
            test_count = min(test_count, len(ordered) - 1)
        test_names = set(ordered[:test_count] if test_count else [])
        for name in group:
            splits[name] = "test" if name in test_names else "candidate"
    return splits


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


def read_name_list(path: Path) -> list[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise ValueError(f"Empty name list: {path}")
    return names


def ensure_output_root(output_root: Path, *, overwrite: bool, mode: str) -> None:
    if mode == "candidates":
        generated = ("test.txt", "candidates.txt", "manifest.csv", "dataset.json", "SPLIT_RECORD.md")
    elif mode == "finalize":
        generated = ("train.txt", "val.txt", "manifest.csv", "normalization.json", "dataset.json")
    else:
        generated = ("train.txt", "val.txt", "manifest.csv", "normalization.json", "dataset.json")
    existing = [output_root / name for name in generated if (output_root / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Metadata already exists; pass --overwrite: {existing}")
    output_root.mkdir(parents=True, exist_ok=True)


def write_name_list(path: Path, names: list[str]) -> None:
    path.write_text("".join(f"{name}\n" for name in names), encoding="utf-8")


def write_manifest(output_root: Path, pairs: list[PairInfo]) -> None:
    with (output_root / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("split", "filename", "width", "height", "dtype", "current"))
        writer.writerows(
            (pair.split, pair.filename, pair.width, pair.height, pair.dtype, current_key_from_filename(pair.filename))
            for pair in pairs
        )


def write_normalization(output_root: Path, *, raw_min: float, raw_max: float) -> None:
    if raw_max <= raw_min:
        raise ValueError("--raw-max must be greater than --raw-min")
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


def counts_by_current(pairs: list[PairInfo], split: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for pair in pairs:
        if pair.split == split:
            counts[current_key_from_filename(pair.filename)] += 1
    return dict(sorted(counts.items()))


def write_split_record_candidates(
    output_root: Path,
    source_root: Path,
    pairs: list[PairInfo],
    *,
    test_fraction: float,
    seed: int,
) -> None:
    test_n = sum(p.split == "test" for p in pairs)
    cand_n = sum(p.split == "candidate" for p in pairs)
    lines = [
        "# SPLIT_RECORD — candidates stage (test holdout)",
        "",
        f"- source_root: `{source_root}`",
        f"- meta_root: `{output_root}`",
        "- mode: candidates",
        f"- test_fraction: {test_fraction} (stratified by current)",
        f"- seed: {seed}",
        f"- pair_count: {len(pairs)}",
        f"- test: {test_n}",
        f"- candidates: {cand_n}",
        f"- test_by_current: {counts_by_current(pairs, 'test')}",
        f"- candidates_by_current: {counts_by_current(pairs, 'candidate')}",
        "",
        "Next: edit `candidates.txt` to remove blank slices, then run",
        "`prepare_paired_tiff_dataset.py --mode finalize ...`.",
        "",
    ]
    (output_root / "SPLIT_RECORD.md").write_text("\n".join(lines), encoding="utf-8")


def write_metadata_train_val(
    output_root: Path,
    source_root: Path,
    pairs: list[PairInfo],
    *,
    split_mode: str,
    seed: int,
    raw_min: float,
    raw_max: float,
    stratify_current: bool,
    test_names: list[str] | None = None,
) -> None:
    for split in ("train", "val"):
        names = [pair.filename for pair in pairs if pair.split == split]
        write_name_list(output_root / f"{split}.txt", names)

    write_manifest(output_root, pairs)
    write_normalization(output_root, raw_min=raw_min, raw_max=raw_max)

    widths = sorted({pair.width for pair in pairs})
    heights = sorted({pair.height for pair in pairs})
    dataset: dict[str, object] = {
        "source_root": str(source_root),
        "pixel_transform": "none",
        "pair_count": len(pairs) + (len(test_names) if test_names else 0),
        "train_pair_count": sum(pair.split == "train" for pair in pairs),
        "val_pair_count": sum(pair.split == "val" for pair in pairs),
        "test_pair_count": len(test_names) if test_names else 0,
        "split_mode": split_mode,
        "stratify_current": stratify_current,
        "seed": seed if split_mode == "random" or stratify_current else None,
        "widths": widths,
        "heights": heights,
        "train_by_current": counts_by_current(pairs, "train"),
        "val_by_current": counts_by_current(pairs, "val"),
    }
    (output_root / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")


def write_metadata_candidates(
    output_root: Path,
    source_root: Path,
    pairs: list[PairInfo],
    *,
    test_fraction: float,
    seed: int,
) -> None:
    test_names = [p.filename for p in pairs if p.split == "test"]
    cand_names = [p.filename for p in pairs if p.split == "candidate"]
    write_name_list(output_root / "test.txt", test_names)
    write_name_list(output_root / "candidates.txt", cand_names)
    write_manifest(output_root, pairs)
    write_split_record_candidates(
        output_root,
        source_root,
        pairs,
        test_fraction=test_fraction,
        seed=seed,
    )
    dataset = {
        "source_root": str(source_root),
        "pixel_transform": "none",
        "mode": "candidates",
        "pair_count": len(pairs),
        "test_pair_count": len(test_names),
        "candidate_pair_count": len(cand_names),
        "test_fraction": test_fraction,
        "seed": seed,
        "stratify_current": True,
        "test_by_current": counts_by_current(pairs, "test"),
        "candidates_by_current": counts_by_current(pairs, "candidate"),
        "widths": sorted({p.width for p in pairs}),
        "heights": sorted({p.height for p in pairs}),
    }
    (output_root / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    lq_files = list_tiffs(source_root / "LQ")
    gt_files = list_tiffs(source_root / "GT")
    all_names = validate_pair_names(lq_files, gt_files)

    if args.mode == "candidates":
        splits = assign_test_holdout(all_names, test_fraction=args.test_fraction, seed=args.seed)
        pairs = validate_pairs(lq_files, gt_files, splits)
        test_count = sum(p.split == "test" for p in pairs)
        cand_count = len(pairs) - test_count
        print(
            f"Validated {len(pairs)} pairs: test={test_count}, candidates={cand_count} "
            f"(stratified current, test_fraction={args.test_fraction})"
        )
        print(f"test_by_current={counts_by_current(pairs, 'test')}")
        print(f"candidates_by_current={counts_by_current(pairs, 'candidate')}")
        if args.dry_run:
            print("Dry run: no metadata written")
            return
        ensure_output_root(output_root, overwrite=args.overwrite, mode=args.mode)
        write_metadata_candidates(
            output_root,
            source_root,
            pairs,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )
        print(f"Wrote candidates metadata: {output_root}")
        return

    if args.mode == "finalize":
        candidates_path = (
            args.candidates_file.expanduser().resolve()
            if args.candidates_file is not None
            else output_root / "candidates.txt"
        )
        if not candidates_path.is_file():
            raise FileNotFoundError(f"Candidates list not found: {candidates_path}")
        candidate_names = read_name_list(candidates_path)
        missing = [n for n in candidate_names if n not in lq_files]
        if missing:
            raise ValueError(f"{len(missing)} candidates missing from LQ/GT. Examples: {missing[:10]}")
        test_path = output_root / "test.txt"
        test_names = read_name_list(test_path) if test_path.is_file() else []
        overlap = sorted(set(candidate_names) & set(test_names))
        if overlap:
            raise ValueError(f"candidates overlap test holdout ({len(overlap)}). Examples: {overlap[:10]}")

        # Finalize always stratifies by current (plan default for this dataset family).
        tv_splits = assign_splits(
            candidate_names,
            val_fraction=args.val_fraction,
            split_mode=args.split_mode,
            seed=args.seed,
            stratify_current=True,
        )
        pairs = validate_pairs(lq_files, gt_files, tv_splits)
        train_count = sum(p.split == "train" for p in pairs)
        val_count = len(pairs) - train_count
        print(
            f"Finalized from {candidates_path.name}: train={train_count}, val={val_count}, "
            f"test_holdout={len(test_names)} (untouched)"
        )
        print(f"train_by_current={counts_by_current(pairs, 'train')}")
        print(f"val_by_current={counts_by_current(pairs, 'val')}")
        if args.dry_run:
            print("Dry run: no metadata written")
            return
        ensure_output_root(output_root, overwrite=args.overwrite, mode=args.mode)
        write_metadata_train_val(
            output_root,
            source_root,
            pairs,
            split_mode=args.split_mode,
            seed=args.seed,
            raw_min=args.raw_min,
            raw_max=args.raw_max,
            stratify_current=True,
            test_names=test_names,
        )
        # Append finalize note to SPLIT_RECORD if present.
        record = output_root / "SPLIT_RECORD.md"
        note = (
            f"\n## finalize\n\n"
            f"- candidates_file: `{candidates_path}`\n"
            f"- train: {train_count}\n"
            f"- val: {val_count}\n"
            f"- val_fraction: {args.val_fraction}\n"
            f"- split_mode: {args.split_mode}\n"
            f"- stratify_current: true\n"
            f"- seed: {args.seed}\n"
            f"- train_by_current: {counts_by_current(pairs, 'train')}\n"
            f"- val_by_current: {counts_by_current(pairs, 'val')}\n"
        )
        if record.exists():
            record.write_text(record.read_text(encoding="utf-8") + note, encoding="utf-8")
        else:
            record.write_text("# SPLIT_RECORD\n" + note, encoding="utf-8")
        print(f"Wrote finalize metadata: {output_root}")
        return

    # Default train_val mode (backward compatible).
    splits = assign_splits(
        all_names,
        val_fraction=args.val_fraction,
        split_mode=args.split_mode,
        seed=args.seed,
        stratify_current=args.stratify_current,
    )
    pairs = validate_pairs(lq_files, gt_files, splits)
    train_count = sum(pair.split == "train" for pair in pairs)
    val_count = len(pairs) - train_count
    print(f"Validated {len(pairs)} aligned uint16 TIFF pairs: train={train_count}, val={val_count}")
    if args.stratify_current:
        print(f"train_by_current={counts_by_current(pairs, 'train')}")
        print(f"val_by_current={counts_by_current(pairs, 'val')}")
    print("Pixel transforms: none")
    if args.dry_run:
        print("Dry run: no metadata written")
        return

    ensure_output_root(output_root, overwrite=args.overwrite, mode=args.mode)
    write_metadata_train_val(
        output_root,
        source_root,
        pairs,
        split_mode=args.split_mode,
        seed=args.seed,
        raw_min=args.raw_min,
        raw_max=args.raw_max,
        stratify_current=args.stratify_current,
    )
    print(f"Wrote metadata: {output_root}")


if __name__ == "__main__":
    main()
