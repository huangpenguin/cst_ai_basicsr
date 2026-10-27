#!/usr/bin/env python3
"""Organize nested scan folders into flat same-name LQ/GT TIFF pairs.

Supports multiple GT folders. Each LQ folder is paired with the GT folder that
shares the same parent group directory (e.g. both under ``5-1/``).

Output pair ids keep the LQ folder prefix so all groups can live in one flat
``LQ/`` + ``GT/`` tree without name collisions.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

TIFF_SUFFIXES = {".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--roles-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--link-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--check-sample", type=int, default=5)
    return parser.parse_args()


def load_roles(path: Path) -> dict[str, str]:
    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    folders = data.get("folders") or {}
    if not isinstance(folders, dict) or not folders:
        raise ValueError(f"No folders mapping in {path}")
    roles: dict[str, str] = {}
    for name, spec in folders.items():
        role = spec.get("role") if isinstance(spec, dict) else spec
        role = str(role).strip().lower()
        if role not in {"gt", "lq", "skip"}:
            raise ValueError(f"Invalid role for {name!r}: {role!r}")
        roles[str(name)] = role
    return roles


def load_slice_ranges(path: Path) -> dict[str, tuple[int, int]]:
    """Load inclusive Z ranges keyed by group directory name (e.g. ``5-1``)."""
    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw = data.get("slice_ranges") or {}
    if not raw:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"slice_ranges must be a mapping in {path}")
    ranges: dict[str, tuple[int, int]] = {}
    for group, spec in raw.items():
        if not isinstance(spec, dict):
            raise ValueError(f"slice_ranges[{group!r}] must be {{start, end}}")
        try:
            start = int(spec["start"])
            end = int(spec["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid slice_ranges[{group!r}]: {spec!r}") from exc
        if end < start:
            raise ValueError(f"slice_ranges[{group!r}]: end ({end}) < start ({start})")
        ranges[str(group)] = (start, end)
    return ranges


def slice_index_from_stem(stem: str) -> int:
    """Parse Z index from ``data_0275`` / ``0275`` style stems."""
    key = stem.removeprefix("data_") if stem.startswith("data_") else stem
    try:
        return int(key)
    except ValueError as exc:
        raise ValueError(f"Cannot parse slice index from stem {stem!r}") from exc


def find_named_dir(source_root: Path, folder_name: str) -> Path:
    matches = [path for path in source_root.rglob(folder_name) if path.is_dir() and path.name == folder_name]
    if not matches:
        raise FileNotFoundError(f"Folder not found under {source_root}: {folder_name}")
    if len(matches) > 1:
        raise ValueError(f"Folder name not unique ({len(matches)}): {folder_name} -> {matches}")
    return matches[0]


def list_tiffs(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in TIFF_SUFFIXES)


def pair_stem(src: Path) -> str:
    stem = src.stem
    if stem.startswith("data_"):
        return stem.removeprefix("data_")
    return stem


def place_file(src: Path, dst: Path, *, link_mode: str, dry_run: bool) -> None:
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if link_mode == "hardlink":
        try:
            dst.hardlink_to(src)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def inspect_tiff(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        arr = np.asarray(image)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return {
        "path": str(path),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": int(arr.min()) if arr.size else None,
        "max": int(arr.max()) if arr.size else None,
    }


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    roles_path = args.roles_file.expanduser().resolve()
    roles = load_roles(roles_path)
    slice_ranges = load_slice_ranges(roles_path)

    resolved: dict[str, Path] = {}
    for folder_name, role in roles.items():
        if role == "skip":
            continue
        resolved[folder_name] = find_named_dir(source_root, folder_name)

    gt_folders = [name for name, role in roles.items() if role == "gt"]
    lq_folders = [name for name, role in roles.items() if role == "lq"]
    if not gt_folders:
        raise ValueError("At least one GT folder is required")
    if not lq_folders:
        raise ValueError("At least one LQ folder is required")

    # Group by parent directory (5-1, 5-2, ...). Each group needs exactly one GT.
    gt_by_parent: dict[Path, tuple[str, Path, dict[str, Path]]] = {}
    for gt_name in gt_folders:
        gt_path = resolved[gt_name]
        parent = gt_path.parent
        if parent in gt_by_parent:
            raise ValueError(f"Multiple GT folders under {parent}: {gt_by_parent[parent][0]} and {gt_name}")
        stems = {path.stem: path for path in list_tiffs(gt_path)}
        if not stems:
            raise ValueError(f"No TIFF in GT folder: {gt_path}")
        gt_by_parent[parent] = (gt_name, gt_path, stems)

    if slice_ranges:
        missing_groups = sorted({parent.name for parent in gt_by_parent} - set(slice_ranges))
        if missing_groups:
            raise ValueError(f"slice_ranges missing groups used by GT folders: {missing_groups}")

    pairs: list[tuple[str, Path, Path, str]] = []
    unmatched: list[str] = []
    skipped_out_of_range = 0
    for lq_folder_name in sorted(lq_folders):
        lq_path = resolved[lq_folder_name]
        parent = lq_path.parent
        if parent not in gt_by_parent:
            raise ValueError(
                f"LQ folder {lq_folder_name} under {parent} has no GT sibling. "
                f"GT parents: {sorted(str(p) for p in gt_by_parent)}"
            )
        gt_name, _, gt_by_stem = gt_by_parent[parent]
        group_range = slice_ranges.get(parent.name)
        for lq_src in list_tiffs(lq_path):
            if group_range is not None:
                z_idx = slice_index_from_stem(lq_src.stem)
                start, end = group_range
                if z_idx < start or z_idx > end:
                    skipped_out_of_range += 1
                    continue
            gt_src = gt_by_stem.get(lq_src.stem)
            if gt_src is None:
                unmatched.append(f"{lq_folder_name}/{lq_src.name} (gt={gt_name})")
                continue
            pair_id = f"{lq_folder_name}_{pair_stem(lq_src)}.tif"
            pairs.append((pair_id, lq_src, gt_src, gt_name))

    if unmatched:
        raise ValueError(
            f"{len(unmatched)} LQ files have no GT with the same original stem. Examples: {unmatched[:20]}"
        )
    if not pairs:
        raise ValueError("No pairs produced")

    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"Output root not empty; pass --overwrite: {output_root}")

    if not args.dry_run:
        if args.overwrite and output_root.exists():
            for sub in ("LQ", "GT"):
                target = output_root / sub
                if target.exists():
                    shutil.rmtree(target)
        (output_root / "LQ").mkdir(parents=True, exist_ok=True)
        (output_root / "GT").mkdir(parents=True, exist_ok=True)

    for pair_id, lq_src, gt_src, _gt_name in pairs:
        place_file(lq_src, output_root / "LQ" / pair_id, link_mode=args.link_mode, dry_run=args.dry_run)
        place_file(gt_src, output_root / "GT" / pair_id, link_mode=args.link_mode, dry_run=args.dry_run)

    problems: list[str] = []
    sample_checks: list[dict[str, Any]] = []
    shapes: set[tuple[int, ...]] = set()
    dtypes: set[str] = set()
    for pair_id, lq_src, gt_src, gt_name in pairs[: max(args.check_sample, 1)]:
        lq_info = inspect_tiff(lq_src)
        gt_info = inspect_tiff(gt_src)
        sample_checks.append({"pair_id": pair_id, "gt_folder": gt_name, "lq": lq_info, "gt": gt_info})
        shapes.add(tuple(lq_info["shape"]))
        shapes.add(tuple(gt_info["shape"]))
        dtypes.add(str(lq_info["dtype"]))
        dtypes.add(str(gt_info["dtype"]))
        if lq_info["dtype"] != "uint16" or gt_info["dtype"] != "uint16":
            problems.append(f"{pair_id}: non-uint16 dtype lq={lq_info['dtype']} gt={gt_info['dtype']}")
        if lq_info["shape"] != gt_info["shape"]:
            problems.append(f"{pair_id}: shape mismatch lq={lq_info['shape']} gt={gt_info['shape']}")

    for pair_id, lq_src, gt_src, _gt_name in pairs[:: max(1, len(pairs) // 50)]:
        with Image.open(lq_src) as im_lq, Image.open(gt_src) as im_gt:
            if im_lq.size != im_gt.size:
                problems.append(f"{pair_id}: PIL size mismatch {im_lq.size} vs {im_gt.size}")

    report = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "gt_folders": gt_folders,
        "lq_folders": lq_folders,
        "slice_ranges": {k: {"start": v[0], "end": v[1]} for k, v in sorted(slice_ranges.items())},
        "skipped_out_of_range": skipped_out_of_range,
        "pair_count": len(pairs),
        "example_pair_ids": [pair_id for pair_id, *_ in pairs[:5]],
        "sample_checks": sample_checks,
        "observed_shapes_in_samples": [list(shape) for shape in sorted(shapes)],
        "observed_dtypes_in_samples": sorted(dtypes),
        "problems": problems,
        "dry_run": args.dry_run,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if problems:
        raise SystemExit("Data quality problems detected.")
    print("OK: pairs ready" if not args.dry_run else "OK: dry-run passed")


if __name__ == "__main__":
    main()
