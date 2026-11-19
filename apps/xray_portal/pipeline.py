"""CPU-only intake validation, reviewed pairing and immutable training exports.

Original uploads are never changed. A candidate match is not an approval.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tarfile
import uuid
import warnings
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

MAX_EXPANDED_BYTES = 128 * 1024**3
MAX_EXPORT_BYTES = 128 * 1024**3
FREE_SPACE_MARGIN = 5 * 1024**3
MAX_FILES = 50_000
MAX_PIXELS = 100_000_000
COPY_CHUNK = 4 * 1024**2
TIFF_SUFFIXES = {".tif", ".tiff"}
SIDECAR_SUFFIXES = {".json", ".csv", ".tsv", ".txt", ".xml"}
SPLITS = {"train", "monitor", "test", "exclude"}


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _id(*parts: object) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()[:24]


def _inside(root: Path, value: str | Path) -> Path:
    root = root.resolve()
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root) or resolved == root:
        raise ValueError("File path must be beneath the batch directory")
    if candidate.is_symlink():
        raise ValueError("Symbolic links are not accepted as inputs")
    return resolved


def _json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _issue(code: str, message: str, file_id: str, severity: str = "error", **extra: object) -> dict:
    return {"severity": severity, "code": code, "message": message, "file_id": file_id, **extra}


def _axis(member: str, uploaded: dict) -> str:
    explicit = str(uploaded.get("axis", "")).strip().upper()
    found = {part.upper() for part in PurePosixPath(member).parts[:-1]} & {"XY", "XZ", "YZ"}
    if len(found) > 1 or (explicit and found and explicit not in found):
        raise ValueError("Conflicting slice directions; provide a consistent axis")
    return explicit or next(iter(found), "unspecified")


def _inspect(path: Path, member: str, uploaded: dict, batch: Path, previews: Path) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(path) as image:
            if image.format != "TIFF":
                raise ValueError("File is not a TIFF image")
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("Multi-page TIFF requires an explicitly reviewed conversion")
            width, height = image.size
            if width * height > MAX_PIXELS or width < 1 or height < 1:
                raise ValueError("Image dimensions exceed the supported limit")
            pixels = np.asarray(image)
            if pixels.ndim != 2 or pixels.dtype.kind != "u" or pixels.dtype.itemsize != 2:
                raise ValueError("Training intake supports only 2D uint16 grayscale TIFF")
            # Canonical little-endian hash catches identical pixels in different TIFF encodings.
            canonical = np.ascontiguousarray(pixels, dtype="<u2")
            pixel_sha = hashlib.sha256(memoryview(canonical)).hexdigest()
            image_id = _id(uploaded["id"], member)
            preview = previews / f"{image_id}.png"
            thumb = Image.fromarray(pixels)
            thumb.thumbnail((512, 512), resample=Image.Resampling.NEAREST)
            thumb8 = (np.asarray(thumb).astype(np.float32) / 257).clip(0, 255).astype(np.uint8)
            Image.fromarray(thumb8).save(preview)
            numbers = re.findall(r"\d+", Path(member).stem)
            slice_id = int(numbers[-1]) if numbers else Path(member).stem.casefold()
            return {
                "id": image_id,
                "file_id": str(uploaded["id"]),
                "relative_path": str(path.relative_to(batch)),
                "filename": member,
                "role": uploaded["role"],
                "condition": str(uploaded["condition"]),
                "scan_group": str(uploaded["scan_group"]),
                "workpiece": str(uploaded["workpiece"]),
                "axis": _axis(member, uploaded),
                "slice_id": slice_id,
                "width": width,
                "height": height,
                "dtype": "uint16",
                "sha256": _sha(path),
                "pixel_sha256": pixel_sha,
                "min": int(pixels.min()),
                "max": int(pixels.max()),
                "mean": float(pixels.mean()),
                "saturated_fraction": float(np.count_nonzero(pixels == 65535) / pixels.size),
                "preview": str(preview.relative_to(batch)),
                "preview_window": [0, 65535],
            }


def _members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    entries = archive.infolist()
    if len(entries) > MAX_FILES:
        raise ValueError("ZIP exceeds the file count limit")
    names: set[str] = set()
    total = 0
    accepted = []
    for entry in entries:
        name = entry.filename
        path = PurePosixPath(name)
        if (
            "\\" in name
            or "\x00" in name
            or path.is_absolute()
            or ".." in path.parts
            or not path.parts
            or ":" in path.parts[0]
        ):
            raise ValueError("Unsafe archive member path")
        folded = name.rstrip("/").casefold()
        if folded in names:
            raise ValueError("Duplicate or case-colliding archive member names")
        names.add(folded)
        mode = (entry.external_attr >> 16) & 0xFFFF
        kind = stat.S_IFMT(mode)
        if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise ValueError("ZIP links and special files are not allowed")
        if entry.flag_bits & 1:
            raise ValueError("Encrypted ZIP is not supported")
        if entry.is_dir():
            continue
        total += entry.file_size
        if total > MAX_EXPANDED_BYTES:
            raise ValueError("ZIP exceeds the expanded size limit")
        suffix = path.suffix.lower()
        if suffix not in TIFF_SUFFIXES | SIDECAR_SUFFIXES:
            if path.name == ".DS_Store" or path.parts[0] == "__MACOSX":
                continue
            raise ValueError("ZIP contains an unsupported file (nested archives and RAW are not supported)")
        accepted.append(entry)
    return accepted


def validate_batch(batch_dir: Path, files: list[dict]) -> dict:
    """Inspect uploaded per-condition TIFF/ZIP files without accepting any pair.

    ``path`` must identify an original beneath batch_dir. Invalid files generate
    issues; usable files from other conditions remain available for review.
    """
    batch = Path(batch_dir).resolve(strict=True)
    derived = batch / "derived"
    previews = derived / "previews"
    previews.mkdir(parents=True, exist_ok=True)
    images: list[dict] = []
    issues: list[dict] = []
    upload_records = []
    seen_ids: set[str] = set()
    expanded_total = 0
    file_count = 0
    for uploaded in files:
        file_id = str(uploaded.get("id", ""))
        try:
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", file_id) or file_id in seen_ids:
                raise ValueError("Upload identifiers must be unique and safe")
            seen_ids.add(file_id)
            if uploaded.get("role") not in {"LQ", "GT"}:
                raise ValueError("Select Noisy images or Reference images (GT)")
            for field in ("condition", "scan_group", "workpiece"):
                if not str(uploaded.get(field, "")).strip():
                    raise ValueError(f"Missing required acquisition field: {field}")
            original = _inside(batch, uploaded["path"])
            if not original.is_file():
                raise ValueError("Original upload is missing")
            filename = str(uploaded.get("filename", original.name))
            suffix = Path(filename).suffix.lower()
            upload_records.append(
                {
                    "id": file_id,
                    "filename": filename,
                    "sha256": _sha(original),
                    "size": original.stat().st_size,
                    "role": uploaded["role"],
                    "condition": uploaded["condition"],
                    "scan_group": uploaded["scan_group"],
                    "workpiece": uploaded["workpiece"],
                }
            )
            members: list[tuple[Path, str]] = []
            if suffix == ".zip":
                folder = derived / "files" / file_id
                stage = derived / "files" / f".{file_id}-{uuid.uuid4().hex}.partial"
                stage.mkdir(parents=True)
                try:
                    with zipfile.ZipFile(original) as archive:
                        entries = _members(archive)
                        expanded_total += sum(e.file_size for e in entries)
                        file_count += len(entries)
                        if expanded_total > MAX_EXPANDED_BYTES or file_count > MAX_FILES:
                            raise ValueError("Batch exceeds expanded bytes or file count limit")
                        actual_total = 0
                        for entry in entries:
                            target = stage / entry.filename
                            target.parent.mkdir(parents=True, exist_ok=True)
                            actual_size = 0
                            with archive.open(entry) as source, target.open("xb") as dest:
                                while chunk := source.read(COPY_CHUNK):
                                    actual_size += len(chunk)
                                    actual_total += len(chunk)
                                    if actual_size > entry.file_size or actual_total > MAX_EXPANDED_BYTES:
                                        raise ValueError("Archive expanded beyond its declared size")
                                    dest.write(chunk)
                                dest.flush()
                                os.fsync(dest.fileno())
                            if actual_size != entry.file_size:
                                raise ValueError("Archive member size mismatch")
                        if folder.exists():
                            # Revalidation is idempotent, but derived bytes must match originals.
                            for entry in entries:
                                existing = _inside(folder, entry.filename)
                                if not existing.is_file() or _sha(existing) != _sha(stage / entry.filename):
                                    raise ValueError("Existing derived files differ; create a new batch")
                        else:
                            stage.rename(folder)
                        members = [
                            (folder / e.filename, e.filename)
                            for e in entries
                            if Path(e.filename).suffix.lower() in TIFF_SUFFIXES
                        ]
                        sidecars = sum(Path(e.filename).suffix.lower() in SIDECAR_SUFFIXES for e in entries)
                        if sidecars:
                            issues.append(
                                _issue(
                                    "sidecars_not_interpreted",
                                    f"{sidecars} metadata files preserved; acquisition fields require review",
                                    file_id,
                                    "warning",
                                )
                            )
                finally:
                    if stage.exists():
                        shutil.rmtree(stage)
            elif suffix in TIFF_SUFFIXES:
                expanded_total += original.stat().st_size
                file_count += 1
                if expanded_total > MAX_EXPANDED_BYTES or file_count > MAX_FILES:
                    raise ValueError("Batch exceeds expanded bytes or file count limit")
                members = [(original, filename)]
            else:
                raise ValueError("Only TIFF and ZIP uploads are supported")
            if not members:
                raise ValueError("Upload contains no TIFF images")
            for path, member in members:
                try:
                    images.append(_inspect(path, member, uploaded, batch, previews))
                except Exception as exc:
                    issues.append(_issue("invalid_image", str(exc), file_id, filename=member))
        except Exception as exc:
            issues.append(_issue("invalid_upload", str(exc), file_id))
    by_pixels: dict[str, list[dict]] = {}
    for item in images:
        by_pixels.setdefault(item["pixel_sha256"], []).append(item)
    for duplicates in by_pixels.values():
        if len(duplicates) > 1:
            issues.append(
                _issue(
                    "duplicate_pixels",
                    "Identical image pixels found in multiple uploaded images",
                    duplicates[0]["file_id"],
                    "warning",
                    image_ids=[i["id"] for i in duplicates],
                )
            )
    gt_by_key: dict[tuple, list[dict]] = {}
    for item in images:
        if item["role"] == "GT":
            key = (item["workpiece"], item["scan_group"], item["axis"], item["slice_id"])
            gt_by_key.setdefault(key, []).append(item)
    pairs = []
    for lq in images:
        if lq["role"] != "LQ":
            continue
        key = (lq["workpiece"], lq["scan_group"], lq["axis"], lq["slice_id"])
        choices = gt_by_key.get(key, [])
        status = "candidate" if len(choices) == 1 else "missing_gt" if not choices else "ambiguous_gt"
        gt = choices[0] if len(choices) == 1 else None
        if gt and (lq["width"], lq["height"]) != (gt["width"], gt["height"]):
            status = "size_mismatch"
        pairs.append(
            {
                "id": _id("pair", lq["id"]),
                "lq_id": lq["id"],
                "gt_id": gt["id"] if gt else None,
                "gt_candidates": [g["id"] for g in choices],
                "group": lq["workpiece"],
                "scan_group": lq["scan_group"],
                "axis": lq["axis"],
                "slice_id": lq["slice_id"],
                "status": status,
            }
        )
    if not images:
        issues.append(_issue("empty_batch", "No usable TIFF images found", ""))
    if not pairs:
        issues.append(_issue("no_lq", "No usable noisy images available for pairing", ""))
    return {
        "schema_version": 1,
        "files": upload_records,
        "images": images,
        "issues": issues,
        "candidate_pairs": pairs,
        "counts": {"uploads": len(files), "images": len(images), "pairs": len(pairs), "expanded_bytes": expanded_total},
    }


def _check_pair(lq: dict, gt: dict) -> None:
    if lq["role"] != "LQ" or gt["role"] != "GT":
        raise ValueError("A pair requires one noisy image and one reference image")
    for key in ("workpiece", "scan_group", "axis", "slice_id", "width", "height", "dtype"):
        if lq[key] != gt[key]:
            raise ValueError(f"Pair has mismatched {key}; correct acquisition metadata and revalidate")


def export_dataset(batch_dir: Path, report: dict, decisions: dict, output_dir: Path) -> dict:
    """Freeze explicitly reviewed pairs. No cloud calls and no training.

    A decision may specify gt_id to resolve an ambiguous automatic candidate.
    Every candidate needs a decision, including explicit exclusions.
    """
    batch = Path(batch_dir).resolve(strict=True)
    output = Path(output_dir).absolute()
    if output.exists():
        raise ValueError("Dataset version already exists; choose a new version")
    if not str(decisions.get("reviewed_by", "")).strip():
        raise ValueError("Reviewer identity is required")
    if not str(decisions.get("rationale", "")).strip():
        raise ValueError("Review notes are required")
    if report.get("issues") and decisions.get("acknowledge_issues") is not True:
        raise ValueError("Acknowledge validation issues and explicitly exclude unusable inputs")
    mode = decisions.get("split_mode", "group")
    if mode not in {"group", "spatial"}:
        raise ValueError("Unknown split mode")
    guard = decisions.get("guard_slices", 16)
    if isinstance(guard, bool) or not isinstance(guard, int) or guard < 1:
        raise ValueError("Spatial guard band must be a positive integer")
    images = {image["id"]: image for image in report["images"]}
    candidates = {pair["id"]: pair for pair in report["candidate_pairs"]}
    rows = decisions.get("pairs", [])
    if len({row["id"] for row in rows}) != len(rows) or {row["id"] for row in rows} != set(candidates):
        raise ValueError("Every candidate needs exactly one explicit decision")
    selected = []
    excluded = []
    used_hashes: dict[str, str] = {}
    group_splits: dict[str, str] = {}
    spatial: dict[tuple, list[tuple[int, str]]] = {}
    for row in rows:
        split = row.get("split")
        if split not in SPLITS:
            raise ValueError("Unknown split assignment")
        if split == "exclude":
            excluded.append(row["id"])
            continue
        if row.get("approved") is not True:
            raise ValueError("Each included pair must be explicitly approved")
        candidate = candidates[row["id"]]
        lq = images[candidate["lq_id"]]
        gt_id = row.get("gt_id") or candidate.get("gt_id")
        if gt_id not in images:
            raise ValueError("Resolve missing or ambiguous GT before including a pair")
        gt = images[gt_id]
        _check_pair(lq, gt)
        if mode == "group":
            group = lq["workpiece"]
            if group in group_splits and group_splits[group] != split:
                raise ValueError("The same workpiece cannot occur in different splits")
            group_splits[group] = split
        else:
            if not isinstance(lq["slice_id"], int):
                raise ValueError("Spatial splitting requires numeric slice indices")
            key = (lq["workpiece"], lq["axis"])
            spatial.setdefault(key, []).append((lq["slice_id"], split))
        for item in (lq, gt):
            for digest in (item["sha256"], item["pixel_sha256"]):
                if digest in used_hashes and used_hashes[digest] != split:
                    raise ValueError("Identical source or pixel content crosses dataset splits")
                used_hashes[digest] = split
        selected.append({"id": row["id"], "filename": f"pair_{row['id']}.tif", "split": split, "lq": lq, "gt": gt})
    if not any(r["split"] == "train" for r in selected) or not any(r["split"] == "monitor" for r in selected):
        raise ValueError("A training version requires both train and monitor pairs")
    if mode == "spatial":
        axes_by_workpiece: dict[str, set[str]] = {}
        for row in selected:
            axes_by_workpiece.setdefault(row["lq"]["workpiece"], set()).add(row["lq"]["axis"])
        if any(len(axes) > 1 for axes in axes_by_workpiece.values()):
            raise ValueError(
                "Spatial splitting supports only one axis per workpiece; "
                "cross-axis slices can share voxels, so use workpiece groups or review a single direction"
            )
        for sequence in spatial.values():
            # Check nearest boundary across all conditions and repeat scans.
            ordered = sorted(set(sequence))
            for (left, left_split), (right, right_split) in zip(ordered, ordered[1:], strict=False):
                if left_split != right_split and right - left <= guard:
                    raise ValueError("Spatial split violates the reviewed slice guard band")
    # A reference reused by many conditions is replicated in the flat training
    # contract. Bound the export itself, not only the incoming/extracted bytes.
    source_sizes = {}
    for row in selected:
        for item in (row["lq"], row["gt"]):
            if item["id"] not in source_sizes:
                source_sizes[item["id"]] = _inside(batch, item["relative_path"]).stat().st_size
    logical_bytes = sum(source_sizes[item["id"]] for row in selected for item in (row["lq"], row["gt"]))
    if logical_bytes > MAX_EXPORT_BYTES:
        raise ValueError("Frozen dataset exceeds the 128 GiB logical export limit; review a smaller version")
    archived_rows = [row for row in selected if row["split"] != "test"]
    archive_input_bytes = sum(source_sizes[item["id"]] for row in archived_rows for item in (row["lq"], row["gt"]))
    # Tar headers, padding/PAX records and text metadata are budgeted above their
    # normal sizes; the free-space margin also covers manifests and temporary I/O.
    archive_estimated_bytes = archive_input_bytes + (len(archived_rows) * 2 + 3) * 4096 + 10240
    required_bytes = logical_bytes + archive_estimated_bytes + FREE_SPACE_MARGIN
    capacity_root = output.parent
    while not capacity_root.exists():
        capacity_root = capacity_root.parent
    if shutil.disk_usage(capacity_root).free < required_bytes:
        raise ValueError(
            "Insufficient space for frozen image copies and training archive; free storage before retrying"
        )
    for item in {i["id"]: i for row in selected for i in (row["lq"], row["gt"])}.values():
        source = _inside(batch, item["relative_path"])
        if _sha(source) != item["sha256"]:
            raise ValueError("Input changed after review; revalidate before freezing")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f".{output.name}-{uuid.uuid4().hex}.partial"
    stage.mkdir()
    try:
        for role in ("GT", "LQ"):
            (stage / role).mkdir()
        for row in selected:
            for role in ("LQ", "GT"):
                source = _inside(batch, row[role.lower()]["relative_path"])
                target = stage / role / row["filename"]
                # Independent copies ensure frozen exports cannot mutate originals.
                with source.open("rb") as src, target.open("xb") as dst:
                    shutil.copyfileobj(src, dst, COPY_CHUNK)
                    dst.flush()
                    os.fsync(dst.fileno())
                if _sha(target) != row[role.lower()]["sha256"]:
                    raise ValueError("Source changed while creating the frozen version")
        for split, name in (("train", "train.txt"), ("monitor", "val_monitor.txt"), ("test", "test.txt")):
            (stage / name).write_text("".join(row["filename"] + "\n" for row in selected if row["split"] == split))
        _json(stage / "normalization.json", {"raw_min": 0.0, "raw_max": 65535.0})
        manifest = {
            "schema_version": 1,
            "reviewed_by": decisions["reviewed_by"],
            "review_notes": decisions["rationale"],
            "split_mode": mode,
            "guard_slices": guard if mode == "spatial" else None,
            "independent_workpiece_test": mode == "group",
            "counts": {split: sum(row["split"] == split for row in selected) for split in ("train", "monitor", "test")},
            "source_uploads": report.get("files", []),
            "pairs": selected,
            "excluded_pair_ids": excluded,
            "acknowledged_issues": report.get("issues", []),
            "test_in_archive": False,
            "logical_bytes": logical_bytes,
            "archive_estimated_bytes": archive_estimated_bytes,
        }
        _json(stage / "pairs_manifest.json", manifest)
        with (
            (stage / "dataset.tar").open("xb") as stream,
            tarfile.open(fileobj=stream, mode="w", dereference=True) as archive,
        ):
            for name in ("train.txt", "val_monitor.txt", "normalization.json"):
                archive.add(stage / name, arcname=name, recursive=False)
            for row in selected:
                if row["split"] == "test":
                    continue
                for role in ("GT", "LQ"):
                    name = f"{role}/{row['filename']}"
                    archive.add(stage / name, arcname=name, recursive=False)
        digest = _sha(stage / "dataset.tar")
        (stage / "dataset.tar.sha256").write_text(digest + "\n")
        manifest.update(archive="dataset.tar", sha256=digest, archive_bytes=(stage / "dataset.tar").stat().st_size)
        _json(stage / "manifest.json", manifest)
        for path in stage.rglob("*"):
            if path.is_file():
                path.chmod(0o440)
        # Server serializes exports; os.rename cannot replace a populated version.
        stage.rename(output)
        return manifest
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
