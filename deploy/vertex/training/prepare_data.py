"""Validate unchanged paired TIFF splits and optionally package a training archive.

No cloud calls; no training. Output archives contain train/monitor data only.
"""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def inspect(root):
    splits = {}
    for split in ("train.txt", "val_monitor.txt", "test.txt"):
        path = root / split
        names = [line.split()[0] for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
        if split != "test.txt" and not names:
            raise ValueError(f"Missing or empty {split}")
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate entries in {split}")
        for name in names:
            if Path(name).name != name or name in (".", ".."):
                raise ValueError("Only flat, safe paired filenames are supported")
        splits[split] = names
    for a, b in [("train.txt", "val_monitor.txt"), ("train.txt", "test.txt"), ("val_monitor.txt", "test.txt")]:
        if set(splits[a]) & set(splits[b]):
            raise ValueError(f"Overlapping splits: {a}, {b}")
    norm = json.loads((root / "normalization.json").read_text())
    if float(norm.get("raw_min", 0)) != 0 or float(norm.get("raw_max", 65535)) != 65535:
        raise ValueError("Expected the original uint16 0..65535 normalization")
    files = [root / s for s in ("train.txt", "val_monitor.txt", "normalization.json")]
    for name in splits["train.txt"] + splits["val_monitor.txt"]:
        for folder in ("GT", "LQ"):
            p = root / folder / name
            if not p.is_file():
                raise ValueError(f"Missing {p}")
            files.append(p)
    return splits, files


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--archive", type=Path, help="Create a new tar archive and SHA256 sidecar; potentially large")
    args = ap.parse_args()
    root = args.data_root.resolve()
    splits, files = inspect(root)
    summary = {
        "splits": {k: len(v) for k, v in splits.items()},
        "file_count": len(files),
        "logical_bytes": sum(p.stat().st_size for p in files),
        "test_in_archive": False,
    }
    if args.archive:
        # Exclusive output: never replace an earlier dataset package.
        with args.archive.open("xb") as out, tarfile.open(fileobj=out, mode="w") as archive:
            for p in files:
                archive.add(p, arcname=str(p.relative_to(root)), recursive=False)
        with args.archive.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        summary.update(archive=str(args.archive), sha256=digest)
        args.archive.with_suffix(args.archive.suffix + ".sha256").write_text(digest + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
