#!/usr/bin/env python3
"""Copy text metadata from a CI experiments root into job artifacts."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

TEXT_SUFFIXES = {".json", ".log", ".yml", ".yaml"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    target = args.target.expanduser().resolve()
    if not source.is_dir():
        raise SystemExit(f"Missing experiments root: {source}")
    target.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in source.rglob("*"):
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            destination = target / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            copied += 1
    print(f"Staged {copied} text files from {source} -> {target}; checkpoints stay on NFS.")


if __name__ == "__main__":
    main()
