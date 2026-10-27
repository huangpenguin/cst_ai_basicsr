#!/usr/bin/env python3
"""Download TSFDAN pretrained SwinTransformer_288800iter.ckpt."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

CKPT_URL = "https://raw.githubusercontent.com/Wasserdawn/TSFDAN/main/SwinTransformer_288800iter.ckpt"
DEFAULT_OUT = Path("experiments/pretrained_models/TSFDAN/SwinTransformer_288800iter.ckpt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists")
    args = parser.parse_args()

    dest: Path = args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not args.force:
        print(f"SKIP {dest} (exists, {dest.stat().st_size} bytes)")
        return
    print(f"GET  {CKPT_URL}")
    urllib.request.urlretrieve(CKPT_URL, dest)
    print(f"Wrote {dest} ({dest.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
