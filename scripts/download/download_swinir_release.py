#!/usr/bin/env python3
"""Download all assets from SwinIR GitHub release v0.0."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

RELEASE_API = "https://api.github.com/repos/JingyunLiang/SwinIR/releases/tags/v0.0"
DEFAULT_OUT = Path("experiments/pretrained_models/SwinIR")


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"SKIP {dest.name} (exists)")
        return
    print(f"GET  {dest.name}")
    urllib.request.urlretrieve(url, dest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Only download files whose names contain any of these substrings.",
    )
    args = parser.parse_args()

    with urllib.request.urlopen(RELEASE_API) as resp:
        release = json.load(resp)

    assets = release["assets"]
    if args.only:
        assets = [a for a in assets if any(key in a["name"] for key in args.only)]

    print(f"Downloading {len(assets)} assets to {args.out_dir}")
    for asset in assets:
        download_file(asset["browser_download_url"], args.out_dir / asset["name"])
    print("Done.")


if __name__ == "__main__":
    main()
