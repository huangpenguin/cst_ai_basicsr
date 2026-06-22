#!/usr/bin/env python3
"""Smoke test SwinIR gray_dn pretrained models on 1_battery TIFF data."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from data_paths import default_battery_tiff_dir, resolve_output_dir
from smoke_test_swinir_gray_dn import main

if __name__ == "__main__":
    battery_dir = default_battery_tiff_dir()
    output_dir = resolve_output_dir("swinir_gray_dn_battery_smoke")
    sys.argv = [
        sys.argv[0],
        "--input-dir",
        str(battery_dir),
        "--output-dir",
        str(output_dir),
        "--devices",
        "0",
        *sys.argv[1:],
    ]
    main()
