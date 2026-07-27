#!/usr/bin/env python3
"""Fill eval_benchmark_v2/comparison_table.md from metrics/*.csv (4 model rows)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

COLUMNS = [
    "battery_260520",
    "battery_260527",
    "cylinder_260520_val20",
    "cylinder_260527_holdout20",
    "phantom_aligned_drive_val20",
]

MODELS = [
    ("old_cylinder_archive", "旧 Cylinder FT (`archive/cylinderblock_2673001900`)"),
    ("phantom_aligned_drive", "Phantom / aligned_drive FT (`…/P128W8/net_g_latest.pth`)"),
    ("new_cylinder_gt90ua", "新 Cylinder90 FT (`cylinder_260520_gt90ua`)"),
    ("cylinder_phantom_mix", "Cylinder+Phantom mix FT (`cylinder_phantom_mix`)"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        default=Path("/mnt/home/huang/cst_ai/outputs/eval_benchmark_v2"),
    )
    return p.parse_args()


def mean_metrics(path: Path) -> str:
    if not path.is_file():
        return "TBD"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        return "TBD"
    ft = sum(float(r["ft_snr"]) for r in rows) / len(rows)
    if "ft_minus_preft_snr" in rows[0]:
        delta = sum(float(r["ft_minus_preft_snr"]) for r in rows) / len(rows)
    elif "preft_snr" in rows[0]:
        delta = ft - sum(float(r["preft_snr"]) for r in rows) / len(rows)
    else:
        delta = float("nan")
    psnr = sum(float(r["ft_psnr"]) for r in rows) / len(rows)
    ssim = sum(float(r["ft_ssim"]) for r in rows) / len(rows)
    return f"{ft:.2f} (Δ{delta:+.2f}) / PSNR {psnr:.1f} / SSIM {ssim:.3f}"


def main() -> None:
    args = parse_args()
    root = args.root
    metrics = root / "metrics"
    lines = [
        "# モデル × 評価列 比較表（SNR 平均）— eval_benchmark_v2",
        "",
        "単位: dB。セル形式: `FT_SNR (Δ vs PreFT) / PSNR / SSIM`。",
        "",
        (
            "| モデル | battery 5/20 (110kV, n=9) | battery 5/27 (150kV, n=6) | "
            "cylinder 5/20 val20 (GT=90µA) | cylinder 5/27 holdout20 (GT=60µA) | "
            "phantom aligned_drive val20 |"
        ),
        (
            "|--------|--------------------------:|--------------------------:|"
            "------------------------------:|----------------------------------:|"
            "--------------------------------------:|"
        ),
    ]
    for tag, label in MODELS:
        cells = [label]
        for col in COLUMNS:
            cells.append(mean_metrics(metrics / f"{tag}__{col}.csv"))
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "### 注",
            (
                "- *1 旧 Cylinder は学習時 GT=300kV＋幾何変換。"
                "本ベンチ cylinder 列は 90/60µA GT・無変換のため旧モデル不利。"
            ),
            "- *2 cylinder 5/27 と battery はいずれの学習にも未使用（holdout / 未学習ドメイン）。",
            (
                "- *3 cylinder 5/20 は漏洩回避のため cyl90/mix train 除外の val プールから抽出。"
                "多電流が取れない場合は単一 µA のまま（強めない）。"
            ),
            "- *4 cylinder 5/27 holdout は µA 層化サンプリング。",
            "- *5 mix 行は `cylinder_phantom_mix`（cyl__ + ph__ hardlink、domain×µA 層化 split）。",
            "",
        ]
    )
    out = root / "comparison_table.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
