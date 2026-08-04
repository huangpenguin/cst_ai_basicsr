#!/usr/bin/env python3
"""基于已有 eval_benchmark panels 重算 EPI 等指标并重画带大字体标注的对比图。

不重跑 SwinIR 推理：
  - LQ/GT 从 bench pairs TIFF 读取
  - PreFT/FT 从已有 ``*_preft.png`` / ``*_ft.png`` 读回（当初按 [0,1]→uint8 保存）

输出：
  - 覆盖 ``panels/<tag>/<col>/*_compare.png``（含残差列）
  - 更新 ``metrics/<tag>__<col>.csv`` 与 panel 内 ``metrics.csv``
  - 写 ``comparison_table.md`` / 刷新 README 指标说明可由调用方另做
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from basicsr.metrics.epi import calculate_epi  # noqa: E402
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim  # noqa: E402
from basicsr.metrics.snr import calculate_snr  # noqa: E402

COLUMNS = [
    "battery_260520",
    "battery_260527",
    "cylinder_260520_val20",
    "cylinder_260527_holdout20",
    "phantom_aligned_drive_val20",
]

MODELS = [
    ("old_cylinder_archive", "旧 Cylinder FT (`archive/cylinderblock_2673001900`)"),
    ("phantom_aligned_drive", "Phantom / aligned_drive FT"),
    ("new_cylinder_gt90ua", "新 Cylinder90 FT"),
    ("cylinder_phantom_mix", "Cylinder+Phantom mix FT"),
]

HEADER_H = 88
FOOTER_H = 150


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bench-root",
        type=Path,
        default=Path("/mnt/home/huang/cst_ai/outputs/eval_benchmark_v2"),
    )
    p.add_argument(
        "--mode",
        choices=("enriched", "classic"),
        default="enriched",
        help=(
            "enriched=EPI+residual large fonts; classic=LQ|PreFT|FT|GT panels "
            "to --delivery-root (does not overwrite bench enriched compares)"
        ),
    )
    p.add_argument(
        "--delivery-root",
        type=Path,
        default=None,
        help="classic 模式必填：输出交付目录（panels/metrics/table），不覆盖 bench 工作树",
    )
    p.add_argument("--columns", nargs="*", default=None)
    p.add_argument("--tags", nargs="*", default=None, help="Model tags; default all four")
    p.add_argument("--max-per-column", type=int, default=0)
    return p.parse_args()


def save_classic_panel(
    columns: list[tuple[str, np.ndarray]],
    path: Path,
    title: str,
    *,
    display_lo: float,
    display_hi: float,
) -> None:
    """Original infer-style grayscale strip: LQ | PreFT | FT | GT."""
    cols_u8 = [to_u8(img, display_lo, display_hi) for _, img in columns]
    labels = [lab for lab, _ in columns]
    h, w = cols_u8[0].shape
    canvas = Image.new("L", (w * len(cols_u8), h + 28), 0)
    canvas.paste(Image.new("L", (w * len(cols_u8), 28), 40), (0, 0))
    draw = ImageDraw.Draw(canvas)
    for i, (col, lab) in enumerate(zip(cols_u8, labels, strict=True)):
        canvas.paste(Image.fromarray(col, mode="L"), (i * w, 28))
        draw.text((i * w + 8, 6), f"{title} | {lab}", fill=220)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def write_classic_comparison_table(out_root: Path, tags: list[str]) -> None:
    label_map = {t: lab for t, lab in MODELS}
    lines = [
        "# モデル × 評価列 比較表（SNR 平均）— eval_benchmark_v2",
        "",
        "単位: dB。セル形式: `FT_SNR (Δ vs PreFT) / PSNR / SSIM`。",
        "（本包は EPI 追加前のクラシック版；残差列なし。）",
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
    for tag in tags:
        cells = [label_map.get(tag, tag)]
        for col in COLUMNS:
            path = out_root / "metrics" / f"{tag}__{col}.csv"
            if not path.is_file():
                cells.append("TBD")
                continue
            rows = list(csv.DictReader(path.open(encoding="utf-8")))
            snr = mean_or_nan(rows, "ft_snr")
            dsnr = mean_or_nan(rows, "ft_minus_preft_snr")
            psnr = mean_or_nan(rows, "ft_psnr")
            ssim = mean_or_nan(rows, "ft_ssim")
            if None in (snr, dsnr, psnr, ssim):
                cells.append("TBD")
            else:
                cells.append(f"{snr:.2f} (Δ{dsnr:+.2f}) / {psnr:.1f} / {ssim:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "### 注",
            "- PreFT = 公式 SwinIR（微調整前）。FT = 各データ微調整後。",
            "- 訓練損失は MSE（L2）。",
            "",
        ]
    )
    (out_root / "comparison_table.md").write_text("\n".join(lines), encoding="utf-8")


def process_column_classic(bench: Path, delivery: Path, tag: str, col: str, names: list[str]) -> None:
    """Rebuild pre-EPI delivery panels into delivery/; do not modify bench compares."""
    pair = bench / "pairs" / col
    raw_min, raw_max = load_norm(pair / "normalization.json")
    src_panel = bench / "panels" / tag / col
    dst_panel = delivery / "panels" / tag / col
    dst_panel.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for name in names:
        stem = Path(name).stem
        ft_path = src_panel / f"{stem}_ft.png"
        pre_path = src_panel / f"{stem}_preft.png"
        if not ft_path.is_file() or not pre_path.is_file():
            raise FileNotFoundError(f"Missing FT/PreFT PNG for {tag}/{col}/{stem}")

        lq = normalize_u16(read_u16_gray(pair / "LQ" / name), raw_min, raw_max)
        gt = normalize_u16(read_u16_gray(pair / "GT" / name), raw_min, raw_max)
        ft = read_float01_png(ft_path)
        pre = read_float01_png(pre_path)

        # Classic metrics: SNR/PSNR/SSIM only (same scale as original infer).
        def snr_psnr_ssim(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
            pred_hwc = pred[..., None] * 255.0
            ref_hwc = ref[..., None] * 255.0
            return {
                "snr": float(calculate_snr(pred_hwc, ref_hwc, crop_border=0)),
                "psnr": float(calculate_psnr(pred_hwc, ref_hwc, crop_border=0)),
                "ssim": float(calculate_ssim(pred_hwc, ref_hwc, crop_border=0)),
            }

        lq_m = snr_psnr_ssim(lq, gt)
        pre_m = snr_psnr_ssim(pre, gt)
        ft_m = snr_psnr_ssim(ft, gt)
        lo = float(min(lq.min(), gt.min()))
        hi = float(max(lq.max(), gt.max()))
        save_classic_panel(
            [("LQ", lq), ("PreFT", pre), ("FT", ft), ("GT", gt)],
            dst_panel / f"{stem}_compare.png",
            stem,
            display_lo=lo,
            display_hi=hi,
        )
        row = {
            "filename": name,
            "lq_snr": f"{lq_m['snr']:.6f}",
            "ft_snr": f"{ft_m['snr']:.6f}",
            "ft_snr_gain": f"{ft_m['snr'] - lq_m['snr']:.6f}",
            "lq_psnr": f"{lq_m['psnr']:.6f}",
            "ft_psnr": f"{ft_m['psnr']:.6f}",
            "lq_ssim": f"{lq_m['ssim']:.6f}",
            "ft_ssim": f"{ft_m['ssim']:.6f}",
            "preft_snr": f"{pre_m['snr']:.6f}",
            "preft_psnr": f"{pre_m['psnr']:.6f}",
            "preft_ssim": f"{pre_m['ssim']:.6f}",
            "ft_minus_preft_snr": f"{ft_m['snr'] - pre_m['snr']:.6f}",
        }
        rows.append(row)
        print(f"  classic {tag}/{col}/{stem}  SNR FT={ft_m['snr']:.2f}", flush=True)

    fieldnames = list(rows[0].keys())
    for dest in (dst_panel / "metrics.csv", delivery / "metrics" / f"{tag}__{col}.csv"):
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
        Path("/usr/share/fonts/truetype/freefont/FreeSans.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def read_u16_gray(path: Path) -> np.ndarray:
    img = np.array(Image.open(path))
    if img.ndim == 3:
        img = img[..., 0]
    return img


def load_norm(path: Path) -> tuple[float, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data["raw_min"]), float(data["raw_max"])


def normalize_u16(arr: np.ndarray, raw_min: float, raw_max: float) -> np.ndarray:
    return np.clip((arr.astype(np.float32) - raw_min) / (raw_max - raw_min), 0.0, 1.0)


def read_float01_png(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)


def ref_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred_hwc = pred[..., None] * 255.0
    gt_hwc = gt[..., None] * 255.0
    return {
        "snr": float(calculate_snr(pred_hwc, gt_hwc, crop_border=0)),
        "psnr": float(calculate_psnr(pred_hwc, gt_hwc, crop_border=0)),
        "ssim": float(calculate_ssim(pred_hwc, gt_hwc, crop_border=0)),
        "epi": float(calculate_epi(pred_hwc, gt_hwc, crop_border=0)),
    }


def to_u8(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if hi <= lo:
        return np.zeros(img.shape, dtype=np.uint8)
    scaled = (img.astype(np.float32) - lo) / (hi - lo)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def fmt_metrics(m: dict[str, float] | None, *, residual_mae: float | None = None) -> list[str]:
    if residual_mae is not None:
        return [f"MAE={residual_mae:.4f}", "(residual |GT-Out|)"]
    if m is None:
        return ["(reference)"]
    return [
        f"SNR {m['snr']:.2f} dB",
        f"PSNR {m['psnr']:.1f}  SSIM {m['ssim']:.3f}",
        f"EPI {m['epi']:.3f}",
    ]


def save_enriched_panel(
    *,
    stem: str,
    columns: list[tuple[str, np.ndarray, dict[str, float] | None, float | None]],
    path: Path,
    display_lo: float,
    display_hi: float,
    residual_hi: float,
) -> None:
    """columns: (label, image01, metrics_or_None, residual_mae_or_None)."""
    font_title = load_font(40)
    font_body = load_font(28)
    h, w = columns[0][1].shape
    n = len(columns)
    canvas = Image.new("RGB", (w * n, h + HEADER_H + FOOTER_H), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, w * n, HEADER_H), fill=(36, 36, 36))
    draw.text((16, 22), stem, fill=(240, 240, 240), font=font_title)

    for i, (lab, img, metrics, res_mae) in enumerate(columns):
        if res_mae is not None:
            u8 = to_u8(img, 0.0, max(residual_hi, 1e-6))
        else:
            u8 = to_u8(img, display_lo, display_hi)
        rgb = Image.fromarray(u8, mode="L").convert("RGB")
        canvas.paste(rgb, (i * w, HEADER_H))
        # Footer bar per column
        y0 = HEADER_H + h
        draw.rectangle((i * w, y0, (i + 1) * w, y0 + FOOTER_H), fill=(28, 28, 28))
        draw.text((i * w + 14, y0 + 10), lab, fill=(255, 220, 120), font=font_title)
        for j, line in enumerate(fmt_metrics(metrics, residual_mae=res_mae)):
            draw.text((i * w + 14, y0 + 58 + j * 30), line, fill=(230, 230, 230), font=font_body)
    canvas.save(path)


def mean_or_nan(rows: list[dict[str, str]], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, "", "nan")]
    if not vals:
        return None
    return sum(vals) / len(vals)


def write_comparison_table(bench: Path, tags: list[str]) -> None:
    label_map = {t: lab for t, lab in MODELS}
    lines = [
        "# モデル × 評価列 比較表 — eval_benchmark_v2（EPI 追加）",
        "",
        "セル: `FT_SNR (Δ vs PreFT) / PSNR / SSIM / EPI`。EPI∈[-1,1]、1 に近いほどエッジ保持。",
        "CNR は欠陥アノテ無しのため今回未実装。",
        "",
        ("| モデル | battery 5/20 | battery 5/27 | cylinder 5/20 val20 | cylinder 5/27 holdout20 | phantom val20 |"),
        "|--------|-------------:|-------------:|--------------------:|----------------------:|-------------:|",
    ]
    for tag in tags:
        cells = [label_map.get(tag, tag)]
        for col in COLUMNS:
            path = bench / "metrics" / f"{tag}__{col}.csv"
            if not path.is_file():
                cells.append("TBD")
                continue
            rows = list(csv.DictReader(path.open(encoding="utf-8")))
            snr = mean_or_nan(rows, "ft_snr")
            dsnr = mean_or_nan(rows, "ft_minus_preft_snr")
            psnr = mean_or_nan(rows, "ft_psnr")
            ssim = mean_or_nan(rows, "ft_ssim")
            epi = mean_or_nan(rows, "ft_epi")
            if None in (snr, dsnr, psnr, ssim, epi):
                cells.append("TBD")
            else:
                cells.append(f"{snr:.2f} (Δ{dsnr:+.2f}) / {psnr:.1f} / {ssim:.3f} / EPI {epi:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "### 注",
            "- EPI: Sobel 勾配強度の GT↔Output 相関。金属/極片エッジがぼけていると下がる。",
            "- 残差パネル: `|GT−Output|`。構造の輪郭が見える = 有用な高周波を消している。",
            "- 訓練損失は MSE（L2）、L1 ではない。",
            "",
        ]
    )
    (bench / "comparison_table.md").write_text("\n".join(lines), encoding="utf-8")


def process_column(bench: Path, tag: str, col: str, names: list[str]) -> None:
    pair = bench / "pairs" / col
    raw_min, raw_max = load_norm(pair / "normalization.json")
    panel_dir = bench / "panels" / tag / col
    if not panel_dir.is_dir():
        raise FileNotFoundError(f"Missing panels: {panel_dir}")

    rows: list[dict[str, object]] = []
    for name in names:
        stem = Path(name).stem
        ft_path = panel_dir / f"{stem}_ft.png"
        pre_path = panel_dir / f"{stem}_preft.png"
        if not ft_path.is_file() or not pre_path.is_file():
            raise FileNotFoundError(f"Missing FT/PreFT PNG for {tag}/{col}/{stem}")

        lq = normalize_u16(read_u16_gray(pair / "LQ" / name), raw_min, raw_max)
        gt = normalize_u16(read_u16_gray(pair / "GT" / name), raw_min, raw_max)
        ft = read_float01_png(ft_path)
        pre = read_float01_png(pre_path)
        if ft.shape != lq.shape or pre.shape != lq.shape:
            raise ValueError(f"Shape mismatch {stem}: lq{lq.shape} ft{ft.shape} pre{pre.shape}")

        lq_m = ref_metrics(lq, gt)
        pre_m = ref_metrics(pre, gt)
        ft_m = ref_metrics(ft, gt)
        res_pre = np.abs(gt - pre)
        res_ft = np.abs(gt - ft)
        residual_hi = float(max(np.percentile(res_pre, 99.5), np.percentile(res_ft, 99.5), 1e-4))
        lo = float(min(lq.min(), gt.min()))
        hi = float(max(lq.max(), gt.max()))

        save_enriched_panel(
            stem=stem,
            columns=[
                ("LQ", lq, lq_m, None),
                ("PreFT", pre, pre_m, None),
                ("FT", ft, ft_m, None),
                ("GT", gt, None, None),
                ("ResPreFT", res_pre, None, float(res_pre.mean())),
                ("ResFT", res_ft, None, float(res_ft.mean())),
            ],
            path=panel_dir / f"{stem}_compare.png",
            display_lo=lo,
            display_hi=hi,
            residual_hi=residual_hi,
        )
        # Also dump residual PNGs for report use
        Image.fromarray(to_u8(res_pre, 0.0, residual_hi), mode="L").save(panel_dir / f"{stem}_res_preft.png")
        Image.fromarray(to_u8(res_ft, 0.0, residual_hi), mode="L").save(panel_dir / f"{stem}_res_ft.png")

        row = {
            "filename": name,
            "lq_snr": f"{lq_m['snr']:.6f}",
            "lq_psnr": f"{lq_m['psnr']:.6f}",
            "lq_ssim": f"{lq_m['ssim']:.6f}",
            "lq_epi": f"{lq_m['epi']:.6f}",
            "preft_snr": f"{pre_m['snr']:.6f}",
            "preft_psnr": f"{pre_m['psnr']:.6f}",
            "preft_ssim": f"{pre_m['ssim']:.6f}",
            "preft_epi": f"{pre_m['epi']:.6f}",
            "ft_snr": f"{ft_m['snr']:.6f}",
            "ft_psnr": f"{ft_m['psnr']:.6f}",
            "ft_ssim": f"{ft_m['ssim']:.6f}",
            "ft_epi": f"{ft_m['epi']:.6f}",
            "ft_minus_preft_snr": f"{ft_m['snr'] - pre_m['snr']:.6f}",
            "ft_minus_preft_epi": f"{ft_m['epi'] - pre_m['epi']:.6f}",
            "res_preft_mae": f"{float(res_pre.mean()):.6f}",
            "res_ft_mae": f"{float(res_ft.mean()):.6f}",
        }
        rows.append(row)
        print(
            f"  {tag}/{col}/{stem}  EPI FT={ft_m['epi']:.3f} PreFT={pre_m['epi']:.3f}  SNR FT={ft_m['snr']:.2f}",
            flush=True,
        )

    fieldnames = list(rows[0].keys())
    for dest in (panel_dir / "metrics.csv", bench / "metrics" / f"{tag}__{col}.csv"):
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    args = parse_args()
    bench = args.bench_root
    columns = args.columns or COLUMNS
    tags = args.tags or [t for t, _ in MODELS]
    if args.mode == "classic":
        if args.delivery_root is None:
            raise SystemExit("--mode classic requires --delivery-root")
        delivery = args.delivery_root
        delivery.mkdir(parents=True, exist_ok=True)
        (delivery / "metrics").mkdir(parents=True, exist_ok=True)
        (delivery / "manifests").mkdir(parents=True, exist_ok=True)
        # Copy manifests + summary for a self-contained classic package
        import shutil

        for col in columns:
            src = bench / "manifests" / f"{col}.txt"
            if src.is_file():
                shutil.copy2(src, delivery / "manifests" / f"{col}.txt")
        summary = bench / "benchmark_summary.json"
        if summary.is_file():
            shutil.copy2(summary, delivery / "benchmark_summary.json")
        for tag in tags:
            for col in columns:
                names = [
                    line.strip()
                    for line in (bench / "manifests" / f"{col}.txt").read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                if args.max_per_column > 0:
                    names = names[: args.max_per_column]
                print(f"=== classic {tag} / {col} (n={len(names)}) ===", flush=True)
                process_column_classic(bench, delivery, tag, col, names)
        write_classic_comparison_table(delivery, tags)
        (delivery / "README_ja.md").write_text(
            "\n".join(
                [
                    "# eval_benchmark_v2 — クラシック版（EPI 追加前）",
                    "",
                    "パネル: `LQ | PreFT | FT | GT`（残差列なし・指標印字なし）。",
                    "本フォルダは Drive 上の EPI 強化版とは別名で保存するための再生成包です。",
                    "推論は再実行せず、既存 `*_ft.png` / `*_preft.png` + pairs TIFF から復元。",
                    "",
                    "## 用語",
                    "- **PreFT**: 公式 SwinIR（微調整前）",
                    "- **FT**: 各データで微調整したモデル",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        print(f"Wrote classic package → {delivery}", flush=True)
        return

    for tag in tags:
        for col in columns:
            names = [
                line.strip()
                for line in (bench / "manifests" / f"{col}.txt").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if args.max_per_column > 0:
                names = names[: args.max_per_column]
            print(f"=== enrich {tag} / {col} (n={len(names)}) ===", flush=True)
            process_column(bench, tag, col, names)
    write_comparison_table(bench, tags)
    print(f"Wrote {bench / 'comparison_table.md'}", flush=True)


if __name__ == "__main__":
    main()
