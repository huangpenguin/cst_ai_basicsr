#!/usr/bin/env bash
# Run PreFT-vs-FT inference for TSFDAN dual-branch on eval_benchmark columns.
# Usage: run_eval_benchmark_tsfdan.sh <root> <ft_ckpt> <tag> [device] [preft_ckpt] [col ...]
set -euo pipefail
ROOT="${1:?benchmark root}"
MODEL="${2:?finetuned model ckpt}"
TAG="${3:?model tag}"
DEVICE="${4:-cuda:0}"
PRE="${5:-experiments/pretrained_models/TSFDAN/SwinTransformer_288800iter.ckpt}"
shift 5 || true
if (($# > 0)); then
  COLUMNS=("$@")
else
  COLUMNS=(
    battery_260520
    battery_260527
    cylinder_260520_val20
    cylinder_260527_holdout20
    phantom_aligned_drive_val20
  )
fi
OUT="${ROOT}/panels/${TAG}"
mkdir -p "${OUT}" "${ROOT}/metrics"
for col in "${COLUMNS[@]}"; do
  echo "=== ${TAG} / ${col} ==="
  col_out="${OUT}/${col}"
  mkdir -p "${col_out}"
  uv run python scripts/infer_tsfdan_tiff.py \
    --model "${MODEL}" \
    --baseline-model "${PRE}" \
    --lq-dir "${ROOT}/pairs/${col}/LQ" \
    --gt-dir "${ROOT}/pairs/${col}/GT" \
    --meta-info "${ROOT}/manifests/${col}.txt" \
    --normalization "${ROOT}/pairs/${col}/normalization.json" \
    --output-dir "${col_out}" \
    --tile 256 \
    --device "${DEVICE}"
  if [[ -f "${col_out}/metrics.csv" ]]; then
    cp -f "${col_out}/metrics.csv" "${ROOT}/metrics/${TAG}__${col}.csv"
  fi
done
echo "DONE ${TAG}"
