#!/usr/bin/env bash
# Serve TensorBoard for CI / NFS training logs (GPU host).
#
# Prefers lifecycle helper scripts/ci_tensorboard.sh (venv). Falls back to Docker
# when the project venv cannot import tensorboard.
#
# Usage:
#   bash scripts/serve_tensorboard.sh
#   bash scripts/serve_tensorboard.sh /mnt/home/huang/cst_ai/ci_outputs/latest/tb_logger
#   TB_PORT=6007 bash scripts/serve_tensorboard.sh
#   TB_FORCE_DOCKER=1 bash scripts/serve_tensorboard.sh
#
# Then from your laptop:
#   ssh -L 6006:localhost:6006 user@gpu-host
#   open http://localhost:6006

set -euo pipefail

LOGIN="${CST_STORAGE_USER:-${USER:-ci}}"
LOGDIR="${1:-}"
PORT="${TB_PORT:-6006}"
NAME="${TB_CONTAINER_NAME:-tb-cst-ai}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TB_IMAGE="${TB_IMAGE:-python:3.11-slim}"

if [[ -z "${LOGDIR}" ]]; then
  for candidate in \
    "/mnt/home/${LOGIN}/cst_ai/ci_outputs/latest/tb_logger" \
    "/home/${LOGIN}/cst_ai/ci_outputs/latest/tb_logger" \
    "/mnt/home/${LOGIN}/cst_ai/ci_outputs" \
    "/home/${LOGIN}/cst_ai/ci_outputs"; do
    if [[ -d "${candidate}" ]]; then
      LOGDIR="${candidate}"
      break
    fi
  done
fi

if [[ -z "${LOGDIR}" || ! -d "${LOGDIR}" ]]; then
  echo "ERROR: no tb_logger / ci_outputs directory found." >&2
  echo "Pass explicitly: bash scripts/serve_tensorboard.sh /mnt/home/${LOGIN}/cst_ai/ci_outputs/latest/tb_logger" >&2
  exit 1
fi

if [[ "${TB_FORCE_DOCKER:-0}" != "1" ]]; then
  if bash "${SCRIPT_DIR}/ci_tensorboard.sh" start "${LOGDIR}"; then
    exit 0
  fi
  echo "venv TensorBoard start failed; falling back to Docker..." >&2
fi

echo "TensorBoard logdir: ${LOGDIR}"
echo "URL (after SSH port forward): http://localhost:${PORT}"
docker rm -f "${NAME}" 2>/dev/null || true
docker run --rm -d --name "${NAME}" \
  -p "${PORT}:6006" \
  -v "${LOGDIR}:/logs:ro" \
  "${TB_IMAGE}" \
  bash -lc 'pip install -q "setuptools>=70,<82" "tensorboard>=2.14,<2.20" && exec tensorboard --logdir /logs --host 0.0.0.0 --port 6006 --reload_interval 30 --load_fast false'
echo "Docker container: ${NAME}"
