#!/usr/bin/env bash
# Serve TensorBoard for all CI runs under a storage ci_outputs tree (GPU host).
# No uv required on the host — uses the official TensorFlow Docker image.
#
# Usage:
#   bash scripts/serve_tensorboard.sh
#   bash scripts/serve_tensorboard.sh /cache/huang.pengbin/cst_ai/ci_outputs
#   TB_PORT=6007 bash scripts/serve_tensorboard.sh
#
# Then from your laptop:
#   ssh -L 6006:localhost:6006 user@gpu-host
#   open http://localhost:6006

set -euo pipefail

LOGIN="${CST_STORAGE_USER:-${USER:-ci}}"
LOGDIR="${1:-}"

if [[ -z "${LOGDIR}" ]]; then
  for candidate in \
    "/mnt/home/${LOGIN}/cst_ai/ci_outputs" \
    "/home/${LOGIN}/cst_ai/ci_outputs"; do
    if [[ -d "${candidate}" ]]; then
      LOGDIR="${candidate}"
      break
    fi
  done
fi

if [[ -z "${LOGDIR}" || ! -d "${LOGDIR}" ]]; then
  echo "ERROR: no ci_outputs directory found." >&2
  echo "Pass explicitly: bash scripts/serve_tensorboard.sh /mnt/home/${LOGIN}/cst_ai/ci_outputs" >&2
  exit 1
fi

PORT="${TB_PORT:-6006}"
NAME="${TB_CONTAINER_NAME:-tb-cst-ai}"

echo "TensorBoard logdir: ${LOGDIR}"
echo "URL (after SSH port forward): http://localhost:${PORT}"

docker rm -f "${NAME}" 2>/dev/null || true
exec docker run --rm -d --name "${NAME}" \
  -p "${PORT}:6006" \
  -v "${LOGDIR}:/logs:ro" \
  tensorflow/tensorflow:latest \
  tensorboard --logdir /logs --host 0.0.0.0 --port 6006
