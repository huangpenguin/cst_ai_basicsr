#!/usr/bin/env bash
# Serve TensorBoard for CI / NFS training logs (GPU host).
#
# Prefers the project venv when tensorboard imports cleanly; otherwise uses a
# slim Python Docker image. Do not use tensorflow/tensorflow:latest — that
# image no longer ships the tensorboard CLI.
#
# Usage:
#   bash scripts/serve_tensorboard.sh
#   bash scripts/serve_tensorboard.sh /mnt/home/huang/cst_ai/ci_outputs
#   TB_PORT=6007 bash scripts/serve_tensorboard.sh
#   TB_FORCE_DOCKER=1 bash scripts/serve_tensorboard.sh
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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TB_IMAGE="${TB_IMAGE:-python:3.11-slim}"

echo "TensorBoard logdir: ${LOGDIR}"
echo "URL (after SSH port forward): http://localhost:${PORT}"

serve_via_venv() {
  local py="${REPO_ROOT}/.venv/bin/python"
  if [[ ! -x "${py}" ]]; then
    return 1
  fi
  if ! "${py}" -c "from tensorboard.main import run_main" >/dev/null 2>&1; then
    return 1
  fi
  echo "Using project venv tensorboard (${py})"
  # setsid detaches from the invoking TTY so TensorBoard survives script exit.
  setsid "${py}" -m tensorboard.main \
    --logdir "${LOGDIR}" --host 0.0.0.0 --port "${PORT}" \
    </dev/null >"/tmp/${NAME}.log" 2>&1 &
  echo $! >"/tmp/${NAME}.pid"
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    if ! kill -0 "$(cat "/tmp/${NAME}.pid")" 2>/dev/null; then
      echo "ERROR: tensorboard exited early; see /tmp/${NAME}.log" >&2
      return 1
    fi
    if "${py}" -c "import socket; s=socket.socket(); s.settimeout(0.5); r=s.connect_ex(('127.0.0.1', ${PORT})); s.close(); raise SystemExit(0 if r==0 else 1)"; then
      break
    fi
    sleep 0.5
  done
  echo "Started PID $(cat "/tmp/${NAME}.pid"); logs: /tmp/${NAME}.log"
  return 0
}

serve_via_docker() {
  docker rm -f "${NAME}" 2>/dev/null || true
  docker run --rm -d --name "${NAME}" \
    -p "${PORT}:6006" \
    -v "${LOGDIR}:/logs:ro" \
    "${TB_IMAGE}" \
    bash -lc 'pip install -q "setuptools>=70,<82" "tensorboard>=2.14,<2.20" && exec tensorboard --logdir /logs --host 0.0.0.0 --port 6006'
  echo "Docker container: ${NAME}"
}

if [[ "${TB_FORCE_DOCKER:-0}" != "1" ]] && serve_via_venv; then
  exit 0
fi

serve_via_docker
