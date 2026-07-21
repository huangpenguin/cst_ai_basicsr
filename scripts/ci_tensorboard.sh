#!/usr/bin/env bash
# Start/stop TensorBoard for a single CI training run (lifecycle-bound).
#
# Usage:
#   bash scripts/ci_tensorboard.sh start /path/to/tb_logger
#   bash scripts/ci_tensorboard.sh stop
#
# Env:
#   TB_PORT              default 6006
#   TB_NAME              pid/log basename (default tb-cst-ai)
#   TB_RELOAD_INTERVAL   seconds (default 30)
#   CI_RUN_ROOT          optional; also writes ${CI_RUN_ROOT}/.tb.pid
#
# Mac viewing (with runner network_mode=host or host-published port):
#   ssh -L 6006:localhost:6006 user@gpu-host
#   open http://localhost:6006

set -euo pipefail

CMD="${1:-}"
LOGDIR="${2:-}"
PORT="${TB_PORT:-6006}"
NAME="${TB_NAME:-tb-cst-ai}"
RELOAD_INTERVAL="${TB_RELOAD_INTERVAL:-30}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PID_FILE_TMP="/tmp/${NAME}.pid"
LOG_FILE_TMP="/tmp/${NAME}.log"
PID_FILE_RUN=""
if [[ -n "${CI_RUN_ROOT:-}" ]]; then
  PID_FILE_RUN="${CI_RUN_ROOT}/.tb.pid"
fi

usage() {
  echo "Usage: $0 start <logdir> | stop" >&2
  exit 2
}

pid_alive() {
  local pid="$1"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

kill_pid_tree() {
  local pid="$1"
  if ! pid_alive "${pid}"; then
    return 0
  fi
  # Kill children first (tensorboard data server), then parent.
  local kids
  kids="$(pgrep -P "${pid}" 2>/dev/null || true)"
  if [[ -n "${kids}" ]]; then
    # shellcheck disable=SC2086
    kill ${kids} 2>/dev/null || true
  fi
  kill "${pid}" 2>/dev/null || true
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    pid_alive "${pid}" || return 0
    sleep 0.2
  done
  kill -9 "${pid}" 2>/dev/null || true
  if [[ -n "${kids}" ]]; then
    # shellcheck disable=SC2086
    kill -9 ${kids} 2>/dev/null || true
  fi
}

free_port() {
  local pids=""
  if command -v fuser >/dev/null 2>&1; then
    pids="$(fuser -n tcp:"${PORT}" 2>/dev/null || true)"
  elif command -v lsof >/dev/null 2>&1; then
    pids="$(lsof -t -iTCP:"${PORT}" -sTCP:LISTEN 2>/dev/null || true)"
  fi
  if [[ -n "${pids}" ]]; then
    echo "Freeing port ${PORT}: ${pids}"
    # shellcheck disable=SC2086
    kill ${pids} 2>/dev/null || true
    sleep 0.5
    # shellcheck disable=SC2086
    kill -9 ${pids} 2>/dev/null || true
  fi
}

stop_from_pidfile() {
  local file="$1"
  [[ -f "${file}" ]] || return 0
  local pid
  pid="$(tr -d '[:space:]' <"${file}" || true)"
  if [[ -n "${pid}" ]]; then
    echo "Stopping TensorBoard pid=${pid} (from ${file})"
    kill_pid_tree "${pid}"
  fi
  rm -f "${file}"
}

cmd_stop() {
  set +e
  stop_from_pidfile "${PID_FILE_TMP}"
  if [[ -n "${PID_FILE_RUN}" ]]; then
    stop_from_pidfile "${PID_FILE_RUN}"
  fi
  # Also clear any leftover listeners on the fixed port.
  free_port
  # Match our typical launch line without killing unrelated TB.
  pkill -f "tensorboard.main --logdir .* --host 0.0.0.0 --port ${PORT}" 2>/dev/null || true
  echo "TensorBoard stop complete (port ${PORT})"
  return 0
}

port_open() {
  local py="${REPO_ROOT}/.venv/bin/python"
  if [[ -x "${py}" ]]; then
    "${py}" -c "import socket; s=socket.socket(); s.settimeout(0.5); r=s.connect_ex(('127.0.0.1', ${PORT})); s.close(); raise SystemExit(0 if r==0 else 1)"
    return $?
  fi
  if command -v python3 >/dev/null 2>&1; then
    python3 -c "import socket; s=socket.socket(); s.settimeout(0.5); r=s.connect_ex(('127.0.0.1', ${PORT})); s.close(); raise SystemExit(0 if r==0 else 1)"
    return $?
  fi
  return 1
}

cmd_start() {
  if [[ -z "${LOGDIR}" ]]; then
    usage
  fi
  mkdir -p "${LOGDIR}"
  LOGDIR="$(cd "${LOGDIR}" && pwd)"

  # Always recycle previous instance first.
  cmd_stop

  local py="${REPO_ROOT}/.venv/bin/python"
  if [[ ! -x "${py}" ]]; then
    echo "ERROR: project venv python missing: ${py}" >&2
    echo "Run uv sync / uv-bootstrap before starting TensorBoard." >&2
    exit 1
  fi
  if ! "${py}" -c "from tensorboard.main import run_main" >/dev/null 2>&1; then
    echo "ERROR: tensorboard not importable in ${py}" >&2
    exit 1
  fi

  echo "Starting TensorBoard"
  echo "  logdir: ${LOGDIR}"
  echo "  port:   ${PORT}"
  echo "  Mac:    ssh -L ${PORT}:localhost:${PORT} <user>@<gpu-host>"
  echo "  URL:    http://localhost:${PORT}"
  echo "  NOTE:   GPU runner needs network_mode=host (or published ${PORT}) for Mac tunnel."

  # Prefer classic event file polling on NFS; avoid sticky data-server cache.
  setsid "${py}" -m tensorboard.main \
    --logdir "${LOGDIR}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --reload_interval "${RELOAD_INTERVAL}" \
    --load_fast false \
    </dev/null >"${LOG_FILE_TMP}" 2>&1 &
  local pid=$!
  echo "${pid}" >"${PID_FILE_TMP}"
  if [[ -n "${PID_FILE_RUN}" ]]; then
    mkdir -p "$(dirname "${PID_FILE_RUN}")"
    echo "${pid}" >"${PID_FILE_RUN}"
  fi

  local ok=0
  for _ in $(seq 1 30); do
    if ! pid_alive "${pid}"; then
      echo "ERROR: TensorBoard exited early; see ${LOG_FILE_TMP}" >&2
      tail -n 40 "${LOG_FILE_TMP}" >&2 || true
      exit 1
    fi
    if port_open; then
      ok=1
      break
    fi
    sleep 0.5
  done
  if [[ "${ok}" -ne 1 ]]; then
    echo "ERROR: TensorBoard did not open port ${PORT}; see ${LOG_FILE_TMP}" >&2
    tail -n 40 "${LOG_FILE_TMP}" >&2 || true
    cmd_stop
    exit 1
  fi
  echo "TensorBoard started pid=${pid}; logs: ${LOG_FILE_TMP}"
}

case "${CMD}" in
  start) cmd_start ;;
  stop) cmd_stop ;;
  *) usage ;;
esac
