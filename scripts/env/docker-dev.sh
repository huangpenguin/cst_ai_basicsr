#!/usr/bin/env bash
# Run commands in the GPU dev container (same compose service as Dev Container).
#
# Examples:
#   export DATA_MOUNT_SOURCE=/mnt/data
#   export HOME_MOUNT_SOURCE=/mnt/home   # optional; auto when /mnt/home exists
#   ./scripts/env/docker-dev.sh shell
#   ./scripts/env/docker-dev.sh run uv run python scripts/ci/ci_gpu_smoke.py
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

COMPOSE_FILE="${DOCKER_COMPOSE_FILE:-docker-compose.yml}"
COMPOSE=(docker compose -f "${COMPOSE_FILE}")
SERVICE="${DOCKER_DEV_SERVICE:-dev}"

SHM_SIZE="${DOCKER_DEV_SHM_SIZE:-16g}"
GPU_DEVICES="${DOCKER_DEV_GPU:-all}"
DATA_MOUNT_SOURCE="${DATA_MOUNT_SOURCE:-}"
HOME_MOUNT_SOURCE="${HOME_MOUNT_SOURCE:-}"
DATA_READONLY="${DOCKER_DEV_DATA_READONLY:-0}"
BOOTSTRAP="${DOCKER_DEV_BOOTSTRAP:-1}"
INTERACTIVE="${DOCKER_DEV_INTERACTIVE:-auto}"
ALLOW_EMPTY_DATA="${DOCKER_DEV_ALLOW_EMPTY_DATA:-0}"
EMPTY_DATA_DIR="${DOCKER_DEV_EMPTY_DATA_DIR:-${HOME}/.local/share/mlops-empty-data}"
RUN_USER="${DOCKER_DEV_USER:-$(id -u):$(id -g)}"

usage() {
  cat <<'EOF'
Usage: scripts/env/docker-dev.sh <command> [args...]

Commands:
  shell              Interactive bash (bootstrap first unless DOCKER_DEV_BOOTSTRAP=0)
  bootstrap          Run scripts/env/uv-bootstrap.sh only
  run <cmd...>       Run a command non-interactively (bootstrap first by default)
  build              docker compose build dev
  help               Show this help

Environment (host):
  DATA_MOUNT_SOURCE          Host dataset directory mounted at /data (required unless
                             DOCKER_DEV_ALLOW_EMPTY_DATA=1)
  HOME_MOUNT_SOURCE          Host personal NFS (default: /mnt/home if present) -> /home
  DOCKER_DEV_DATA_READONLY   Set to 1 to mount /data read-only (CI-style)
  DOCKER_DEV_BOOTSTRAP       Set to 0 to skip uv-bootstrap.sh (default: 1)
  DOCKER_DEV_USER            Container user (default: current uid:gid)

See docs/infrastructure.md
EOF
}

resolve_home_mount() {
  if [[ -n "${HOME_MOUNT_SOURCE}" ]]; then
    if [[ ! -d "${HOME_MOUNT_SOURCE}" ]]; then
      echo "ERROR: HOME_MOUNT_SOURCE is not a directory: ${HOME_MOUNT_SOURCE}" >&2
      exit 1
    fi
    printf '%s' "${HOME_MOUNT_SOURCE}"
    return 0
  fi
  if [[ -d /mnt/home ]]; then
    printf '%s' "/mnt/home"
    return 0
  fi
  mkdir -p "${EMPTY_DATA_DIR}/home-stub"
  echo "WARNING: /mnt/home not found; using stub home mount: ${EMPTY_DATA_DIR}/home-stub" >&2
  printf '%s' "${EMPTY_DATA_DIR}/home-stub"
}

resolve_data_mount() {
  if [[ -n "${DATA_MOUNT_SOURCE}" ]]; then
    if [[ ! -d "${DATA_MOUNT_SOURCE}" ]]; then
      echo "ERROR: DATA_MOUNT_SOURCE is not a directory: ${DATA_MOUNT_SOURCE}" >&2
      exit 1
    fi
    printf '%s' "${DATA_MOUNT_SOURCE}"
    return 0
  fi
  if [[ "${ALLOW_EMPTY_DATA}" == "1" ]]; then
    mkdir -p "${EMPTY_DATA_DIR}"
    echo "WARNING: DATA_MOUNT_SOURCE unset; using empty data dir: ${EMPTY_DATA_DIR}" >&2
    printf '%s' "${EMPTY_DATA_DIR}"
    return 0
  fi
  echo "ERROR: export DATA_MOUNT_SOURCE before running (or set DOCKER_DEV_ALLOW_EMPTY_DATA=1)." >&2
  exit 1
}

compose_env() {
  export DATA_MOUNT_SOURCE="$(resolve_data_mount)"
  export HOME_MOUNT_SOURCE="$(resolve_home_mount)"
  if [[ "${DATA_READONLY}" == "1" ]]; then
    export DOCKER_DEV_DATA_MODE=ro
  else
    export DOCKER_DEV_DATA_MODE=rw
  fi
}

compose_run_args() {
  local -a args=(run --rm)
  if [[ "${INTERACTIVE}" == "1" || "${INTERACTIVE}" == "true" || "${INTERACTIVE}" == "yes" ]]; then
    args+=(-it)
  elif [[ "${INTERACTIVE}" == "auto" && -t 0 && -t 1 ]]; then
    args+=(-it)
  fi
  args+=(
    --user "${RUN_USER}"
    "${SERVICE}"
  )
  printf '%s\n' "${args[@]}"
}

run_in_container() {
  compose_env
  local -a run_args=()
  mapfile -t run_args < <(compose_run_args)
  "${COMPOSE[@]}" "${run_args[@]}" "$@"
}

bootstrap_cmd() {
  printf '%s' 'bash scripts/env/uv-bootstrap.sh && export VIRTUAL_ENV=/workspace/.venv PATH=/workspace/.venv/bin:${PATH}'
}

cmd_shell() {
  local inner="exec bash"
  if [[ "${BOOTSTRAP}" == "1" ]]; then
    inner="$(bootstrap_cmd); exec bash"
  fi
  run_in_container bash -lc "${inner}"
}

cmd_bootstrap() {
  run_in_container bash -lc "$(bootstrap_cmd)"
}

cmd_run() {
  if [[ "$#" -lt 1 ]]; then
    echo "ERROR: scripts/env/docker-dev.sh run requires a command." >&2
    usage >&2
    exit 1
  fi
  local quoted
  quoted="$(printf '%q ' "$@")"
  local inner="${quoted}"
  if [[ "${BOOTSTRAP}" == "1" ]]; then
    inner="$(bootstrap_cmd); ${quoted}"
  fi
  run_in_container bash -lc "${inner}"
}

cmd_build() {
  compose_env
  "${COMPOSE[@]}" build "${SERVICE}"
}

main() {
  local cmd="${1:-shell}"
  case "${cmd}" in
    -h | --help | help)
      usage
      ;;
    shell)
      shift || true
      cmd_shell "$@"
      ;;
    bootstrap)
      shift || true
      cmd_bootstrap "$@"
      ;;
    run)
      shift
      cmd_run "$@"
      ;;
    build)
      shift || true
      cmd_build "$@"
      ;;
    *)
      echo "ERROR: unknown command: ${cmd}" >&2
      usage >&2
      exit 1
      ;;
  esac
}

main "$@"
