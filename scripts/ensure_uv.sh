#!/usr/bin/env bash
# Install uv without root — required when GitLab Docker jobs run as NFS uid (storage_huang).
set -euo pipefail

ensure_uv_home() {
  local storage_user="${CST_STORAGE_USER:-}"

  if [[ -n "${storage_user}" && -d "/home/${storage_user}" && -w "/home/${storage_user}" ]]; then
    if [[ -z "${HOME:-}" || "${HOME}" == "/" || ! -w "${HOME}" ]]; then
      export HOME="/home/${storage_user}"
    fi
  elif [[ -n "${CI_PROJECT_DIR:-}" && -w "${CI_PROJECT_DIR}" ]]; then
    if [[ -z "${HOME:-}" || ! -w "${HOME:-/}" ]]; then
      export HOME="${CI_PROJECT_DIR}"
    fi
  fi

  mkdir -p "${HOME}/.local/bin"
  export PATH="${HOME}/.local/bin:${PATH}"
}

install_uv_via_pip() {
  command -v python3 >/dev/null 2>&1 || return 1
  python3 -m pip --version >/dev/null 2>&1 || return 1

  echo "Installing uv via python3 -m pip install --user..."
  python3 -m pip install --user --upgrade uv
  export PATH="${HOME}/.local/bin:${PATH}"
  command -v uv >/dev/null 2>&1
}

download_uv_installer() {
  local dest="$1"
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh -o "${dest}"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "${dest}" https://astral.sh/uv/install.sh
  elif command -v python3 >/dev/null 2>&1; then
    python3 - <<PY
import urllib.request

url = "https://astral.sh/uv/install.sh"
req = urllib.request.Request(url, headers={"User-Agent": "curl/8.0 (BasicSR-CI)"})
with urllib.request.urlopen(req) as resp, open("${dest}", "wb") as out:
    out.write(resp.read())
PY
  elif [[ "$(id -u)" == "0" ]]; then
    apt-get update
    apt-get install -y --no-install-recommends ca-certificates curl
    rm -rf /var/lib/apt/lists/*
    curl -LsSf https://astral.sh/uv/install.sh -o "${dest}"
  else
    return 1
  fi
}

install_uv_via_installer_script() {
  local installer
  installer="$(mktemp /tmp/uv-install.XXXXXX.sh)"
  download_uv_installer "${installer}"
  sh "${installer}"
  rm -f "${installer}"
  export PATH="${HOME}/.local/bin:${PATH}"
  command -v uv >/dev/null 2>&1
}

install_uv_if_missing() {
  if command -v uv >/dev/null 2>&1; then
    return 0
  fi

  ensure_uv_home
  echo "Installing uv to ${HOME}/.local/bin (uid $(id -u))..."

  if install_uv_via_pip; then
    return 0
  fi
  if install_uv_via_installer_script; then
    return 0
  fi

  echo "ERROR: failed to install uv (tried pip, install.sh)." >&2
  exit 1
}

install_uv_if_missing
uv --version
