#!/usr/bin/env bash
# Start the existing CPU-only frontend on a chosen LAN host. No cloud mutations.
set -euo pipefail

usage() {
  cat <<'TEXT'
Usage: UI_BIND_IP=<host LAN IPv4> UI_ADC_PATH=<local user ADC JSON> \
       bash deploy/vertex/start-internal-ui.sh [--check]

Optional: UI_PORT=7860, UI_CONTAINER=denoise-vertex-ui, UI_IMAGE=<image URI>,
          VERTEX_PROJECT=<project>, VERTEX_PREDICT_URL=<dedicated predict URL>
--check validates prerequisites without starting/stopping containers or calling Vertex.
The image must already be pulled. Existing containers are never replaced.
TEXT
}

check_only=false
case "${1:-}" in
  --check) check_only=true ;;
  --help|-h) usage; exit 0 ;;
  '') ;;
  *) usage >&2; exit 2 ;;
esac
if [[ $# -gt 1 ]]; then usage >&2; exit 2; fi

: "${UI_BIND_IP:?Set UI_BIND_IP to the host LAN IPv4 address}"
: "${UI_ADC_PATH:?Set UI_ADC_PATH to an existing local authorized_user ADC file}"
UI_PORT="${UI_PORT:-7860}"
UI_CONTAINER="${UI_CONTAINER:-denoise-vertex-ui}"
UI_IMAGE="${UI_IMAGE:-asia-northeast1-docker.pkg.dev/hallowed-pipe-497802-i0/denoise/denoise-vertex@sha256:fac4b16b9454f8b4685743477959cff89175467d001d3407ecefcf2b4747a73f}"
VERTEX_PROJECT="${VERTEX_PROJECT:-hallowed-pipe-497802-i0}"
VERTEX_PREDICT_URL="${VERTEX_PREDICT_URL:-https://8911645097217490944.asia-northeast1-580558088344.prediction.vertexai.goog/v1/projects/580558088344/locations/asia-northeast1/endpoints/8911645097217490944:predict}"
script_dir="$(dirname -- "${BASH_SOURCE[0]}")"
repo_dir="$(cd -- "$script_dir/../.." && pwd -P)"
mount_type="$(findmnt -n -o FSTYPE -T "$repo_dir")"
case "$mount_type" in
  nfs|nfs4) ;;
  *) printf 'Expected a mounted NFS checkout; got %s for %s\n' "$mount_type" "$repo_dir" >&2; exit 1 ;;
esac
[[ -w "$repo_dir" ]] || { printf 'Checkout is not writable: %s\n' "$repo_dir" >&2; exit 1; }
[[ -r "$repo_dir/deploy/vertex/internal_ui.py" ]] || { printf 'Frontend source is missing\n' >&2; exit 1; }
UI_ADC_PATH="$(realpath -e -- "$UI_ADC_PATH")"
[[ -r "$UI_ADC_PATH" ]] || { printf 'ADC is not readable by the current user\n' >&2; exit 1; }

python3 - "$UI_BIND_IP" "$UI_PORT" "$UI_ADC_PATH" <<'PY'
import ipaddress
import json
import sys
address = ipaddress.IPv4Address(sys.argv[1])
if not address.is_private or address.is_loopback or address.is_unspecified:
    raise SystemExit('Use a specific private LAN IPv4 address for this host.')
if not 1 <= int(sys.argv[2]) <= 65535:
    raise SystemExit('UI_PORT must be in 1..65535.')
with open(sys.argv[3]) as source:
    credential = json.load(source)
if credential.get('type') != 'authorized_user':
    raise SystemExit('The current frontend requires authorized_user ADC.')
# Do not print credential fields or exchange tokens during this check.
PY

docker info --format 'Docker storage: {{.DockerRootDir}}'
docker image inspect "$UI_IMAGE" --format 'Cached image: {{.Id}} ({{.Size}} bytes)'
printf 'NFS source: %s\nFrontend: http://%s:%s\nContainer: %s\n' \
  "$repo_dir" "$UI_BIND_IP" "$UI_PORT" "$UI_CONTAINER"
if docker container inspect "$UI_CONTAINER" >/dev/null 2>&1; then
  docker container inspect "$UI_CONTAINER" --format 'Existing container: {{.Name}}, running={{.State.Running}}'
  if [[ "$check_only" == false ]]; then
    printf 'Container already exists; inspect/reuse it instead of replacing it automatically.\n' >&2
    exit 1
  fi
fi
if [[ "$check_only" == true ]]; then
  printf 'Prerequisites passed. No container, token exchange, or cloud operation was started.\n'
  exit 0
fi

docker run -d --name "$UI_CONTAINER" \
  --restart unless-stopped \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e NVIDIA_VISIBLE_DEVICES=void \
  -e CUDA_VISIBLE_DEVICES= \
  -e GOOGLE_APPLICATION_CREDENTIALS=/adc.json \
  -e VERTEX_PROJECT="$VERTEX_PROJECT" \
  -e VERTEX_PREDICT_URL="$VERTEX_PREDICT_URL" \
  -p "$UI_BIND_IP:$UI_PORT:7860" \
  --mount "type=bind,src=$UI_ADC_PATH,dst=/adc.json,readonly" \
  --mount "type=bind,src=$repo_dir/deploy/vertex/internal_ui.py,dst=/opt/internal_ui.py,readonly" \
  "$UI_IMAGE" python /opt/internal_ui.py
