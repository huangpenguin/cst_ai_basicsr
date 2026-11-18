#!/usr/bin/env bash
# Native refreshable user ADC, no second CLI login or fixed one-hour access token.
set -euo pipefail
: "${UI_ADC_PATH:=/home/huang/.config/basicsr-vertex/application_default_credentials.json}"
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
[[ -r "$UI_ADC_PATH" ]] || { echo 'A readable local user ADC is required.' >&2; exit 1; }
extra_mounts=()
if [[ -n "${VERTEX_DATA_SOURCE:-}" ]]; then
  data_source="$(realpath -e -- "$VERTEX_DATA_SOURCE")"
  [[ -f "$data_source" && "$data_source" != *,* ]] || { echo 'Use a regular data file without commas in its path.' >&2; exit 1; }
  extra_mounts+=(--mount "type=bind,src=$data_source,dst=/input/data.tar,readonly")
fi
if [[ -n "${VERTEX_UPLOAD_STATE:-}" ]]; then
  [[ -d "$VERTEX_UPLOAD_STATE" ]] || { echo 'Upload state directory does not exist.' >&2; exit 1; }
  extra_mounts+=(--mount "type=bind,src=$VERTEX_UPLOAD_STATE,dst=/upload-state")
fi
docker run --rm -i --read-only --tmpfs /tmp \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= \
  -e CLOUDSDK_CONFIG=/tmp/gcloud \
  -e CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE=/adc.json \
  -e CLOUDSDK_STORAGE_TRACKER_FILES_DIRECTORY=/upload-state \
  --mount "type=bind,src=$UI_ADC_PATH,dst=/adc.json,readonly" \
  --mount "type=bind,src=$repo_dir,dst=/workspace,readonly" \
  "${extra_mounts[@]}" --workdir /workspace \
  gcr.io/google.com/cloudsdktool/google-cloud-cli@sha256:a652e9fffcff1e3b0528492a2a8278dee5e5ec7ef8d08dcd05e3e4fea5415b35 \
  gcloud "$@"
