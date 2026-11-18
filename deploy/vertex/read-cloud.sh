#!/usr/bin/env bash
set -euo pipefail
: "${UI_ADC_PATH:=/home/huang/.config/basicsr-vertex/application_default_credentials.json}"
source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
docker run --rm --read-only --tmpfs /tmp \
  --user "$(id -u):$(id -g)" -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= \
  --mount "type=bind,src=$UI_ADC_PATH,dst=/adc.json,readonly" \
  --mount "type=bind,src=$source_dir/inspect-cloud.py,dst=/cloud_status.py,readonly" \
  --entrypoint /usr/lib/google-cloud-sdk/platform/bundledpythonunix/bin/python3 \
  gcr.io/google.com/cloudsdktool/google-cloud-cli@sha256:a652e9fffcff1e3b0528492a2a8278dee5e5ec7ef8d08dcd05e3e4fea5415b35 /cloud_status.py
