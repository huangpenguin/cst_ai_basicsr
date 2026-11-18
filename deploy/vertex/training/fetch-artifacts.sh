#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 ]]; then echo 'Usage: fetch-artifacts.sh gs://bucket/run-prefix /absolute/new-local-output-directory' >&2; exit 2; fi
: "${UI_ADC_PATH:=/home/huang/.config/basicsr-vertex/application_default_credentials.json}"
output_dir="$2"
[[ "$output_dir" == /* && ! -e "$output_dir" && "$output_dir" != *,* ]] || { echo 'Use a new absolute output path without commas.' >&2; exit 1; }
output_parent="$(dirname -- "$output_dir")"
[[ -d "$output_parent" ]] || { echo 'Output parent must already exist.' >&2; exit 1; }
source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
docker run --rm --read-only --tmpfs /tmp --user "$(id -u):$(id -g)" \
  -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= \
  --mount "type=bind,src=$UI_ADC_PATH,dst=/adc.json,readonly" \
  --mount "type=bind,src=$source_dir/fetch_artifacts.py,dst=/fetch_artifacts.py,readonly" \
  --mount "type=bind,src=$output_parent,dst=/output" \
  --entrypoint /usr/lib/google-cloud-sdk/platform/bundledpythonunix/bin/python3 \
  gcr.io/google.com/cloudsdktool/google-cloud-cli@sha256:a652e9fffcff1e3b0528492a2a8278dee5e5ec7ef8d08dcd05e3e4fea5415b35 \
  /fetch_artifacts.py --prefix "$1" --output "/output/$(basename -- "$output_dir")"
