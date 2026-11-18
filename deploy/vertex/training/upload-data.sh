#!/usr/bin/env bash
# Explicit operator-triggered upload, never called by the page or training runner.
set -euo pipefail
check_only=false
if [[ "${1:-}" == --check ]]; then check_only=true; shift; fi
if [[ $# != 2 ]]; then
  echo 'Usage: bash deploy/vertex/training/upload-data.sh [--check] /absolute/dataset.tar gs://bucket/new-version/dataset.tar' >&2
  exit 2
fi
source_file="$(realpath -e -- "$1")"
object_uri="$2"
[[ -f "$source_file" && "$source_file" != *,* ]] || { echo 'Expected a regular archive file.' >&2; exit 1; }
python3 - "$object_uri" <<'PY'
import sys
from urllib.parse import urlparse
uri=urlparse(sys.argv[1])
if uri.scheme!='gs' or not uri.netloc or not uri.path.strip('/') or uri.query or uri.fragment or '*' in sys.argv[1]:
    raise SystemExit('Expected one literal gs://bucket/new-object URI')
PY
if [[ "$check_only" == true ]]; then
  stat --format='Input: %n (%s bytes)' "$source_file"
  printf 'Target: %s\nNo hash, authentication or upload performed.\n' "$object_uri"
  exit 0
fi
archive_sha="$(python3 - "$source_file" <<'PY'
import hashlib,sys
with open(sys.argv[1],'rb') as source: print(hashlib.file_digest(source,'sha256').hexdigest())
PY
)"
printf 'Input SHA256: %s\n' "$archive_sha"
: "${UI_ADC_PATH:=/home/huang/.config/basicsr-vertex/application_default_credentials.json}"
state_dir="$(dirname -- "$UI_ADC_PATH")/upload-state"
install -d -m 700 "$state_dir"
vertex_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
VERTEX_DATA_SOURCE="$source_file" VERTEX_UPLOAD_STATE="$state_dir" \
  bash "$vertex_dir/cloud-cli.sh" storage cp \
  --project=hallowed-pipe-497802-i0 --billing-project=hallowed-pipe-497802-i0 \
  --if-generation-match=0 --custom-metadata="sha256=$archive_sha" \
  /input/data.tar "$object_uri"
printf 'Uploaded: %s\nArchive SHA256: %s\n' "$object_uri" "$archive_sha"
