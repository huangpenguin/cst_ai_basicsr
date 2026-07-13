#!/usr/bin/env bash
# CI / devcontainer expect uv in the project image (Dockerfile). No apt-get in job.
set -euo pipefail

if command -v uv >/dev/null 2>&1; then
  uv --version
  exit 0
fi

echo "ERROR: uv not found in job container." >&2
echo "CI and Dev Container share the project image built from Dockerfile (not bare pytorch/pytorch)." >&2
echo "On gpu01 (once): bash scripts/build_mlops_image.sh" >&2
echo "Then ensure GitLab CI uses MLOPS_PROJECT_IMAGE=basicsr-mlops-gpu:2.6.0-cu124 with pull_policy: if-not-present." >&2
exit 1
