#!/usr/bin/env bash
# Build the shared GPU image used by Dev Container and GitLab CI (see Dockerfile).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${MLOPS_PROJECT_IMAGE:-basicsr-mlops-gpu:2.6.0-cu124}"

cd "${ROOT}"
echo "Building ${IMAGE} from Dockerfile (base: pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel)..."
docker build -t "${IMAGE}" -f Dockerfile \
  --build-arg USERNAME=vscode \
  --build-arg USER_UID=1000 \
  --build-arg USER_GID=1000 \
  .

docker run --rm "${IMAGE}" uv --version
echo ""
echo "OK: ${IMAGE}"
echo "  Dev Container: Rebuild Container in IDE"
echo "  GitLab CI: set MLOPS_PROJECT_IMAGE=${IMAGE} (default in train.yml) and pull_policy: if-not-present on gpu01"
