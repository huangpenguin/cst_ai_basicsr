#!/usr/bin/env bash
# One-time host setup for autofs NFS GPU dev + CI (main_gpu).
#
# Layout:
#   /mnt/data/<PROJECT_NAME>/     → container /data/<PROJECT_NAME>/     (raw, ro)
#   /mnt/home/<user>/<PROJECT_NAME>/ → container /home/<user>/<PROJECT_NAME>/ (outputs, rw)

set -euo pipefail

PROJECT_NAME="${PROJECT_NAME:-cst_ai}"
USER_NAME="${USER_NAME:-$(id -un)}"

echo "=== Personal project output dir ==="
mkdir -p "/mnt/home/${USER_NAME}/${PROJECT_NAME}/"{outputs,ci_outputs}
echo "  /mnt/home/${USER_NAME}/${PROJECT_NAME}/"

echo ""
echo "=== GitLab Runner config.toml (same for all users — no per-user paths) ==="
cat <<'EOF'
  volumes = [
    "/cache",
    "/mnt/data:/data:ro",
    "/mnt/home:/home:rw",
  ]
EOF
echo "  sudo gitlab-runner restart"
echo ""
echo "=== Per-repo CI variables (optional; defaults in .gitlab/ci/train.yml) ==="
echo "  PROJECT_NAME = cst_ai"
echo "  DATASET_DIR  = /data/cst_ai/..."
echo "  OUTPUT_DIR   = /home/\$GITLAB_USER_LOGIN/cst_ai/ci_outputs/\$CI_PIPELINE_ID/..."
echo ""
echo "=== Local dev (before Rebuild Container) ==="
echo "  export DATA_MOUNT_SOURCE=/mnt/data"
echo "  export HOME_MOUNT_SOURCE=/mnt/home"
echo "  export WORK_DIR=/home/${USER_NAME}"
echo "  export PROJECT_NAME=${PROJECT_NAME}"
