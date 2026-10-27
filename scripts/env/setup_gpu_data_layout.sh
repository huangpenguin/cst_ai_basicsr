#!/usr/bin/env bash
# One-time host setup for autofs NFS GPU dev + CI (main_gpu / storage_huang).
#
# Layout:
#   /mnt/data/<PROJECT_NAME>/     → container /data/<PROJECT_NAME>/     (raw, ro)
#   /mnt/home/<user>/<PROJECT_NAME>/ → container /home/<user>/<PROJECT_NAME>/ (outputs, rw)

set -euo pipefail

PROJECT_NAME="${PROJECT_NAME:-cst_ai}"
USER_NAME="${USER_NAME:-$(id -un)}"
STORAGE_UID="$(id -u "${USER_NAME}")"
STORAGE_GID="$(id -g "${USER_NAME}")"

echo "=== Personal project output dir ==="
mkdir -p "/mnt/home/${USER_NAME}/${PROJECT_NAME}/"{outputs,ci_outputs,datasets}
echo "  /mnt/home/${USER_NAME}/${PROJECT_NAME}/"

echo ""
echo "=== GitLab Web (project jil_atr/basicsr) ==="
cat <<'EOF'
  Settings → CI/CD → Runners → New project runner
    Tags: linux, docker, main_gpu, storage_huang
    Run untagged jobs: OFF
  Copy glrt-... token for register on gpu01.
EOF

echo ""
echo "=== gpu01: register second runner (do NOT change shared main_gpu runner) ==="
cat <<EOF
  sudo gitlab-runner register \\
    --url https://gitlab.com \\
    --token glrt-<paste-from-gitlab> \\
    --executor docker \\
    --docker-image basicsr-mlops-gpu:2.6.0-cu124

  # Build the same image once on gpu01 (same as Dev Container):
  #   cd /path/to/basicsr && bash scripts/env/build_mlops_image.sh
  # In [runners.docker] for this runner, prefer: pull_policy = ["if-not-present"]

  # Then edit /etc/gitlab-runner/config.toml — find the NEW [[runners]] block by name/token:
  #   name = "gpu01-storage-huang-runner"   (or your registration name)
  # Under that block's [runners.docker] section (NOT the shared main_gpu runner):
  tags = ["linux", "docker", "main_gpu", "storage_huang"]
  gpus = "all"
  volumes = [
    "/cache",
    "/mnt/data:/data:ro",
    "/mnt/home:/home:rw",
  ]
  user = "${STORAGE_UID}:${STORAGE_GID}"

  # Common mistake: user= added to the wrong [[runners]] block, or job still shows Container user: 0:0

  sudo gitlab-runner verify && sudo gitlab-runner restart

  # Write test (NFS root_squash):
  docker run --rm --user ${STORAGE_UID}:${STORAGE_GID} -v /mnt/home:/home:rw alpine \\
    touch /home/${USER_NAME}/${PROJECT_NAME}/ci_outputs/.runner_write_test
EOF

echo ""
echo "=== Shared runner (unchanged) ==="
cat <<'EOF'
  tags = ["linux", "docker", "main_gpu"]
  # no user= — containers run as root; cannot write /home/<user>/cst_ai on NFS
EOF

echo ""
echo "=== Project GPU image (CI + Dev Container) ==="
echo "  bash scripts/env/build_mlops_image.sh   # on gpu01, once after clone/pull"
echo "  MLOPS_PROJECT_IMAGE = basicsr-mlops-gpu:2.6.0-cu124"
echo ""
echo "=== Per-repo CI variables (optional) ==="
echo "  CST_STORAGE_USER = huang"
echo "  WANDB_API_KEY = <masked>  (optional)"
