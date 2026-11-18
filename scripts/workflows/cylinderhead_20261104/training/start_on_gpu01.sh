#!/usr/bin/env bash
set -euo pipefail
[[ "$(hostname)" == "gpu01" ]] || { echo 'Run this on gpu01 (192.168.3.18).' >&2; exit 1; }
run_dir=/mnt/home/huang/cst_ai/experiments/cylinderhead_300kv_mean3_20261104_173035_JST
pair_dir=/mnt/data/cst_ai/datasets/cylinderhead_300kv_pairs_raw_mean3_20261104_173035_JST
environment_dir=/home/huang/.local/share/basicsr-training/20261104_173035_JST
container_name=cylinderhead-swinir-20261104-173035
image_name=basicsr-mlops-gpu:2.6.0-cu124
test -x "$environment_dir/venv/bin/python" || test -L "$environment_dir/venv/bin/python"
test -x "$environment_dir/python/bin/python3.11"
test -s "$pair_dir/train.txt"
test -s "$pair_dir/test.txt"
if docker container inspect "$container_name" >/dev/null 2>&1; then
    docker container inspect "$container_name" --format '{{.State.Status}}'
    exit 0
fi
docker image inspect "$image_name" --format '{{.Id}}' > "$run_dir/base_image_id.txt"
docker run -d --name "$container_name" --restart on-failure:3 \
  --gpus all --user 1003:1003 --shm-size 16gb \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --log-opt max-size=50m --log-opt max-file=5 \
  --env PYTHONPATH=/workspace --env PYTHONDONTWRITEBYTECODE=1 --env HOME=/tmp \
  --env BASICSR_TB_LOGGER_ROOT=/training_run/tb_logger \
  --env BASICSR_TRAIN_PYTHON=/home/huang/.local/share/venvs/basicsr/bin/python \
  --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=1 \
  --env TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 --env NCCL_DEBUG=WARN \
  --mount "type=bind,src=$environment_dir/venv,dst=/home/huang/.local/share/venvs/basicsr,readonly" \
  --mount "type=bind,src=$environment_dir/python,dst=/home/huang/.local/share/uv/python/cpython-3.11-linux-x86_64-gnu,readonly" \
  --mount "type=bind,src=$run_dir/code,dst=/workspace,readonly" \
  --mount "type=bind,src=$pair_dir,dst=/data,readonly" \
  --mount "type=bind,src=$run_dir,dst=/training_run" \
  --workdir /workspace --entrypoint /home/huang/.local/share/venvs/basicsr/bin/python \
  "$image_name" /training_run/control/pipeline.py
