# Dev Container 工作流

IDE 用 **Dev Container** 进环境；底座由 [`docker-compose.yml`](../docker-compose.yml) 提供（与 `scripts/docker-dev.sh`、GitLab CI 同一 PyTorch 镜像族 + `uv-bootstrap`）。

## 架构（三层，不是二选一）

| 层 | 位置 | 作用 |
|----|------|------|
| Docker 镜像 | `Dockerfile` → `pytorch:2.6.0-cuda12.4-devel` | OS、CUDA 用户态、uv |
| Python venv | `/workspace/.venv`（宿主机即仓库内 `.venv`，**已 gitignore**） | 项目依赖，与 CI 一致 |
| Dev Container | 本目录 `devcontainer.json` | 仅 IDE：解释器、扩展、UID 映射、`postCreateCommand` |

**venv 仍在仓库根目录**，但不提交 Git；容器内路径为 `/workspace/.venv`。不要再用 `.venv-huang` 或宿主机全局 pip。

## 一次性准备（宿主机）

```bash
# 写入 ~/.zshrc 或每次开 IDE 前 export
export DATA_MOUNT_SOURCE=/mnt/data
export HOME_MOUNT_SOURCE=/mnt/home   # 有 NFS 时
```

示例见 [`data-mount.env.example`](data-mount.env.example)。

## 日常开发

1. Cursor / VS Code：**Rebuild Container**（会跑 `postCreateCommand` → `uv-bootstrap.sh`）
2. 终端里直接用：`uv run python ...`（已在容器内，解释器指向 `.venv`）
3. Agent 对话：**在 Dev Container 窗口里** 提问、改代码、跑命令（推荐）

无 IDE 时等价入口：

```bash
export DATA_MOUNT_SOURCE=/mnt/data
bash scripts/docker-dev.sh shell    # 或 run uv run python ...
```

## 标准任务步骤

### 预处理（宿主机数据 → NFS 数据集）

在 **容器内** 或 `docker-dev.sh run`：

```bash
uv run python scripts/preprocess_cylinderblock_swinir.py \
  --data-root /data/cst_ai/data_0607 \
  --output-root /home/${USER}/cst_ai/datasets/swinir_cylinderblock_paired \
  --overwrite
```

### 预览 / 预训练对比

```bash
uv run python scripts/preview_cylinderblock_preprocess.py --slice 250
uv run python scripts/download_swinir_release.py --only 004_grayDN
uv run python scripts/compare_swinir_cylinder_pretrained_dn.py --slice 250
```

### 正式微调训练 → GitLab CI

1. 确认 NFS 上已有 `train/LQ`、`train/GT`、`val/...`、`normalization.json`
2. push 到 `main`，GitLab 手动 play **`train_cylinderblock_swinir_graydn`**
3. 产出：`/mnt/home/${USER}/cst_ai/ci_outputs/{PipelineID}/`

本地短调试可用 `docker-dev.sh run` + 单卡 `basicsr/train.py`，长跑仍建议 CI。

## 权限

- Dev Container：`remoteUser=vscode` + `updateRemoteUserUID=true` → 容器 uid 对齐宿主机，`.venv` 不会变成 root 文件
- `docker-dev.sh`：`docker compose run --user $(id -u):$(id -g)`

## 还能在「普通 shell」里用 Agent 吗？

可以，但 **Python/GPU 任务应通过容器**：

- ✅ **最佳**：Cursor **Dev Container** 窗口里和 Agent 对话（左下角显示 `Dev Container`）
- ✅ 宿主机 Agent 执行：`bash scripts/docker-dev.sh run uv run python ...`
- ❌ 避免：宿主机 `conda base` 里直接 `uv run`（易漂移；若 shell 里还有 `export UV_PROJECT_ENVIRONMENT=.venv-huang` 请 **unset**）

## 重建 venv

```bash
# 在容器内或 docker-dev.sh 中（root 删旧 venv 仅当曾被 root 占用时）：
docker run --rm --user root -v "$PWD:/workspace" basicsr-mlops-gpu:2.6.0-cu124 rm -rf /workspace/.venv /workspace/.venv-huang
bash scripts/docker-dev.sh bootstrap
unset UV_PROJECT_ENVIRONMENT
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
