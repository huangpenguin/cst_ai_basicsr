# Docker Compose 开发环境

与 [`.devcontainer/devcontainer.json`](../.devcontainer/devcontainer.json) 共用本文件；完整工作流见 [`.devcontainer/README.md`](../.devcontainer/README.md)。

## 前置条件

```bash
export DATA_MOUNT_SOURCE=/mnt/data
export HOME_MOUNT_SOURCE=/mnt/home   # 可选
```

## 入口

| 方式 | 命令 |
|------|------|
| **Cursor Dev Container** | Rebuild Container（推荐） |
| **无 IDE** | `bash scripts/docker-dev.sh shell` |

## 挂载

| 宿主机 | 容器 |
|--------|------|
| 仓库 | `/workspace` |
| `DATA_MOUNT_SOURCE` | `/data` |
| `HOME_MOUNT_SOURCE` | `/home` |

## Python 环境

- 容器内执行 `scripts/uv-bootstrap.sh` → **`/workspace/.venv`**
- 宿主机路径为仓库根 **`.venv`**（gitignore，不提交）
- 命令统一：`uv run python ...`

## 与 CI

- 同一 PyTorch 镜像族 + 同一 bootstrap 脚本
- CI job 不跑 compose；直接在 GitLab GPU 镜像里 bootstrap `.venv`
