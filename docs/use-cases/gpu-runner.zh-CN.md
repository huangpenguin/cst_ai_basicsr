# GPU Runner Workflow

GitLab 网页派发 job，执行在带 `linux,docker,gpu` tag 的 self-hosted GPU Docker Runner 上。

## 数据约定（团队统一）

```text
/mnt/data/<PROJECT_NAME>/              原始 / 公共数据（只读）
/mnt/home/<user>/<PROJECT_NAME>/       个人输出、脚本衍生结果（可写）
```

容器内对应：

```text
/data/<PROJECT_NAME>/...               输入（DATA_DIR + PROJECT_NAME）
/home/<user>/<PROJECT_NAME>/...        输出（WORK_DIR + PROJECT_NAME）
```

**换项目**只改 `PROJECT_NAME` 和 `DATASET_DIR`；**换用户** CI 自动用 `GITLAB_USER_LOGIN`，无需改 Runner。

## Runner config.toml（全员共用，一次配置）

```toml
[runners.docker]
  gpus = "all"
  shm_size = 17179869184
  volumes = [
    "/cache",
    "/mnt/data:/data:ro",
    "/mnt/home:/home:rw",
  ]
```

```bash
sudo gitlab-runner restart
```

不要写 `/mnt/home/huang` 这种个人路径 — 挂整个 `/mnt/home`，容器里每人用自己的 `/home/<username>`。

## 环境变量

| 变量 | 示例 | 说明 |
|------|------|------|
| `DATA_DIR` | `/data` | 公共 NFS 根（只读） |
| `PROJECT_NAME` | `cst_ai` | 项目目录名 |
| `WORK_DIR` | `/home/huang`（dev）或 CI 自动 `/home/$GITLAB_USER_LOGIN` | 个人 home |
| `DATASET_DIR` | `/data/cst_ai/data_0607/.../260520_battery image tiff data` | 输入目录（可选，有默认值） |
| `OUTPUT_DIR` | `/home/huang/cst_ai/ci_outputs/$CI_PIPELINE_ID/...` | 输出目录（CI 默认已配） |

### `DATASET_DIR` 调用链

```text
DATASET_DIR → scripts/data_paths.py :: resolve_dataset_dir()
           → scripts/ci_gpu_smoke.py
           → scripts/smoke_test_swinir_gray_dn_battery.py
```

未设置时：在 `/data/<PROJECT_NAME>/` 下按相对路径查找。

## 本地 devcontainer

```bash
export DATA_MOUNT_SOURCE=/mnt/data
export HOME_MOUNT_SOURCE=/mnt/home
export WORK_DIR=/home/huang    # 其他同事改成 /home/alice 等
export PROJECT_NAME=cst_ai
```

Rebuild Container。

## 结果查看

- GitLab Job **Artifacts**（14 天）
- 宿主机：`/mnt/home/<user>/cst_ai/ci_outputs/<pipeline_id>/...`

## 注意

- Docker bind mount 左侧必须是 GPU 宿主机上 autofs 已挂载的本地路径。
- 原始数据备份在 `/mnt/data/cst_ai/`；不要在 `/mnt/data` 下建个人子目录。
