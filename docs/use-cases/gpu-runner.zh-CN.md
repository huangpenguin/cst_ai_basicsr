# GPU Runner Workflow

GitLab 网页负责派发 job，真正执行发生在带 `linux`, `docker`, `main_gpu`, `storage_huang` tag 的 project runner 上（gpu01 上与公共 runner 并存）。Runner 拉起 GPU job 容器并以 **huang uid** 运行，以便在 NFS `root_squash` 下写入 `/home/huang/cst_ai`。

要求：

- 项目专用 runner（gpu01 注册步骤见 `scripts/setup_gpu_data_layout.sh`）
- tag 与 `.gitlab/ci/train.yml` 一致：`linux`, `docker`, `main_gpu`, `storage_huang`
- executor 使用 `docker`
- `[runners.docker] user = "1003:1003"`（或 `id -u huang` / `id -g huang`）
- GPU 宿主机已安装 NVIDIA driver、Docker 和 NVIDIA Container Toolkit
- GitLab Runner 的 `config.toml` 中设置 `[runners.docker] gpus = "all"`
- 为了 Mac 上 `ssh -L 6006:localhost:6006` 能访问 CI 内自动起的 TensorBoard，请在同一 runner 设置：
  `[runners.docker] network_mode = "host"`（或等价地把容器 `6006` 映射到宿主机）。训练 job 结束会在 `after_script` 里自动 `ci_tensorboard.sh stop` 回收进程。
- 数据集与产出通过 volumes 挂载：`/mnt/home:/home:rw`，`/mnt/data:/data:ro`

## 服务器端注册（storage_huang）

**不要**修改现有公共 `main_gpu` runner 的 `user=`。在 gpu01 上 **新增** 第二个 `[[runners]]`：

1. GitLab → Project → Settings → CI/CD → Runners → New project runner  
   Tags: `linux`, `docker`, `main_gpu`, `storage_huang`；关闭 Run untagged jobs。
2. 在 gpu01 执行 `scripts/setup_gpu_data_layout.sh` 查看 register 命令与 `config.toml` 片段
3. 编辑新 runner 的 `[runners.docker]`：`user`、`gpus`、`volumes`（见脚本输出）
4. `sudo gitlab-runner verify && sudo gitlab-runner restart`
5. 验证写权限：

```bash
docker run --rm --user 1003:1003 -v /mnt/home:/home:rw alpine \
  touch /home/huang/cst_ai/ci_outputs/.runner_write_test
```

也可运行 `scripts/setup_gpu_data_layout.sh` 打印 uid/gid 与 volumes 片段。

## 服务器端注册（仅 smoke / 无 NFS 写需求时）

若只需 GPU smoke、不写个人 home，可继续使用仅带 `main_gpu` 的公共 runner（容器默认 root，**不能**写 `/home/huang/cst_ai`）。

触发方式：

- GitLab Pipeline 页面先手动 Play `gpu_smoke`
- `gpu_smoke` 通过后，再手动 Play `run_training`
- commit message 含 `[run train]`

`gpu_smoke` 用于快速验证一个仓库是否具备最小 DL GPU job 环境：

```text
GitLab Runner docker executor
→ pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel job container
→ bash scripts/uv-bootstrap.sh
→ run ${SMOKE_COMMAND}
```

新仓库即使没有 `pyproject.toml`、`uv.lock` 或 `requirements.txt`，也可以先使用默认 PyTorch/CUDA 镜像 + 模板 `train.py` 完成 smoke test。后续需要加包时，应提交依赖文件，或切换为已经构建并推送到 registry 的项目镜像；不要安装到 GPU Runner 宿主机。

`run_training` 会以前台 GitLab job 形式运行训练命令；CI 日志就是训练日志，job 状态就是训练状态。

## 默认镜像与 uv

默认：

```yaml
MLOPS_GPU_IMAGE: "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"
```

该镜像不保证内置 `uv`，模板会在 job 内安装 `uv` 并执行 `scripts/uv-bootstrap.sh`：

1. `pyproject.toml` + `uv.lock`：`uv sync --frozen --dev`
2. 半迁移（空 `dependencies` + `requirements.txt`）：requirements + cu124 index + dev 工具
3. 只有 `requirements.txt`：`uv venv` + requirements（含 torch 时 cu124 index）+ 可选 editable install
4. 只有 `pyproject.toml`：`uv sync --dev`
5. 没有环境文件：直接使用基础 PyTorch 镜像

## 配置数据挂载

Docker executor 的数据挂载由 Runner 服务器上的 `config.toml` 控制，不由 `.gitlab-ci.yml` 动态设置。

模板默认在 job 内使用：

```yaml
DATA_DIR: "/data"
```

如果训练需要数据集，应在 GPU Runner 宿主机上配置：

```toml
volumes = ["/cache", "/mnt/data:/data:ro", "/mnt/home:/home:rw"]
```

训练产出与数据集路径：`/home/<CST_STORAGE_USER>/cst_ai/...`（容器内）=
`/mnt/home/<CST_STORAGE_USER>/cst_ai/...`（宿主机 NFS）。**不要**把数据集或 checkpoint 写到 runner `/cache`。

### NFS root_squash 与 `user =`

NFS 默认 **root_squash**：CI 容器以 `root` 运行时无法写入 `/home/huang/cst_ai`（属主为 huang）。
症状：`ci_storage.py` 报错 `not writable: /home/huang/cst_ai`。

**推荐**：为本项目注册带 `storage_huang` tag 的专用 runner，在 `[runners.docker]` 设置：

```toml
user = "1003:1003"   # id -u huang; id -g huang
```

完整 `config.toml` 片段见 `scripts/setup_gpu_data_layout.sh`。然后 `sudo gitlab-runner restart`。

这里的 `/mnt/data` 是 **GPU Runner 所在宿主机** 上的路径，也就是 Docker daemon 能看到的本地路径。

常见例子：

- autofs：`/mnt/data:/data:ro`
- 手动 NFS 挂载：`/mnt/nfs_data:/data:ro`
- 本机数据盘：`/data/datasets:/data:ro`

不要把 NFS 服务器 export 字符串写进去，例如 `192.168.3.14:/mnt/storage/data`。Docker bind mount 的左侧必须是宿主机本地路径。
