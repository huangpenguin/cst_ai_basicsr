# 基础设施：镜像、开发容器、GPU Runner 与代码质量

本仓库由 `init-ai` 模板的 core / python-quality / ci-quality / mlops-gpu 四个 pack 注入了 Cursor 规则、Ruff/Pyright、GitLab CI 和 GPU Docker 环境。本文合并了原先分散的 Docker、Dev Container、GPU Runner、数据挂载与 pack 说明。

## 1. 统一 GPU 镜像

Dev Container 与 GitLab CI 使用**同一个**镜像 `basicsr-mlops-gpu:2.6.0-cu124`，由根目录 `Dockerfile` 构建：

| 层 | 内容 |
|----|------|
| 镜像 | `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel` + uv/curl/git，**不含**项目 Python 包 |
| Python 环境 | `scripts/env/uv-bootstrap.sh` → 仓库根 `.venv`（容器内 `/workspace/.venv`，已 gitignore） |
| 依赖声明 | `pyproject.toml` + `uv.lock`（torch 走 cu124 index）；`requirements.txt` 仅供 `setup.py` editable 安装使用 |

gpu01 上首次（或改 Dockerfile 后）构建：`bash scripts/env/build_mlops_image.sh`；CI 使用 `pull_policy: if-not-present`。

`uv-bootstrap.sh` 行为：

| 项目状态 | 行为 |
|----------|------|
| `pyproject.toml` + `uv.lock`（当前状态） | `uv sync --frozen --dev`，再 `uv pip install -e .` |
| 半迁移（`dependencies = []` + `requirements.txt`） | 从 requirements 装运行时依赖（cu124）+ dev 工具 |
| 仅 `requirements.txt` | `uv venv` + requirements |
| 仅 `pyproject.toml` | `uv sync --dev` |

原则：

- 不要把项目依赖装到 GPU Runner 宿主机；宿主机只负责 Docker、NVIDIA driver/Container Toolkit、GitLab Runner 和数据盘挂载。
- 需要系统包时在 Dockerfile 里 `apt-get install`（如 `libgl1`、`libglib2.0-0`、`ffmpeg`），只装真实需要的。
- 若要把 Python 依赖固化进镜像，venv 放 `/opt/mlops-venv` 而不是 `/workspace/.venv`（后者会被运行时 bind mount 覆盖）。

### CUDA / driver / torch 版本

| 层 | 默认 |
|----|------|
| 镜像基底 | `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel` |
| PyTorch | `torch 2.6.x + cu124`，cuDNN 9.x |
| 已验证 GPU | V100 32GB（`sm_70`） |
| 宿主机 | Ubuntu 24.04 + NVIDIA `570-server` 或更新 |

规则：宿主机 driver 必须支持容器内 CUDA runtime（driver 新可以跑旧 CUDA 容器）。依赖中 torch 保持 `>=2.6.0,<2.7.0`，torchvision 跟随同一 minor，避免 resolver 把 `.venv` 升到未验证的版本。升级前先单独验证：

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## 2. 开发环境

### 2.1 Dev Container（推荐）

宿主机一次性（写进 `~/.zshrc`）：

```bash
export DATA_MOUNT_SOURCE=/mnt/data
export HOME_MOUNT_SOURCE=/mnt/home
```

然后在 Cursor/VS Code 中 **Rebuild Container**（`postCreateCommand` 会运行 `uv-bootstrap.sh`）。容器内直接 `uv run python ...`。详细步骤和权限说明见 [`.devcontainer/README.md`](../.devcontainer/README.md)。

仅做 GPU smoke、不需要真实数据时：

```bash
export DATA_MOUNT_SOURCE="${HOME}/.local/share/mlops-empty-data"
mkdir -p "$DATA_MOUNT_SOURCE"
```

### 2.2 无 IDE：docker compose

`docker-compose.yml` 与 Dev Container 共用：

```bash
bash scripts/env/docker-dev.sh shell                 # 交互 shell
bash scripts/env/docker-dev.sh run uv run python ...  # 单条命令
bash scripts/env/docker-dev.sh bootstrap             # 只跑 uv-bootstrap
```

| 宿主机 | 容器 |
|--------|------|
| 仓库 | `/workspace` |
| `DATA_MOUNT_SOURCE` | `/data` |
| `HOME_MOUNT_SOURCE` | `/home` |

compose 默认没有 `ports:`，需要从浏览器访问 Gradio（7860）或 TensorBoard（6006）时，加 `-p`、`ports:` 或 `network_mode: host`。

### 2.3 数据路径契约

宿主机路径、容器路径、应用代码三层分离：

| 层 | 提交到 Git | 每人 / 每环境 |
|----|------------|---------------|
| 容器挂载目标 | `/data` | — |
| 容器环境变量 | `DATA_DIR=/data` | — |
| 宿主机 bind 源 | `${localEnv:DATA_MOUNT_SOURCE}` | Rebuild 前 export |
| 应用代码 | 读 `DATA_DIR` + 相对路径 | 可选 per-dataset env override |

- bind 源必须是 **Docker 宿主机本地路径**（autofs 下 `ls /mnt/data` 可触发挂载），不能写 NFS export 字符串如 `192.168.x.x:/...`。
- 不要在已提交的 JSON / Python 里写 `/home/...` 等个人宿主机路径；不要用 Dockerfile 做数据挂载；不要依赖指向 workspace 外的 symlink。
- helper：`scripts/data/data_paths.py`（`data_root()`、`resolve_dataset_dir(..., env_override=...)`；解析顺序为 env override → `DATA_DIR` → `datasets/` 旧路径）。

环境变量：`DATA_DIR`、`PROJECT_NAME=cst_ai`、`WORK_DIR`、`CST_STORAGE_USER`（默认 `huang`）、可选 `DATASET_DIR`、`OUTPUT_DIR`。

### 2.4 代码位置与多机使用

- **唯一工作副本**在 NFS：`/mnt/home/huang/code/cst_ai/BasicSR`。gpu01（训练）和 demo 机（Gradio）挂载同一路径、共用这一份代码；gpu01 上的 `/home/huang/code/cst_ai/BasicSR` 是指向它的 symlink。
- **venv 不放 NFS**（venv 不可重定位，且 NFS 上小文件 IO 慢）。每台机器各自在本地盘建 venv，`~/.zshrc` 里按目录限定：

  ```zsh
  # cst_ai/BasicSR lives on NFS; keep its uv venv on this host's local disk.
  _basicsr_uv_env() {
    local venv="$HOME/.local/share/venvs/basicsr"
    case "$PWD/" in
      /mnt/home/huang/code/cst_ai/BasicSR/*|/home/huang/code/cst_ai/BasicSR/*)
        export UV_PROJECT_ENVIRONMENT="$venv" ;;
      *)
        [[ "${UV_PROJECT_ENVIRONMENT:-}" == "$venv" ]] && unset UV_PROJECT_ENVIRONMENT ;;
    esac
  }
  autoload -Uz add-zsh-hook
  add-zsh-hook chpwd _basicsr_uv_env
  _basicsr_uv_env
  ```

  然后在仓库里 `uv sync --frozen --dev`。uv 缓存用默认的 `~/.cache/uv`。如果某台机器的 `$HOME` 本身就在 NFS 上，把 `venv` 和 `UV_CACHE_DIR` 改到本地盘，例如 `/var/tmp/$USER/`。
- **Demo 机启动**：

  ```bash
  tmux new -s denoise-demo
  cd /mnt/home/huang/code/cst_ai/BasicSR
  uv run --frozen python apps/denoise_demo/app.py
  ```

  需要 NVIDIA driver ≥ 550（cu124 wheel）；启动日志里各 preset 的 ckpt 应为 `exists: true`。
- **训练**仍然通过 gpu01 上的 GitLab CI 跑，CI 自己 clone 代码，跟工作副本放在哪里无关。训练产物、数据集元数据、评测输出都在 `/mnt/home/huang/cst_ai/`。
- **Git remotes**：`gitlab`（主仓库，`master` 跟踪 `gitlab/master`）、`github`（私有镜像 `huangpenguin/cst_ai_basicsr`）、`all`（一次 push 同时推两边：`git push all master`）、`upstream`（XPixelGroup/BasicSR，只 fetch）。
- **同一时间只在一台机器上做 git 写操作**（commit / checkout / rebase），否则 NFS 上的 `.git/index.lock` 会冲突。

## 3. GitLab CI 与 GPU Runner

### 3.1 Pipeline 结构

`.gitlab-ci.yml` 只做编排，job 拆在 `.gitlab/ci/`：

| 文件 | 内容 |
|------|------|
| `quality.yml` | `uv run ruff check .`、`ruff format --check .`、`pyright` |
| `train.yml` | `gpu_smoke`、SwinIR-M debug/正式训练 |
| `train_tsfdan.yml` | TSFDAN+LoRA 训练与评测 |

pipeline 在 MR、`main`、`master` 上触发；训练 job 均为手动 Play。GitLab `cache:` 只复用 `.venv/`、`.cache/uv/` 和预训练权重。

### 3.2 Runner 要求

训练 job 跑在 gpu01 上的项目专用 runner：

- tags：`linux`, `docker`, `main_gpu`, `storage_huang`；关闭 Run untagged jobs。
- `executor = "docker"`，`[runners.docker]` 中设置：
  - `gpus = "all"`
  - `user = "1003:1003"`（`id -u huang` / `id -g huang`）：NFS 是 root_squash，容器以 root 运行会写不进 `/home/huang/cst_ai`（症状：`ci_storage.py` 报 `not writable`）。
  - `volumes = ["/cache", "/mnt/data:/data:ro", "/mnt/home:/home:rw"]`
  - `network_mode = "host"`：CI 内自动启动的 TensorBoard 才能通过 SSH 转发访问。
- **不要**修改现有公共 `main_gpu` runner 的 `user=`，而是新增第二个 `[[runners]]`。

注册步骤：

1. GitLab → Project → Settings → CI/CD → Runners → New project runner（填上述 tags）。
2. 在 gpu01 运行 `bash scripts/env/setup_gpu_data_layout.sh`，按输出执行 register 并粘贴 `config.toml` 片段。
3. `sudo gitlab-runner verify && sudo gitlab-runner restart`。
4. 验证写权限：

```bash
docker run --rm --user 1003:1003 -v /mnt/home:/home:rw alpine \
  touch /home/huang/cst_ai/ci_outputs/.runner_write_test
```

只跑 smoke、不写个人 home 时，可以用只带 `main_gpu` tag 的公共 runner。

## 4. 代码质量

| 阶段 | 工具 | 配置 |
|------|------|------|
| `git commit` | Ruff lint + format（自动修复） | `.pre-commit-config.yaml` |
| CI | Ruff + Pyright（basic） | `ruff.toml`、`pyrightconfig.json` |

- Ruff 只检查 `ruff.toml` 中 `include` 列出的本项目文件；上游 BasicSR 模块不在范围内。
- 首次启用 hook：`uv sync --dev && uv run pre-commit install`。
- 依赖只用 `uv add` / `uv add --dev` 添加，命令统一 `uv run ...`。

## 5. Agent 规则与项目记忆

| 位置 | 内容 |
|------|------|
| `.cursorrules`、`.cursor/rules/*.mdc` | Agent 行为、uv、语言、单一脚本等规则 |
| `.cursor/project-context/decisions.md` | 稳定的路径与架构决策（英文、短条目） |
| `.cursor/lessons-learned/` | 可复用的故障复盘（CI 超时、TensorBoard setuptools、bs=8 OOM 等） |
