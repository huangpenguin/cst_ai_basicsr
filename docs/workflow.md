# 项目工作流：配对 TIFF 灰度去噪

本仓库基于 BasicSR，针对工业 X 光 CT 重建切片（uint16 灰度 TIFF）做低剂量 → 高剂量的**监督去噪微调**。主模型为 SwinIR-M（官方 grayDN noise25 初始化），另有 TSFDAN 双分支 + LoRA 变体。

整体链路：

```text
原始扫描 ──organize──▶ 扁平 LQ/GT 对 ──prepare──▶ train/val 元数据
                                                    │
                          CI 训练 (basicsr/train.py) ◀┘
                                   │
                  固定评测集 eval_benchmark_v2 ──▶ 面板 / 指标表 ──▶ Gradio Demo
```

## 1. 目录速查

| 路径 | 内容 |
|------|------|
| `basicsr/` | 训练框架（含本项目新增的 `PairedTiffImageDataset`、`TSFDANSwinIR` arch/model 等） |
| `options/train/SwinIR/` | 本项目训练 YAML（`train_SwinIR_grayDN_*`、`train_TSFDAN_lora_*`） |
| `apps/denoise_demo/` | Gradio 去噪演示，见 [`apps/denoise_demo/README.md`](../apps/denoise_demo/README.md) |
| `scripts/data/` | 数据整理 / 校验划分 / 混合集构建；`folder_roles*.yaml` 为各数据集的目录角色规则 |
| `scripts/infer/` | TIFF 推理（SwinIR / TSFDAN）与共享 loader、分块推理 helper |
| `scripts/eval/` | 固定评测集构建、批量推理、填表、传统 CV 基线、面板增强 |
| `scripts/download/` | 预训练权重下载（SwinIR release、TSFDAN） |
| `scripts/ci/` | CI 专用：存储准备、YAML → env、续训种子、产物归档、TensorBoard |
| `scripts/env/` | 镜像构建、uv bootstrap、docker-dev、GPU 主机数据布局 |
| `scripts/upstream/` | BasicSR 上游原有脚本（`dist_train.sh`、`data_preparation/`、`metrics/`、`inference/` 等） |
| `tests/` | pytest（`tests/manual/` 为上游手动调试脚本，不参与 pytest） |
| `docs/upstream/` | BasicSR 上游原版文档 + Sphinx 配置 |

脚本规则：同一能力只保留一个入口脚本，需求变化时加 CLI flag，而不是复制 `*_v2.py`（见 `.cursor/rules/singular-scripts.mdc`）。

## 2. 存储布局

| GPU 宿主机 | CI 容器 | 用途 |
|------------|---------|------|
| `/mnt/data/cst_ai` | `/data/cst_ai` | 共享原始数据（只读） |
| `/mnt/home/huang/cst_ai` | `/home/huang/cst_ai` | 元数据、实验权重、日志、评测输出（可写） |

```text
/mnt/home/huang/cst_ai/
├── datasets/<name>/            # train.txt / val.txt / manifest.csv / normalization.json / dataset.json
├── experiments/<dataset>/<exp_name>/   # 正式训练权重（稳定路径，跨 pipeline 续训）
│   ├── models/net_g_*.pth
│   └── training_states/*.state
├── ci_outputs/<pipeline_id>/   # 每次 CI 的 tb_logger 与附属输出；latest -> 最近一次正式训练
├── outputs/eval_benchmark_v{1,2}/, eval_benchmark_cv/   # 固定评测
└── archive/cylinderblock_2673001900/   # 旧几何变换实验归档（只读）
```

不要把数据集或 checkpoint 写到 runner 的 `/cache`。

## 3. 数据准备

### 3.1 新数据集放置

```text
/mnt/data/cst_ai/datasets/<dataset_name>/
  LQ/*.tif
  GT/*.tif
```

LQ 与 GT 必须**同名、同尺寸、2D uint16、像素级已配准**。准备阶段不做旋转、缩放、裁剪或重采样。

原始扫描是嵌套目录时，先按角色规则整理成扁平硬链接对：

```bash
uv run python scripts/data/organize_paired_tiff_by_folder.py \
  --source-root <raw_root> \
  --roles-file scripts/data/folder_roles_<dataset>.yaml \
  --output-root /mnt/data/cst_ai/datasets/<dataset_name>
```

`folder_roles*.yaml` 描述哪个子目录是 GT、哪些是 LQ、要跳过哪些条件以及 `slice_ranges`（去掉首尾空白切片）。

### 3.2 校验与划分

只写小体积元数据，不复制、不修改图像：

```bash
uv run python scripts/data/prepare_paired_tiff_dataset.py \
  --source-root /mnt/data/cst_ai/datasets/<dataset_name> \
  --output-root /mnt/home/huang/cst_ai/datasets/<dataset_name> \
  --split-mode tail \
  --val-fraction 0.1
```

- `--dry-run` 只校验不写。
- `--split-mode random --seed 0` 仅在相邻文件名不是相邻切片时使用。
- 注意 `tail` 按字典序取尾部，可能导致 val 集中在单一电流组（见 [experiments.md](experiments.md) 中 v1 的教训）。3D 切片**必须按体数据划分**，相邻切片高度相关，随机按切片划分会造成数据泄漏。

### 3.3 已有数据集

| 数据集 | 说明 | 训练 YAML |
|--------|------|-----------|
| `aligned_drive` | Phantom；Z 过滤 5-1 275–623、5-2 185–740、5-3/5-4 185–745 | `train_SwinIR_grayDN_paired_tiff{,_bs8_30k}.yml` |
| `cylinder_260520_gt90ua` | GT=`1-110kV_90uA`，跳过 `0-300kV_100uA`，切片 130–750 | `train_SwinIR_grayDN_cylinder_gt90ua_bs8_30k.yml`（实际 bs=4） |
| `cylinder_260527_eval_gt60ua` | 仅评测 holdout，不参与训练 | — |
| `cylinder_phantom_mix` | 上两者硬链接合并（`cyl__` / `ph__` 前缀），domain×µA 分层划分；`scripts/data/build_cylinder_phantom_mix.py` | `train_SwinIR_grayDN_cylinder_phantom_mix_bs4_30k.yml`、`train_TSFDAN_lora_cylinder_phantom_mix_bs4_10k.yml` |
| `alcast_150kv_gt66ma` | GT=66mA，LQ=4/7/17/33mA，切片 70–820；每电流 8 test + 8 val 分层留出 | `train_SwinIR_grayDN_alcast_gt66ma_bs4_30k.yml` |

## 4. 训练

### 4.1 配置约定

- 训练语义（`name`、路径、`num_gpu`、端口、超参）全部写在 YAML 里；CI 只通过 `TRAIN_OPT` 选 YAML 并启动。
- `path.experiments_root` 指向稳定路径（如 `/home/huang/cst_ai/experiments/<dataset>/`），这样 `--auto_resume` 能跨 pipeline 生效。
- 默认像素损失为 **MSE**（直接优化 PSNR/SNR 相关的误差功率）。做对照只改一处：`--force_yml "train:pixel_opt:type=L1Loss"` 或 `CharbonnierLoss`。
- 归一化用固定物理范围（`normalization.json`，通常 uint16 0–65535 → [0,1]），**不要**用逐图百分位拉伸做定量比较。
- 32GiB V100 上 SwinIR-M P128 每卡 bs=8 会 OOM，使用 bs=4。
- ALcast 与 TSFDAN YAML 设 `val: ~`（不跑在线 DDP val），正式指标只在训练后的评测 job 里产出。

### 4.2 本地调试

```bash
uv run python scripts/download/download_swinir_release.py --only noise25
uv run python basicsr/train.py \
  -opt options/train/SwinIR/train_SwinIR_grayDN_paired_tiff.yml \
  --force_yml "num_gpu=1"
```

### 4.3 GitLab CI（正式训练）

| Job（均为手动） | 作用 |
|-----------------|------|
| `gpu_smoke` | 验证 GPU / 数据挂载 / 依赖 |
| `train_paired_tiff_swinir_debug` → `train_paired_tiff_swinir` | SwinIR-M；YAML 由 `.gitlab/ci/train.yml` 的 `TRAIN_OPT` 指定（当前为 ALcast） |
| `train_tsfdan_lora_mix_debug` → `train_tsfdan_lora_mix` → `eval_tsfdan_lora_mix` | TSFDAN+LoRA（`.gitlab/ci/train_tsfdan.yml`，不影响默认 `TRAIN_OPT`） |

CI 流程：`scripts/ci/ci_emit_opt_env.py` 把 YAML 转成环境变量 → `scripts/ci/ci_storage.py prepare` 准备输出目录 → `scripts/ci/ci_seed_resume.py` 把以往 `ci_outputs` 中最新的 `*.state` + `net_g_*.pth` 复制到稳定实验目录 → `torchrun basicsr/train.py --auto_resume`。

### 4.4 断点续训

`training_states/` 下有 `.state` 即可续训：

```bash
uv run python basicsr/train.py -opt <yaml> --auto_resume
```

TensorBoard 曲线从约 N 步开始而不是 0，通常就是续训；如果本意是从头训，需要清掉同名 `tb_logger/<exp_name>/` 中的旧 event 文件。

### 4.5 TensorBoard

- CI 训练 job 自动在 6006 端口启动 TensorBoard（`scripts/ci/ci_tensorboard.sh`，logdir 为本次 run 的 `tb_logger/`），`after_script` 自动停止。runner 需 `network_mode = "host"`。
- Mac 查看：`ssh -L 6006:localhost:6006 <user>@<gpu-host>`，然后打开 `http://localhost:6006`。
- 训练结束后重新查看：

```bash
bash scripts/ci/serve_tensorboard.sh /mnt/home/huang/cst_ai/ci_outputs/latest/tb_logger
```

- 只看文本日志：`tail -f /mnt/home/huang/cst_ai/ci_outputs/latest/experiments/*/train_*.log`。
- wandb 可选（设置 `WANDB_API_KEY`）。

## 5. 评测

### 5.1 单模型 TIFF 推理

生成对比面板，给定 GT 时还会写 `metrics.csv`（LQ/输出 SNR、SNR 增益、PSNR、SSIM）：

```bash
uv run python scripts/infer/infer_swinir_graydn_tiff.py \
  --model /mnt/home/huang/cst_ai/experiments/<dataset>/<exp>/models/net_g_latest.pth \
  --baseline-model experiments/pretrained_models/SwinIR/004_grayDN_DFWB_s128w8_SwinIR-M_noise25.pth \
  --lq-dir /mnt/data/cst_ai/datasets/<dataset>/LQ \
  --gt-dir /mnt/data/cst_ai/datasets/<dataset>/GT \
  --meta-info /mnt/home/huang/cst_ai/datasets/<dataset>/val.txt \
  --pick-indices 0 202 405 608 \
  --normalization /mnt/home/huang/cst_ai/datasets/<dataset>/normalization.json \
  --output-dir /mnt/home/huang/cst_ai/outputs/<dataset>_preft_vs_ft
```

带 `--baseline-model` 时，每张 `*_compare.png` 为 `LQ | PreFT | FT | GT`。TSFDAN 用 `scripts/infer/infer_tsfdan_tiff.py`（参数相同）。默认 `--tile 256 --tile-overlap 32`。

### 5.2 固定评测集

| 能力 | 脚本 |
|------|------|
| 建评测树（泄漏优先） | `scripts/eval/build_eval_benchmark.py --output-root …` |
| 单模型跑全部列 | `scripts/eval/run_eval_benchmark_model.sh <root> <ckpt> <tag> [device] [preft] [col ...]` |
| TSFDAN 版本 | `scripts/eval/run_eval_benchmark_tsfdan.sh`（参数同上） |
| 填对比表 | `scripts/eval/fill_eval_benchmark_table.py --root … --preset v1\|v2` |
| 传统 CV 基线 | `scripts/eval/run_eval_benchmark_cv.py`（复用 v2 pairs，不拷贝图像） |
| 面板增强（EPI + 残差图 + 大字号指标，不重推理） | `scripts/eval/enrich_eval_benchmark_panels.py`；`--mode singles-fixed` 输出 4 张共享 `[0,1]` 窗口的 PPT 用 PNG |

v2 评测根目录：`/mnt/home/huang/cst_ai/outputs/eval_benchmark_v2/`，五列：`battery_260520`、`battery_260527`、`cylinder_260520_val20`、`cylinder_260527_holdout20`、`phantom_aligned_drive_val20`。

### 5.3 如何判断效果

- PSNR / SSIM / SNR 只作参考：指标高不代表条纹、环形伪影已消失，也可能是过度平滑。
- 最终标准是**肉眼看图**：极片、孔边缘不能被抹糊，也不能凭空长出结构（hallucination）。
- CNR 暂未做（没有缺陷 ROI 标注）。

## 6. 演示应用

```bash
uv run python apps/denoise_demo/app.py   # 0.0.0.0:7860
```

模型注册表在 `apps/denoise_demo/models.yaml`，详见 [`apps/denoise_demo/README.md`](../apps/denoise_demo/README.md)。
