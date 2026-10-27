# cst_ai / BasicSR：工业 CT 灰度去噪

基于 [BasicSR](https://github.com/XPixelGroup/BasicSR) 的工业 X 光 CT 切片去噪项目。输入低剂量（低管电流）uint16 灰度 TIFF，输出接近高剂量参考的去噪结果。

- **模型**：SwinIR-M（官方 grayDN noise25 初始化）配对微调；TSFDAN 双分支 + LoRA 变体。
- **数据**：像素级已配准的同名 LQ/GT TIFF 对（battery、cylinder、phantom、ALcast）。
- **流程**：整理数据对 → 校验划分 → GitLab CI 多卡训练 → 固定评测集出面板和指标表 → Gradio 演示。

## 快速开始

```bash
uv sync --dev

# 下载预训练权重
uv run python scripts/download/download_swinir_release.py --only noise25
uv run python scripts/download/download_tsfdan_pretrain.py

# 校验并划分一个配对数据集（只写元数据）
uv run python scripts/data/prepare_paired_tiff_dataset.py \
  --source-root /mnt/data/cst_ai/datasets/<name> \
  --output-root /mnt/home/huang/cst_ai/datasets/<name> \
  --split-mode tail --val-fraction 0.1

# 本地单卡调试训练（正式训练走 GitLab CI）
uv run python basicsr/train.py \
  -opt options/train/SwinIR/train_SwinIR_grayDN_paired_tiff.yml --force_yml "num_gpu=1"

# 推理并生成 LQ | PreFT | FT | GT 对比面板
uv run python scripts/infer/infer_swinir_graydn_tiff.py --model <ckpt> \
  --baseline-model experiments/pretrained_models/SwinIR/004_grayDN_DFWB_s128w8_SwinIR-M_noise25.pth \
  --lq-dir <LQ> --gt-dir <GT> --output-dir <out>

# 去噪演示（0.0.0.0:7860）
uv run python apps/denoise_demo/app.py
```

## 目录结构

```text
apps/denoise_demo/   Gradio 去噪演示
basicsr/             训练框架（含 PairedTiffImageDataset、TSFDANSwinIR 等项目扩展）
options/train/       训练 YAML（项目配置在 SwinIR/ 下）
scripts/
  data/              数据整理、校验划分、混合集构建、folder_roles*.yaml
  infer/             TIFF 推理（SwinIR / TSFDAN）
  eval/              固定评测集、批量推理、填表、传统 CV 基线、面板增强
  download/          预训练权重下载
  ci/                CI 存储准备、续训、TensorBoard
  env/               镜像构建、uv bootstrap、docker-dev
  upstream/          BasicSR 上游原有脚本
tests/               pytest（tests/manual/ 为上游手动脚本）
docs/                项目文档；docs/upstream/ 为 BasicSR 原版文档
```

`datasets/`、`experiments/`、`results/`、`tb_logger/` 为本地运行产物目录，已 gitignore。

## 配置

| 位置 | 作用 |
|------|------|
| `options/train/SwinIR/*.yml` | 训练的全部语义：数据路径、`name`、`num_gpu`、超参、`experiments_root` |
| `.gitlab/ci/train.yml` 中 `TRAIN_OPT` | CI 默认训练哪个 YAML |
| `.gitlab/ci/train_tsfdan.yml` 中 `TSFDAN_TRAIN_OPT` | TSFDAN 训练 YAML |
| `apps/denoise_demo/models.yaml` | Demo 可选的模型列表 |
| `DATA_DIR`、`CST_STORAGE_USER`、`DATA_MOUNT_SOURCE`、`HOME_MOUNT_SOURCE` | 数据挂载与存储用户，见 [docs/infrastructure.md](docs/infrastructure.md) |

## 文档

- [docs/workflow.md](docs/workflow.md)：数据准备、训练、评测、演示的完整操作流程
- [docs/experiments.md](docs/experiments.md)：实验记录与结果
- [docs/infrastructure.md](docs/infrastructure.md)：Docker / Dev Container / GitLab Runner / 代码质量
- [docs/research-notes.md](docs/research-notes.md)：数据特性、改进方向与开源方案调研
- [docs/upstream/](docs/upstream/)：BasicSR 上游 README 与文档

本项目沿用 BasicSR 的 [Apache 2.0 许可](LICENSE.txt)，引用信息见 `CITATION.cff`。
