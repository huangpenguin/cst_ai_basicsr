# cst_ai / BasicSR：工业 CT 灰度去噪

基于 [BasicSR](https://github.com/XPixelGroup/BasicSR) 的工业 X 光 CT 切片去噪项目。输入低剂量（低管电流）uint16 灰度 TIFF，输出接近高剂量参考的去噪结果。

- **模型**：SwinIR-M（官方 grayDN noise25 初始化）配对微调；TSFDAN 双分支 + LoRA 变体。
- **数据**：同名 LQ/GT 灰度 TIFF 对（battery、cylinder、phantom、ALcast、Cylinderhead）；各批次的配准情况见实验记录。
- **流程**：整理数据对 → 审核配对与划分 → 手动提交 Vertex 训练 → 固定评测集验证 → 发布到现有 Vertex 推理入口。

## 当前部署入口

去噪页面：http://192.168.3.11:7861 。扫描上传与配对审核：http://192.168.3.11:7862 。
上传门户支持按拍摄条件提交 TIFF/ZIP、断点续传、人工审核与批准后同步 GCS；详见
[同事上传与管理员审核操作](docs/deploy/xray-portal-operations-20261119.md)。在 work1 管理：

```bash
cd /mnt/home/huang/code/cst_ai/BasicSR
make -f deploy/vertex/Makefile ui-status
make -f deploy/vertex/Makefile ui-check
```

页面依赖位于 work1 Docker 镜像内（Python 3.11 / Torch 2.6 / Gradio 6.25），
不要求给宿主 Python 安装 Torch。GPU 推理和训练运行在 Vertex。
详见 [当前运行与界面整合](docs/deploy/work1-ui-consolidation-20261118.md)。

## 源码开发与历史本地工作流（不是当前服务启动步骤）

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

# 训练：在 gpu01 上按 YAML 直接启动。单卡调试把 num_gpu 改成 1
uv run python basicsr/train.py \
  -opt options/train/SwinIR/train_SwinIR_grayDN_paired_tiff.yml --force_yml "num_gpu=1"

# 推理并生成 LQ | PreFT | FT | GT 对比面板
uv run python scripts/infer/infer_swinir_graydn_tiff.py --model <ckpt> \
  --baseline-model experiments/pretrained_models/SwinIR/004_grayDN_DFWB_s128w8_SwinIR-M_noise25.pth \
  --lq-dir <LQ> --gt-dir <GT> --output-dir <out>

# 当前内网 CPU → Vertex 页面（http://192.168.3.11:7861）
make -f deploy/vertex/Makefile ui-status
make -f deploy/vertex/Makefile ui-check
```

## 项目要点：uint16 预处理与单通道模型

以下为 2026-11-04～05 Cylinderhead mean3 训练的实际处理方式，读取实现见
[`PairedTiffImageDataset`](basicsr/data/paired_tiff_image_dataset.py)。

1. **磁盘数据保持 uint16**：输入为 1518×1518 的单通道 TIFF。200 µA 的三次扫描按相同断面编号取算术平均，先提升为 uint32 求和，再按 `(a + b + c + 1) // 3` 四舍五入保存为 uint16 GT。低电流 LQ 保持原始像素；本次未做空间配准修正或强度匹配。
2. **读取后转 float32，再归一化**：LQ、GT 使用同一组固定参数 `raw_min=0`、`raw_max=65535`：

   ```python
   x = image.astype(np.float32)
   x = np.clip((x - 0.0) / 65535.0, 0.0, 1.0)
   # H×W → H×W×1 → 1×H×W；组成 batch 后为 N×1×H×W
   # N 是一批图片的数量：本次每 GPU 为 4，四卡全局 batch 为 16
   ```

   这些参数记录在数据集的 `normalization.json` 中；本次没有逐图 min-max、均值/标准差或 ImageNet 归一化，也没有先转成 uint8 或除以 255。
3. **原生灰度基模**：使用官方 `004_grayDN_DFWB_s128w8_SwinIR-M_noise25.pth`，在本次实验中复制为 `pretrained/swinir_noise25.pth`。这是灰度高斯去噪模型（σ=25），配置为 `in_chans: 1`、`img_range: 1.0`、`upscale: 1`。已检查初始权重：`conv_first.weight` 为 `[180, 1, 3, 3]`，`conv_last.weight` 为 `[1, 180, 3, 3]`；输入输出均为单通道，无需将灰度复制为 RGB，也没有改造 RGB 卷积权重。模型内部的灰度 mean 为 0。卷积权重按 `[输出通道, 输入通道, 核高, 核宽]` 排列：首层将 1 个灰度通道映射为 180 个特征通道，末层再将 180 个特征通道重建为 1 个灰度通道。
4. **训练与推理**：训练使用 float32 张量，LQ/GT 同步随机裁剪 128×128、翻转和旋转。随机裁剪用于控制反向传播显存，并在不同步骤采样不同局部；验证和 Demo 对整图分块后拼回原尺寸。训练进行全参数监督微调（MSE，非 LoRA）。四张 V100 32 GB 通过 DDP 训练，每卡 batch=4，全局 batch=16，共 30,000 步。Demo 和最终测试同样以 float32、`x/65535` 读入 TIFF；Demo 的 CUDA 快速推理使用混合精度，并将 RSTB 块保持为 float32 以避免溢出。
5. **保存与显示分开**：Demo 的 `tiff16` 下载将结果裁剪到 `[0,1]`、乘 65535、四舍五入后保存为 uint16；网页预览转换为 uint8 仅用于显示。

本次独立 test 共 2,020 对，未参与训练或模型选择；训练结束后，对预先选定的 40 张（4 个断面 × 10 个电流）生成结果。相对于 mean3 GT，平均 PSNR 从 31.46 提升至 40.47 dB，SSIM 从 0.6708 提升至 0.9312；这些是 40 张样本结果，尚非完整 test 集评估，也不是与既有模型的比较。

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
  workflows/         完整实验流程；本次见 cylinderhead_20261104/
  upstream/          BasicSR 上游原有脚本
tests/               pytest（tests/manual/ 为上游手动脚本）
docs/                项目文档；docs/upstream/ 为 BasicSR 原版文档
```

`datasets/`、`experiments/`、`results/`、`tb_logger/` 为本地运行产物目录，已 gitignore。

## 配置

| 位置 | 作用 |
|------|------|
| `options/train/SwinIR/*.yml` | 训练的全部语义：数据路径、`name`、`num_gpu`、超参、`experiments_root`。换实验就换这份 YAML |
| `apps/denoise_demo/models.yaml` | Demo 可选的模型列表 |
| `DATA_DIR`、`CST_STORAGE_USER`、`DATA_MOUNT_SOURCE`、`HOME_MOUNT_SOURCE` | 数据挂载与存储用户，见 [docs/infrastructure.md](docs/infrastructure.md) |

## 实际存放位置与 Git 提交

**主 Git 工作副本为 `/mnt/home/huang/code/cst_ai/BasicSR`，分支为 `master`。**
当前 work1（192.168.3.11）运行 CPU Gradio 容器，推理和手动训练使用东京 Vertex GPU；不注入 work1 本机显卡。gpu01 的四卡训练与 work1 的本机 GPU Demo 为历史部署，旧页面已停止。

| 位置 | 内容与用途 | 本仓库 push 是否包含 |
|------|------------|----------------------|
| `/mnt/home/huang/code/cst_ai/BasicSR` | 源码、项目文档、通用训练配置、Demo 模型注册表 | 已提交的 Git 跟踪文件会包含 |
| `scripts/workflows/cylinderhead_20261104/`（本仓库内） | 本次 data / training / evaluation / demo / config 的统一代码入口 | 作为源码提交，数据与权重除外 |
| `deploy/vertex/`（本仓库内） | 当前 CPU 页面、Vertex 管理及训练入口 | 需 commit/push 后才会发布新增改动 |
| `/home/huang/scripts/basicsr-demo` 与 `cylinderhead-*` | 旧兼容链接已移除，不再作为部署入口 | 历史源码仍在本仓库，分析产物保留 NFS |
| `/mnt/data/cst_ai/datasets` | 原始 ZIP、mean3 GT、配对数据与独立 test，均使用带时间戳的目录 | 不包含，位于仓库外 |
| `/mnt/home/huang/cst_ai/experiments` | 实验目录：冻结源码、实际配置、日志、模型、评估结果 | 不包含，位于仓库外 |
| `/mnt/home/huang/cst_ai/demo_models` | 带时间戳的 Demo 发布权重 | 不包含，位于仓库外 |

`/home/huang/scripts` 本身不是 Git 仓库；其中 `tools` 存放下载工具，`kintai`、`trading system` 是其他项目。本次过程代码已集中到 [`scripts/workflows/cylinderhead_20261104/`](scripts/workflows/cylinderhead_20261104/README.md)，包含执行顺序、配置、容器路径契约和实验产物入口。旧的本机兼容目录链接已经移除；三个历史辅助目录及分析产物归档于 `/mnt/home/huang/cst_ai/archive/scripts-closeout-20261118/`。旧 Demo Compose 仅保留显式 `legacy-local-gpu` profile，默认不启动；当前部署统一使用 `deploy/vertex/`。OAuth 凭据仍保存在主机用户配置中，独立于源码与数据。

本次实验的定位入口：

- 数据根目录：`/mnt/data/cst_ai/datasets/cylinderhead_300kv_pairs_raw_mean3_20261104_173035_JST`；`train.txt`、`val.txt`、`test.txt` 为固定划分，`test_holdout/LQ` 与 `test_holdout/GT` 为独立 test 文件。
- Demo 测试包：上述目录的 `test_demo_LQ_40.zip`，解压后上传单张 TIFF。
- 实验根目录：`/mnt/home/huang/cst_ai/experiments/cylinderhead_300kv_mean3_20261104_173035_JST`；`control/train.yml` 为本次实际配置，`control/` 为执行脚本，`artifacts/` 为训练产物，`test_preview/` 为 40 张去噪 TIFF、对比 PNG 和 `summary.json`，`resource_review/` 为资源测量记录。
- 实验目录的 `code/` 是提交 `41d7bfd` 的冻结源码快照，没有 `.git`；后续改代码、commit、push 均回到主工作副本。
- 最终权重：`/mnt/home/huang/cst_ai/demo_models/SwinIR_Cylinderhead_mean3_20261105_054425_JST_final_i30000.pth`（Demo 使用 `params_ema`）。
- 当前 Vertex 页面：<http://192.168.3.11:7861/>，选择 `Cylinderhead mean3 | 2026-11-05 05:44 JST | FINAL step 30000`。

远端 `gitlab` 指向 `jil_atr/basicsr`，`github` 指向 `huangpenguin/cst_ai_basicsr`，`upstream` 为 XPixelGroup/BasicSR。`all` 的两个 push URL 分别为自己的 GitLab 和 GitHub 仓库。操作示例：

```bash
cd /mnt/home/huang/code/cst_ai/BasicSR
git status --short
git diff -- README.md apps/denoise_demo/models.yaml
git status --short -- scripts/workflows/cylinderhead_20261104
# 审阅后，只暂存准备发布的文件；原始 TIFF 单独管理
git add README.md apps/denoise_demo/models.yaml scripts/workflows/cylinderhead_20261104
git commit -m "chore: consolidate cylinderhead training workflow"
git push all master
```

同一时间只在一台机器上执行共享仓库的 Git 写操作。`git push` 仅发布已提交内容，不会上传外部数据、权重、Docker 镜像或运行产物。

## 部署

- [Vertex 训练与在线推理](deploy/vertex/README.md)：内部 CPU 页面、云 GPU 接口和镜像构建。
- [操作与恢复](docs/deploy/vertex-operations-guide.zh-CN.md)：启动、停止、恢复与验证范围。
- [Gradio 基础镜像](deploy/serve/README.md)：供 Vertex 镜像复用的依赖和权重准备。

原始 TIFF、模型权重、打包样本和一次性云端诊断不提交 Git。历史部署记录归档于 `/mnt/home/huang/cst_ai/archive/basicsr-review-20261118/`。

## 文档

- [docs/workflow.md](docs/workflow.md)：数据准备、训练、评测、演示的完整操作流程
- [docs/experiments.md](docs/experiments.md)：实验记录与结果
- [docs/infrastructure.md](docs/infrastructure.md)：Docker / Dev Container / GitLab Runner / 代码质量
- [docs/research-notes.md](docs/research-notes.md)：数据特性、改进方向与开源方案调研
- [docs/upstream/](docs/upstream/)：BasicSR 上游 README 与文档

本项目沿用 BasicSR 的 [Apache 2.0 许可](LICENSE.txt)，引用信息见 `CITATION.cff`。

当前运行环境、界面整合与 Git 状态见 [work1 界面整合记录](docs/deploy/work1-ui-consolidation-20261118.md)。
