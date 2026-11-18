# Cylinderhead 300 kV mean3：2026-11-04 四卡微调流程

这是本次已完成实验的数据准备、训练、测试和部署代码的统一维护入口。
源码由 work1 的分散辅助目录归入本仓库；运行产物保留在 NFS，未放进 Git。2026-11-18 去重后，work1 的三个 `cylinderhead-*` 旧目录链接到 `/mnt/home/huang/cst_ai/archive/scripts-closeout-20261118/` 下的同名归档，内部脚本入口仍指向本仓库。历史训练和最终模型注册已完成，旧入口无需常驻运行。
本目录是这一次实验的具体配方，包含站点路径、容器挂载点和主机设置；移植新机器或新批次时需修改配置，不是通用云端一键启动器。

## 文件布局

```text
data/         下载、扫描检查、mean3 GT、对齐诊断、GT 内容统计、test 导出
training/     配对及划分、实验配置生成、四卡 smoke/resume/正式训练流水线、gpu01 启动
evaluation/   最终 test 推理、显存/时间测量、带时间戳的 Demo 权重发布
config/       本次路径、输入元数据、实际训练配置与 smoke 配置快照
demo/         work1 Dockerfile、Compose、依赖快照、Demo 冒烟检查
```

`config/train.yml` 是本次实际执行配置的快照；`training/setup_run.py` 记录其生成逻辑。
`config/source_commit.txt` 为训练使用的冻结源码提交 `41d7bfd`，本次整理不改变历史实验。
`config/gt_content_profile.json` 是生成本次确定性划分所需的 GT 内容统计输入。
`config/source_manifest.json` 仅含原始数据来源和文件元数据；Google OAuth 凭据不在本仓库。

## 本次执行过程与入口

1. **下载**：`data/download.py` 从 `config/destination.txt` 读取目标 NFS 路径，使用主机上现有 rclone 的只读 remote。目标目录的 `source_manifest.json` 需预先存在，模板归档于 `config/`；下载后执行 MD5 校验、生成 ZIP 清单。脚本生成的文件列表写入目标数据目录，不写入源码目录。rclone 和授权配置仍位于主机用户目录。
2. **检查与平均**：`data/assess_pairing.py` 检查 ZIP 内的断面名称、元数据和偏移，`data/show_repeats.py` 对比三次 200 µA 扫描；`data/mean_three_scans.py` 按同名断面取 uint16 四舍五入算术平均；`data/compare_currents.py` 对比各电流；`data/profile_gt.py` 统计平均 GT 中的结构含量。偏移仅用于诊断，没有对源图做配准修正。
3. **配对与划分**：`training/prepare_pairs.py` 读取 `config/job.json` 和 GT 内容统计，提取原始 LQ、硬链接 mean3 GT，以断面编号为组分配 train/val/test，并保留 16 个断面的隔离带；同一断面的全部电流条件始终属于同一组。先固定测试预览列表，再开始训练。
4. **冻结与启动**：`training/setup_run.py` 从主仓库 `git archive HEAD` 创建源码快照、复制 noise25 初始权重、生成配置并复制执行入口；已有实验配置时停止，避免改写历史记录。`training/start_on_gpu01.sh` 是本次 gpu01 专用启动记录，使用已准备好的主机 Python/venv 快照和本地训练镜像。新实验需要更换时间戳路径、容器名称和环境快照，并先提交待冻结源码。
5. **四卡流水线**：`training/pipeline.py` 通过 `torch.distributed.run --nproc_per_node=4` 进行四卡 smoke、断点续训验证、正式 30,000 步训练。每卡 batch=4，全局 batch=16；使用 MSE 全参数微调，验证每 2,000 步运行固定 20 张验证图。
6. **独立测试**：正式训练后，`evaluation/evaluate_test.py` 使用最终 EMA 对预选的 40 张 test 图进行推理，输出去噪 uint16 TIFF、LQ/预测/GT 比较 PNG 和浮点 PSNR/SSIM。`data/export_test_samples.py` 根据既有列表建立完整 test 硬链接目录和 40 张 Demo 测试包。test 不参与训练、验证或 checkpoint 选择。
7. **资源测量与发布**：`evaluation/benchmark_inference.py` 在验证图上测量推理显存与时间；`evaluation/register_model.py --watch` 等待流水线完成，复制带时间戳的最终权重、更新主仓库模型注册表并重启 Demo。直接调用该发布脚本时，默认注册最新的编号 checkpoint；历史实验已发布完成，日常启动 Demo 不需要再次运行。
8. **Demo**：本目录 `demo/` 为部署文件的维护入口；实际 work1 Compose 目录仍为 `/home/huang/scripts/basicsr-demo`，以兼容已存在的容器。2026-11-18 去重后，该路径为指向本目录 `demo/` 的目录链接，Docker 继续使用完整构建上下文；修改直接在本目录维护，无需同步副本。Compose 固定项目名 `basicsr-demo`，继续使用 `denoise-demo` 容器名和原来的 NFS 挂载。

## 容器内路径契约

数据检查和推理脚本依赖下列路径，它们是容器挂载点，不是主机上要新建的数据副本。
Python 依赖来自本地已有 Demo/训练镜像；仅下载脚本使用主机 rclone。

| 入口 | 只读输入挂载 | 输出挂载 |
|------|--------------|----------|
| `assess_pairing.py` / `show_repeats.py` | 原始数据根 → `/dataset` | QC 目录 → `/analysis` |
| `mean_three_scans.py` | 原始数据根 → `/input` | mean3 根目录 → `/output`（需有 `GT/`） |
| `compare_currents.py` | 原始数据根 → `/input` | QC 目录 → `/analysis` |
| `profile_gt.py` | mean3 根目录 → `/mean` | QC 目录 → `/analysis` |
| `pipeline.py` / `evaluate_test.py` | 冻结源码 → `/workspace`；配对数据 → `/data` | 实验根目录 → `/training_run` |
| `benchmark_inference.py` | 仓库 → `/workspace`；输入目录 → `/benchmark_input`；发布权重保持原绝对路径 | 资源记录目录 → `/benchmark_out` |

基准测试从 `/benchmark_out/demo_model_status.json` 读取模型，从 `/benchmark_input/input_validation.tif` 读取验证图。
实际配置生成和本地配对使用 `config/job.json` 中的主机绝对路径。

## 产物位置与结果

- 原始 ZIP：`/mnt/data/cst_ai/datasets/cylinderhead_300kv_raw_20261102_191914_JST`
- mean3 GT：`/mnt/data/cst_ai/datasets/cylinderhead_300kv_gt200ua_mean3_20261104_165839_JST`
- 配对数据：`/mnt/data/cst_ai/datasets/cylinderhead_300kv_pairs_raw_mean3_20261104_173035_JST`
- 实验目录：`/mnt/home/huang/cst_ai/experiments/cylinderhead_300kv_mean3_20261104_173035_JST`
- 最终 Demo 权重：`/mnt/home/huang/cst_ai/demo_models/SwinIR_Cylinderhead_mean3_20261105_054425_JST_final_i30000.pth`

训练 6,090 对、验证 1,010 对、独立 test 2,020 对。V100 32 GB ×4 学习 30,000 步，用时 11:27:03。
预选 test 40 张：平均 PSNR 31.46 → 40.47 dB，SSIM 0.6708 → 0.9312。这不是完整 test 集评估，也不是与既有模型的对比。

输入处理为 `uint16 → float32 → clip(x / 65535, 0, 1)`，batch 张量为 `N×1×H×W`。
`N` 是当前进程的一批图片数，本次为每 GPU 4；4 张 GPU 合计 16。
训练随机裁剪 128×128 是为了限制反向传播显存，并在不同步骤采样不同局部；LQ/GT 使用相同裁剪位置与增强。
验证和 Demo 对完整图像分块推理并拼回原尺寸，不会把最终图像永久裁成 128×128。
卷积权重采用 `[输出通道, 输入通道, 卷积核高, 卷积核宽]`：首层 `[180,1,3,3]` 提取 180 个特征通道，末层 `[1,180,3,3]` 重建单通道输出。
基模为原生灰度 SwinIR-M noise25，无需 RGB 通道复制。

## Git 入口

本目录通过 `/mnt/home/huang/code/cst_ai/BasicSR` 的 Git 管理；此处与两台主机使用的 NFS 工作副本是同一份源码。
`git push all master` 推送到 `jil_atr/basicsr`（GitLab）和 `huangpenguin/cst_ai_basicsr`（GitHub）。
上游 `XPixelGroup/BasicSR` 的 remote 名为 `upstream`，不包含在 `all` 的 push URL 中。
数据、权重、镜像、授权信息和运行日志独立于 Git。
