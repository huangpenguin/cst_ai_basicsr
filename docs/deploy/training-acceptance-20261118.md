# work1 / Vertex 真实短训练验收与后续操作

日期：2026-11-18（JST）。正式长训练未启动；本轮只提交了一次获准的受限真实 GPU 训练。

## 实际结果

东京 Vertex CustomJob **3085482377774891008** 已返回 **JOB_STATE_SUCCEEDED**。
[控制台任务](https://console.cloud.google.com/vertex-ai/locations/asia-northeast1/training/3085482377774891008?project=hallowed-pipe-497802-i0)

| 检查 | 结果 |
|---|---|
| 机器 | `n1-standard-8`，1 × Tesla T4，150 GiB SSD |
| 训练 | 原 BasicSR/SwinIR 网络，float32，batch 1，128×128 crop，8 次真实优化器更新 |
| 数据 | 4 对真实训练 TIFF + 1 对真实监控 TIFF，原始像素，不含测试集；已获用户明确上传授权 |
| 时间边界 | 1800 秒上限，关闭自动重试；仅提交这一个任务 |
| 云状态 | 17:32:45 JST 创建，17:35:36 运行，17:39:39 成功结束 |
| 实际训练程序 | worker 170.32 秒；runner 212.50 秒（包括部分持久化），不能当作每步耗时或正式训练估时 |
| 更新证据 | 8 次 optimize 调用、8 条有限 loss，全部可训练参数初末哈希不同 |
| 验证 | 第 4、8 步各完成一次完整监控图的分块验证，数值有限 |
| 产物 | 第 4/8 步及 latest 模型，第 4/8 步优化器与调度器状态，日志、TensorBoard、验证图和配置 |
| 下载校验 | 15 个产物、936139029 字节全部 SHA256 验证通过 |
| 独立 CPU 检查 | latest 与第 8 步模型一致；所有保存参数有限；490 组优化器参数状态均为 step 8；CUDA 不可用 |
| 推理服务 | 未替换线上模型；work1 继续 CPU 前端调用已有 Vertex 推理资源 |

最后一次监控结果 PSNR 25.901254、SSIM 0.514987；只有一对监控图和八步更新，**不能据此认定模型质量达标或发布模型**。loss 并非单调下降；本轮目标是验证真实训练、显存和产物链路。

## 显存与资源选择

| PyTorch 测量项 | 峰值 |
|---|---:|
| 训练 allocated | 6.892 GiB |
| 训练 reserved | 7.029 GiB |
| 验证 allocated | 0.578 GiB |
| 验证 reserved | 6.576 GiB |
| GPU 可见总显存 | 14.562 GiB |

allocated/reserved 是 PyTorch 分配器统计，不是整个设备占用；reserved 包含缓存。
当前固定配置有明显余量，建议先使用一张 T4，无需为这个配置申请更大的 GPU。
完整数据集增加样本数不等同于同时把所有样本放进显存；改变 crop、batch、图像维度或验证策略仍需重新测量。
原本四张 V100、global batch 16 的训练配方与现在 batch 1 不等价；当前没有新增梯度累积。正式训练前需按样本曝光次数和验证表现重新确认学习率、步数与超时，不能只照搬 30000 步。

东京 T4 训练配额实查为 6。更大显存的条件性备选为 1 × H100 80GB，
`a3-highgpu-1g` + `FLEX_START`，需申请 preemptible H100 训练额度 0 → 1。
当前不需要提交该申请。A100 40GB 和 L4 在东京区域表带不支持训练的脚注，不能用其在线推理配额替代训练配额。
详见[配额申请草案与官方依据](training-gpu-quota-request-20261118.md)。申请尚未提交，H100 实际库存未验证。

## 冻结输入与镜像

- 镜像：`asia-northeast1-docker.pkg.dev/hallowed-pipe-497802-i0/denoise/denoise-vertex-training@sha256:458483e3a2fb73d5c3e5083b5e39217b970de69c1c994c7936072ed4cdc2adb3`
- 输入：`gs://hallowed-pipe-497802-i0-denoise-vertex/datasets/acceptance-2ca67152392c4ce8/dataset.tar`
- 输入 SHA256：`2ca67152392c4ce8a270982038369367e8c02912f07aaad8f6ddc75ad2208c94`
- 输出：`gs://hallowed-pipe-497802-i0-denoise-vertex/training-readiness/acceptance-20261118T083239Z`
- 最新模型 SHA256：`06dcddb1497f2714dd0efb877d23919407defc45d3bb9cd16c76db3bc161c0a5`

运行身份继续使用 `denoise-vertex@hallowed-pipe-497802-i0.iam.gserviceaccount.com`。
提交与下载使用 work1 本地合法用户 ADC；没有新增 key、修改 IAM、创建桶或重建 Endpoint。
凭据未进入 NFS、Git、镜像或报告。

原始证据目录：[training-readiness-20261118](training-readiness-20261118/)。
`job-spec.json` 为实际提交配置，`job-latest.json` 为最终控制面状态，
`training-summary.json` 为运行记录，`checkpoint-verification.json` 为独立 CPU 检查，
`download-verification.json` 为全部下载校验（仅保留在 NFS，不发布），`data-manifest.json` 为源数据逐文件哈希（仅保留在 NFS，不发布），
`source-manifest.json` 为构建时文件哈希。完整图像与大 checkpoint 保留在该目录
被 Git 忽略的 `artifacts/` 及私有 GCS 中。

## 你手动再运行一次

在 work1 以 huang 登录后：

```bash
cd /mnt/home/huang/code/cst_ai/BasicSR
python3 deploy/vertex/training/acceptance.py prepare /home/huang/basicsr-acceptance-manual-01
cat /home/huang/basicsr-acceptance-manual-01/job-spec.json
python3 deploy/vertex/training/acceptance.py submit /home/huang/basicsr-acceptance-manual-01
python3 deploy/vertex/training/acceptance.py status /home/huang/basicsr-acceptance-manual-01
```

`prepare` 只生成文件；`submit` 才启动付费训练。每次复现用一个新目录；程序自动
生成新的 GCS 输出前缀，固定本轮数据、镜像与八步配置，不覆盖旧产物。
状态为 `JOB_STATE_SUCCEEDED` 后：

```bash
python3 deploy/vertex/training/acceptance.py fetch /home/huang/basicsr-acceptance-manual-01
cat /home/huang/basicsr-acceptance-manual-01/artifacts/summary.json
cat /home/huang/basicsr-acceptance-manual-01/artifacts/telemetry/worker-0.json
```

应看到 `status: SUCCEEDED`、`exit_code: 0`、`optimizer_updates: 8`、
`validation_calls: 2`、`weights_changed: true`，以及下载校验 `PASS`。
下载中断时先将不完整的 `artifacts/` 改名保留，再重跑 `fetch`；不完整文件不能当作校验通过。
GPU 浮点计算不保证不同机器之间逐位相同，但输入、代码、配置与验收条件一致。
需要停止时运行 `python3 deploy/vertex/training/acceptance.py cancel /home/huang/basicsr-acceptance-manual-01`。

提交后即使终端断开，任务也继续执行；不要重复提交。提交标记会阻止误操作。
若 CLI 返回丢失，先按 `run.json` 中 display_name 查找云任务，再处理本地记录。
本轮仅执行过手动入口的 prepare/status/fetch，未替你提交第二次训练。
更多正式数据与恢复操作见[训练 README](../../deploy/vertex/training/README.md)。

## 同事上传的实现方向

以 TIFF 为首期输入，同事分别提交 `Noisy images` 和 `Reference images (GT)`，
保留工件、扫描序列与切片身份。按这些字段、尺寸及空间对应关系提出配对候选，
由你审核，不将同名文件自动当成合格 GT/LQ。

流程：浏览器断点上传 → work1 接收并写入真实 NFS → 完整性/格式检查 →
你审核配对和数据划分 → 后台续传到私有 GCS → 你手动启动训练。
同事无需 GCP 账号或 Google Drive。现有去噪入口继续使用，英文界面保持。

RAW 压缩包先保存并标记待转换。已重建体数据在格式参数完整时可通过代码切片；
原始投影则需要重建与设备几何参数，当前没有确认这批 RAW 属于哪种。
详见[内网工作流设计](internal-xray-workflow-design-20261118.md)及其中官方格式/重建依据。

## 尚未完成、没有擅自执行的项目

- 正式长训练、完整 6090+20 对数据上传、独立测试集评估及模型发布。
- 新 TIFF/GT 接收门户的开发部署；当前交付为可实施设计，现有网页仍为去噪页面。
- 大批量断点续传实际中断恢复、GPU 作业中断后继续训练的第二次付费验证。
- H100 配额申请提交及 H100 实际资源分配；当前 T4 配置不需要新增额度。
- 同事实际 TIFF 单页/多页、批次大小、RAW 格式、公司内部身份/证书等上线参数。

旧 gpu01 页面已在新入口真实预测验收成功后停止，旧 work1 GPU demo 同样停止；
旧容器和共享代码、权重、数据均保留。新入口为 `http://192.168.3.11:7861`。
清理的是旧部署兼容软链接，历史工作流归档仍保留在共享存储；没有复制新项目。
