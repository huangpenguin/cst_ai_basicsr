# 东京训练 GPU 与配额申请准备（2026-11-18）

**申请状态：文案已准备，尚未提交。** 本轮单张 T4 的真实短训练已经成功，当前 128×128 crop、batch 1 配置无需申请 H100。是否需要更大显存，依据后续更大 crop/batch 的测量和吞吐需求判断；不按旧的四张 V100 数量申请资源。H100 的文档支持、配额获批和实际能分配到卡是三个不同条件，本轮未实测 H100 库存。

## 先按实际需求选卡

本轮受限验收为 CustomJob `3085482377774891008`：东京、`n1-standard-8`、一张 T4、batch 1、128×128 crop、8 次优化器更新，最多 1800 秒；验证使用一对真实监控图片、128 tile，在第 4/8 步执行。任务已成功完成，确认 8 次优化器更新、`weights_changed: true`、2 次真实监控验证。runner 耗时 212.50 秒，worker 耗时 170.32 秒（包含两次验证，不能直接解释为纯训练每步耗时）。

| 实测项 | 字节数 | GiB（2³⁰ bytes） |
|---|---:|---:|
| T4 可见显存总量 | 15,636,037,632 | 14.562 |
| 训练峰值 allocated | 7,400,487,424 | 6.892 |
| 训练峰值 reserved | 7,547,650,048 | 7.029 |
| 验证峰值 allocated | 620,428,288 | 0.578 |
| 验证峰值 reserved | 7,061,110,784 | 6.576 |

训练 reserved 峰值约占可见总显存的 48.3%。验证 reserved 包含训练后保留的 allocator 缓存，不能视为验证本身最低显存需求。这次实测说明单 T4 足够运行已测试的配置；不证明更大 crop/batch 的需求，也不代表正式训练已经达到降噪质量目标。8 步只能验收训练、验证和产物通路，完整数据集的收敛与质量仍需后续正式训练评估。云端已成功；全部产物的本地下载和 SHA256 核验以主验收记录为准。

该单卡配置的训练和验证已成功。后续调整配置时继续比较 `training_peak_allocated_bytes`、`training_peak_reserved_bytes`、`validation_peak_allocated_bytes`、`validation_peak_reserved_bytes`。PyTorch allocator 数据不包含所有驱动上下文，应留出约 15–20% 的工程余量。若单 T4 有足够余量且训练时长可接受，继续用一张 T4，当前无需增加 GPU 配额。只有增大 crop/batch 或吞吐需求经测量确有必要时，才提交下方一张 H100 的申请。减少 GPU 数量或 batch 会改变优化批次；当前没有把 batch 1 声称为旧 global batch 16 的等价训练。

## 东京官方支持与配额快照

2026-11-18 查询的官方区域表列出东京 `asia-northeast1` 支持 T4、H100、H100 Mega 等训练加速器。A100 40GB 和 L4 带 `†`，脚注表示不支持训练；V100 未列入东京条目。不能依据某 GPU 在东京可做在线推理或 Compute Engine VM，就推断 Vertex Custom Training 也支持。[官方区域表](https://docs.cloud.google.com/vertex-ai/docs/general/locations)

| 方案 | 规格/约束 | 本项目当前决定 |
|---|---|---|
| 一张 T4 | `n1-standard-8` + `NVIDIA_TESLA_T4`，16 GB 级显存 | 本轮已实测成功；当前 128 crop / batch 1 足够 |
| 一张 H100 | `a3-highgpu-1g` + `NVIDIA_H100_80GB`，80 GB HBM3 | 仅作为条件性升级方案；尚未分配或测试 |
| H100 Mega / B200 | 面向更大的多卡机型 | 当前没有证明需要，暂不申请 |
| A100 / L4 / V100 | 东京训练区域条件不满足 | 不作为本项目东京训练申请目标 |

`n1-standard-8` 可配一张 T4，`a3-highgpu-1g` 对应一张 H100，属于官方训练机器兼容表中的组合。[训练计算配置](https://docs.cloud.google.com/vertex-ai/docs/training/configure-compute) A3 High 小规格的单卡 H100 有 80 GB GPU 显存，Google 的机器说明要求使用 Spot 或 Flex-start；官方发布说明也明确列出 Vertex Training 的这两种方式。[GPU 机器规格](https://docs.cloud.google.com/compute/docs/gpus/)、[Vertex 单卡 H100 支持说明](https://cloud.google.com/blog/products/compute/announcing-smaller-machine-types-for-a3-high-vms)

以下是本轮 `/tmp/basicsr-current-quotas.json` 在 **2026-11-18 17:24:36 JST** 保存的实际项目快照摘录。项目 `hallowed-pipe-497802-i0`，项目编号 `580558088344`；各行 region 均为 `asia-northeast1`。

| 指标（均以 `aiplatform.googleapis.com/` 开头） | 当前 effective limit | 这次申请目标 |
|---|---:|---:|
| `custom_model_training_nvidia_t4_gpus` | 6 | 不增加；本轮只用 1 |
| `custom_model_training_preemptible_nvidia_t4_gpus` | 6 | 不增加 |
| `custom_model_training_nvidia_h100_gpus` | 0 | 不申请此项 |
| `custom_model_training_preemptible_nvidia_h100_gpus` | 0 | **条件性申请到 1** |
| `custom_model_training_nvidia_h100_mega_gpus` | 0 | 不申请 |
| `custom_model_training_preemptible_nvidia_h100_mega_gpus` | 0 | 不申请 |

零值行的原始 `quotaBuckets` 仅返回 `dimensions.region`，未序列化 `effectiveLimit` 数值，按该响应的零值解释。提交前仍需在控制台复核当前值。该表是并发额度，不是当前空闲卡数量；也不包含已在运行任务的瞬时使用量。在线推理 `custom_model_serving_*` 配额与上表训练配额分开。

## H100 方案为何申请 preemptible 项

建议采用 Vertex CustomJob 的 `scheduling.strategy: FLEX_START`，接受排队等待容量。DWS Flex-start 使用名称带 `preemptible` 的训练 GPU 配额：H100 对应 `aiplatform.googleapis.com/custom_model_training_preemptible_nvidia_h100_gpus`。这里的配额命名不表示 DWS 任务会像 Spot 一样被抢占。申请普通 on-demand H100 项并不能代替这个额度。[官方 DWS 配额说明](https://docs.cloud.google.com/vertex-ai/docs/training/schedule-jobs-dws)

未来 H100 任务需另行生成并评审以下字段，当前未提交：

```json
{
  "workerPoolSpecs": [{
    "machineSpec": {
      "machineType": "a3-highgpu-1g",
      "acceleratorType": "NVIDIA_H100_80GB",
      "acceleratorCount": 1
    },
    "replicaCount": "1"
  }],
  "scheduling": {
    "strategy": "FLEX_START",
    "maxWaitDuration": "7200s",
    "timeout": "3600s",
    "disableRetries": true,
    "restartJobOnWorkerRestart": false
  }
}
```

这是供审核的字段片段，缺少镜像、数据等必需设置，不是可直接提交的完整任务。7200 秒排队上限和 3600 秒运行上限是拟议的首次 H100 验收范围；完整训练的运行上限须另按实际速度设定。已有任务生成器若未设置 `FLEX_START`，不能只换 GPU 名称就启动这份 H100 方案。DWS 的 `maxWaitDuration` 单独控制等待资源的时长。[官方 DWS 调度字段](https://docs.cloud.google.com/vertex-ai/docs/training/schedule-jobs-dws)

## 可复制的配额申请正文

仅在单 T4 实测不足，或团队明确决定评估更高吞吐后使用。文案没有声称 80 GB 是模型的最低需求，也没有声称已验证 H100 容量。

**Subject**

```text
Vertex AI Custom Training — request 1 preemptible-quota H100 GPU in asia-northeast1 for DWS Flex-start
```

**Request details**

```text
Project ID: hallowed-pipe-497802-i0
Project number: 580558088344
Service: aiplatform.googleapis.com
Region: asia-northeast1 (Tokyo)
Quota metric: aiplatform.googleapis.com/custom_model_training_preemptible_nvidia_h100_gpus
Current quota: 0
Requested quota: 1 GPU total
Planned machine: a3-highgpu-1g, 1 x NVIDIA_H100_80GB
Scheduling: FLEX_START (Dynamic Workload Scheduler)
Maximum concurrent training jobs using this quota: 1

We are preparing an internal industrial X-ray image-denoising workflow using BasicSR/SwinIR. Our baseline completed eight optimizer updates and two validation passes on one T4 with 128 x 128 training crops and batch size 1. The model weights changed. Training peak allocated memory was 6.892 GiB and peak reserved memory was 7.029 GiB out of 14.562 GiB visible device memory (CustomJob 3085482377774891008). That tested configuration does not require an H100; we will retain it when its runtime and model-quality results meet our needs.

We request a quota of one H100 GPU to allow a bounded evaluation of a larger-memory training configuration when required. We are sizing from actual training and validation measurements, rather than reproducing a previous four-GPU local setup. We have not validated H100 capacity in Tokyo and can accept queueing with DWS Flex-start.

The first H100 evaluation will have a one-hour execution limit, a two-hour resource-wait limit, one worker, and no automatic retries. Formal training will be started manually after reviewing the result and setting an explicit timeout. Training inputs and checkpoints use our existing private GCS bucket in Tokyo, and the job uses our existing dedicated training service account.

This request is for Vertex AI Custom Training using the preemptible-named quota required by DWS Flex-start. It is not a request for online prediction quota, H100 Mega, or four concurrent GPUs. Please let us know if any additional regional service quota is required for this one-worker configuration.
```

申请表如要求 expected usage、monthly hours、正式训练持续时间或业务截止日期，应按团队确认值填写，不能从本轮 8 步试验推算并承诺长期用量。上述正文已加入实际成功任务和显存证据。若提交此条件性申请，需补充触发升级的新增需求或测量结果；不要仅凭这次短训练申请更大显存，也不要将包含验证的 worker 总时长除以 8 后作为纯训练吞吐承诺。

## 控制台提交步骤

1. 打开本项目的 [Quotas & System Limits](https://console.cloud.google.com/iam-admin/quotas?project=hallowed-pipe-497802-i0)，核对项目 ID。
2. 按服务 `aiplatform.googleapis.com`、区域 `asia-northeast1` 和指标名 `custom_model_training_preemptible_nvidia_h100_gpus` 过滤。核对是 **training / preemptible / H100 / Tokyo**，不是 serving、H100 Mega 或普通 H100 项。
3. 选中该配额，进入 **Edit** / **Edit quotas**，将 **New value** 填为 `1`。这是申请后的总并发额度。
4. 填写联系信息和上述申请理由，复核指标、区域、额度后由操作者点击提交。记录申请编号及状态；本文件的准备过程没有执行这一步。
5. 获批后重新读取项目配额，确认目标项 effective limit 已变为 1，再准备完整 H100 CustomJob。获批仍不保证即时库存，也不自动启动训练。

界面操作依据 [Google Cloud 配额调整说明](https://docs.cloud.google.com/docs/quotas/view-manage)。若当前账号无法编辑，应由有项目配额管理权限的人员提交；保留同一份申请内容即可。
