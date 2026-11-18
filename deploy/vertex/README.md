# Vertex AI 训练 smoke 与在线 GPU 推理

当前内网入口：[http://192.168.3.11:7861](http://192.168.3.11:7861)（work1，CPU 页面 → 现有 Vertex GPU）。

2026-11-18 work1 页面与真实预测验收已通过，旧 work1 本地 GPU demo 和 gpu01 页面均已停止、容器保留。完整状态与副本归零结果见 [本轮迁移记录](../../docs/deploy/work1-migration-20261118.md)。项目统一使用 [Makefile](Makefile) 管理；原 [交接说明](../../docs/deploy/work1-migration-handoff-20261118.md) 保留作历史依据。

2026-11-18：内网页面已恢复英文、原来的 11 个模型预设目录和原始线性灰度预览。当前 Vertex 推理部署仍只支持 Cylinderhead final；其余选项标注 `not deployed to Vertex`，选择后禁用 Denoise，并在服务调用前检查，不能仅切换名称而使用错误模型。该修改未重新部署云端，也未新增云端预测测试。各权重位置与 work1 迁移准备见 [界面修正及迁移说明](../../docs/deploy/vertex-ui-restoration-20261118.md)。

2026-11-17 已验证 Vertex T4 真实训练 2 iter 成功（不是 30k 完整训练）；正式 final 模型完成 1518×1518 TIFF 推理，云端处理 46.364 秒、页面提交到下载 61.912 秒。前端只用 CPU，实际容器内没有 NVIDIA GPU 设备。闲置缩容已实测通过：全图推理完成约 6 分钟后，在线副本从 1 降为 0；随后单次冷请求自动唤醒成功，页面提交到下载约 200.3 秒，闲置后再次实测归零。

目标是在 Vertex AI 内执行真实训练与部署受 IAM 保护的在线推理，仅供内部使用。本机 CPU Gradio 页面负责上传、调用 Vertex 和下载，模型计算全部在 Vertex GPU 上执行；不创建 Compute Engine 独立 VM 或 Cloud Run 入口。GPU/区域/权限及实际部署状态由本轮云端执行记录确认。

训练镜像 `denoise-vertex:local` 保留原已验证版本与下方不可变 digest，包含 BasicSR 和少量真实训练样本。云端推理另用 `denoise-vertex-inference:local`，由 `Dockerfile.inference` 精简重复的 Torch/CUDA、uv 缓存及重复文件层；PyTorch 2.6/cu124、推理代码与正式 final 权重保持一致。本机 CPU 内网页面继续使用已缓存的 `denoise-vertex:local`，不需要改用推理镜像或安装宿主依赖、驱动。

## 训练准备

仅从已有训练 split 选取 4 对真实 TIFF，保持字节与像素不变；不把测试集用于训练。执行：

```bash
python3 deploy/vertex/stage_smoke_data.py \
  --data-root /mnt/data/cst_ai/datasets/cylinderhead_300kv_pairs_raw_mean3_20261104_173035_JST \
  --train-list /mnt/data/cst_ai/datasets/cylinderhead_300kv_pairs_raw_mean3_20261104_173035_JST/train.txt \
  --count 4
docker build -f deploy/vertex/Dockerfile -t denoise-vertex:local .
```

忽略 Git 的 `bundle/smoke-data.tar.gz` 是约 23 MB 的真实样本包；旁边的 manifest 记录选择与源文件 SHA-256。所有其他训练数据留在原位置。

`train_smoke.py` 使用现有 `basicsr.train` 完整训练入口和 SwinIR-M 架构，初始化为镜像内官方 noise25 预训练权重。默认 2 次 optimizer 更新，参数只允许 1–8 次，关闭整图验证、TensorBoard 和 W&B。配置为单 GPU、batch 1、128×128 随机 crop。历史正式训练为 4×V100、batch 4/GPU、global batch 16；此 smoke 只检查数据/GPU/反向传播/优化器/产物链路，不代表复现正式训练或精度。

东京训练选兼容的单 T4 worker（例如 `n1-standard-8` + `NVIDIA_TESLA_T4`）；东京 L4 在线推理能力不代表可用于东京 Custom Training。以 [Vertex 计算配置](https://docs.cloud.google.com/vertex-ai/docs/training/configure-compute) 和项目 Vertex 配额为准。

CustomJob 的 container 配置保留镜像原 ENTRYPOINT，只覆盖 args：

```json
{
  "imageUri": "asia-northeast1-docker.pkg.dev/hallowed-pipe-497802-i0/denoise/denoise-vertex@sha256:fac4b16b9454f8b4685743477959cff89175467d001d3407ecefcf2b4747a73f",
  "args": [
    "python", "/opt/vertex/train_smoke.py",
    "--iterations", "2",
    "--output-uri", "gs://BUCKET/vertex-smoke/RUN_ID"
  ]
}
```

使用专用训练服务账号，向其授权该私有输出 bucket 的对象读写；操作者需 actAs。CustomJob 设置有限 timeout，完成后训练 worker 自动释放；取消请求使用 Vertex CustomJob cancel。输出：

- `model/net_g_latest.pth`：本次 2 次更新后的训练 checkpoint；不替换正式 final。
- `training/train.log`、`training/effective_train.json`、`training/summary.json`：日志、实际配置与退出状态。

运行内的 GCS 传输使用 metadata server 短期凭据和镜像已有 requests，无服务账号密钥。未提供 `--output-uri` 时仅供本地临时验证，产物留在 `--output-dir`。

本轮云端训练已验收：CustomJob `1044436952901746688` 在东京单 T4 上执行 2 次更新并以 `SUCCEEDED` 结束；GCS 的 checkpoint、日志、配置和 summary 均已下载并通过大小/MD5 校验，checkpoint 可加载且参数确有更新。完整请求、资源名、修复过程与产物 URI 见 训练部署与验收记录（NFS：`/mnt/home/huang/cst_ai/archive/basicsr-review-20261118/docs/deploy/vertex-training-20261117.json`）。这是训练链路验收，未启动 30k 正式训练。

### 复跑已验证的 2 iter 训练

[`custom-job.smoke.json`](custom-job.smoke.json) 从上述成功请求提取，固定镜像 digest、专用服务账号、单 T4、batch 1、2 iter、1800 秒上限，并关闭自动重试。文件根节点是 `CustomJobSpec`，符合 [`gcloud ai custom-jobs create --config`](https://docs.cloud.google.com/sdk/gcloud/reference/ai/custom-jobs/create) 的格式；不要再包一层 `jobSpec`。两处输出路径的 `RUN_ID` 必须替换，以免覆盖历史产物。

以下命令在仓库根目录执行。当前 GPU 主机没有 gcloud，使用已有 Cloud SDK 容器与原 ADC：凭据只读挂载，短期 token 仅在容器进程内传给 [`CLOUDSDK_AUTH_ACCESS_TOKEN`](https://docs.cloud.google.com/sdk/docs/authenticate)，无需新增 CLI 登录或复制凭据。每次调用刷新 token，CLI 配置只写容器临时目录。

```bash
cd /mnt/home/huang/code/cst_ai/BasicSR

vertex_gcloud() {
  docker run --rm -i --read-only --tmpfs /tmp \
    --user "$(id -u):$(id -g)" \
    -e CLOUDSDK_CONFIG=/tmp/gcloud \
    --mount type=bind,src=/home/huang/.config/pai_ai_control/gcloud/application_default_credentials.json,dst=/adc.json,readonly \
    --mount "type=bind,src=${PWD},dst=/workspace,readonly" \
    --workdir /workspace \
    --entrypoint /usr/lib/google-cloud-sdk/platform/bundledpythonunix/bin/python3 \
    gcr.io/google.com/cloudsdktool/google-cloud-cli:stable \
    -c 'import os, sys
sys.path.insert(0, "/usr/lib/google-cloud-sdk/lib/third_party")
import google.auth
from google.auth.transport.requests import Request
credentials, _ = google.auth.load_credentials_from_file(
    "/adc.json", scopes=["https://www.googleapis.com/auth/cloud-platform"])
credentials.refresh(Request())
environment = dict(os.environ, CLOUDSDK_AUTH_ACCESS_TOKEN=credentials.token)
os.execvpe("/usr/lib/google-cloud-sdk/bin/gcloud", ["gcloud", *sys.argv[1:]], environment)' "$@"
}

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
export RUN_ID
python3 - <<'PY'
import json, os
from pathlib import Path
source = Path("deploy/vertex/custom-job.smoke.json")
spec = json.loads(source.read_text().replace("RUN_ID", os.environ["RUN_ID"]))
destination = Path("deploy/vertex/bundle") / f"custom-job.smoke.{os.environ['RUN_ID']}.json"
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(spec, indent=2) + "\n")
print(destination)
PY

# 此 create 命令会启动一份新的付费 GPU smoke；本轮只提供复跑命令，未再次执行。
TRAINING_JOB="$(vertex_gcloud ai custom-jobs create \
  --project=hallowed-pipe-497802-i0 \
  --billing-project=hallowed-pipe-497802-i0 \
  --region=asia-northeast1 \
  --display-name="denoise-smoke-${RUN_ID}" \
  --config="deploy/vertex/bundle/custom-job.smoke.${RUN_ID}.json" \
  --format='value(name)')"

vertex_gcloud ai custom-jobs describe "${TRAINING_JOB##*/}" \
  --project=hallowed-pipe-497802-i0 \
  --billing-project=hallowed-pipe-497802-i0 \
  --region=asia-northeast1
```

查询到 `JOB_STATE_SUCCEEDED` 后，在 `gs://hallowed-pipe-497802-i0-denoise-vertex/vertex-smoke/${RUN_ID}/` 核对前述四个产物。需要提前停止时，对同一 job ID 执行 `vertex_gcloud ai custom-jobs cancel`，并沿用查询命令的 project、billing-project 与 region 参数；取消后查询终态。复跑仍需保留现有仓库拉取权限、服务账号 actAs 和输出桶对象权限。

### 完整训练仍需准备的部分

当前 `train_smoke.py` 和 `train-template.json` 不能直接提交完整训练。修改 `--iterations` 为 30000 会被参数范围拒绝；只改配置文件也仍会被脚本覆盖为 smoke 参数。本轮没有上传完整数据，也没有创建完整训练任务。

已核对本机原始训练目录：`/mnt/data/cst_ai/datasets/cylinderhead_300kv_pairs_raw_mean3_20261104_173035_JST`。`train.txt` 有 6090 对，`val_monitor.txt` 有 20 对，两者无交集；训练与监控验证共 6110 对、12220 个 TIFF，所需文件均存在。文件逻辑大小合计 **52.452 GiB**；由于不同电流的 GT 使用硬链接，按源文件 inode 去重为 **28.849 GiB**。上传或打包方式会影响去重是否保留，不能把本机占用直接当作云端展开大小。`normalization.json` 固定为 0–65535；`test.txt` 的 2020 对应继续与训练/监控验证分开。

以下保留为旧 smoke 入口的差距对照。新的 [完整训练入口](training/README.md) 已提供 GCS 大文件读取、原四卡配置、周期持久化、恢复和任务配置生成器，并发布了准备镜像；完整云端训练尚未执行，四卡大显存配额与容量仍是前提：

1. **完整数据与 split。** 将训练/监控验证所需数据、`train.txt`、`val_monitor.txt`、`normalization.json` 和清单保存到私有 GCS，保留已有分组划分和源像素。当前镜像只有 4 对样本，当前入口只解包镜像内小包，没有完整数据的 GCS 下载/挂载路径；不应把完整数据塞进服务镜像。
2. **正式配置与路径。** 以原 `scripts/workflows/cylinderhead_20261104/config/train.yml` 为来源，适配 `/data/GT`、`/data/LQ`、两个 split、归一化、预训练权重、实验输出和 TensorBoard 路径。恢复 30000 iter、15000/25000 学习率节点、每 2000 iter 的监控验证和每 1000 iter 的 checkpoint；当前 smoke 已移除这些设置。
3. **GPU、batch 与 launcher。** 原训练为 4×V100 32 GB、batch 4/GPU、global batch 16、128 crop，实测每卡占用约 25527 MiB。当前单 T4 的 batch 1 只证明 smoke 可运行，不等价于原训练。需要先确定显存足够的可用训练硬件或明确接受新的批次方案；多 GPU DDP 应使用对应的 `torchrun`/`--launcher pytorch` 入口。现有 `SRModel.optimize_parameters` 每 batch 直接执行 optimizer step，没有可直接开启的梯度累积参数。
4. **长任务产物与恢复。** 当前入口仅在 subprocess 结束后上传 latest、日志和 summary；正式训练需要让周期 checkpoint、optimizer/scheduler state、验证结果及 TensorBoard 输出在任务运行期间持久化，并适配恢复路径。按镜像、输入归档、展开数据和周期产物重新核算训练盘；重新配置长任务 timeout，不能沿用 smoke 的 1800 秒上限。

以上差距与数据规模也记录在训练验收 JSON 的 `full_training_readiness` 中。完成这些准备并确定训练参数后，才能给出已验证可运行的完整训练提交配置；本轮保留已成功的 smoke 入口与冻结镜像。

## 在线推理协议

先在本机构建独立推理镜像，保持已验收训练镜像不变：

```bash
docker build -f deploy/vertex/Dockerfile.inference \
  -t denoise-vertex-inference:local .
```

本次精简推理镜像大小为 7,119,306,413 bytes，原镜像为 28,423,073,023 bytes，减少约 74.95%；已通过本机 GPU 检查、Vertex T4 部署和内网页面完整图像推理。云端使用不可变镜像 `asia-northeast1-docker.pkg.dev/hallowed-pipe-497802-i0/denoise/denoise-vertex-inference@sha256:d6a078989cb622fa40be6d5887908f6e0ec26b9ded22077aaf9b0688e7f231fc`；闲置缩至零已通过下述实际副本指标验收。

默认启动命令为 `python /opt/vertex/serve.py`。它监听 `0.0.0.0:${AIP_HTTP_PORT:-8080}`，从镜像内 registry 选中正式缸盖 final step 30000，启动时预加载到 CUDA。注册 Vertex Model 时配置 container port 8080、healthRoute `/health`、predictRoute `/predict`；服务也读取 Vertex 注入的 AIP 路由变量。

默认不依赖自定义预测服务账号或 GCS。在受 Vertex IAM 保护的 Endpoint 上，调用 `endpoints.predict`，JSON 输入如下：

```json
{"instances": [{"image_b64": "BASE64_ENCODED_PNG_OR_UINT16_TIFF"}]}
```

响应 `predictions[0]` 包含 `image_b64`（16 位 TIFF）、`shape`、`dtype`、`preset`、`device` 和 `seconds`。Base64 增加传输体积；需根据所选 Endpoint 类型核对请求/响应大小限制。云端预测容器不运行 Gradio；平台 IAM 控制调用，本机内部页面也不开启公网分享链接。

也支持可选 GCS 模式：

```json
{"instances": [{"input_uri": "gs://BUCKET/input.tif", "output_uri": "gs://BUCKET/output.tif"}]}
```

这一路径需要预测运行身份对输入对象的读取和输出对象的创建权限；未配置这些访问前使用 base64 默认模式。`input_uri` 和 `image_b64` 只能提供一个。若提供 `output_uri`，响应仅返回输出 URI，不返回图像 base64。

在线推理已部署到东京 `n1-standard-8` + 单张 `NVIDIA_TESLA_T4`，使用独立 prediction GPU 配额。新 dedicated Endpoint 为 `8911645097217490944`，Model 为 `6657078912276758528`，DeployedModel 为 `1239217774224474112`，详见 本轮部署记录（NFS：`/mnt/home/huang/cst_ai/archive/basicsr-review-20261118/docs/deploy/vertex-inference-v2-20261117.json`）。已通过 64×64 PNG 小图及 1518×1518 TIFF 原图上传、Vertex 计算与下载，均返回 16 位 TIFF，并实测闲置后在线副本降为 0。原 L4 库存失败与旧大镜像 T4 的通用 code 13 失败保留在 部署诊断记录（NFS：`/mnt/home/huang/cst_ai/archive/basicsr-review-20261118/docs/deploy/vertex-inference-readiness-20261117.json`），原失败 Endpoint 已删除。当前部署使用 **v1beta1** `deployModel` 请求的 `deployedModel.dedicatedResources` 配置：

```json
{
  "machineSpec": {
    "machineType": "n1-standard-8",
    "acceleratorType": "NVIDIA_TESLA_T4",
    "acceleratorCount": 1
  },
  "minReplicaCount": 0,
  "maxReplicaCount": 1,
  "initialReplicaCount": 1,
  "scaleToZeroSpec": {
    "minScaleupPeriod": "300s",
    "idleScaledownPeriod": "300s"
  }
}
```

Scale To Zero 是 Preview 功能，不支持 shared public Endpoint；此处只部署一个模型、单机单 GPU。上述 5 分钟配置控制启动后最短保留时间与无流量缩容等待时间，不保证恰好在第 300 秒完成缩容。

缩到零后的首次请求会触发启动并返回 429，该请求被丢弃，客户端必须等待后重新提交；本机页面已实现重试。首次实测副本归零后，仅提交一次 64×64 PNG 冷启动小图，页面自动等待并成功下载结果，端到端耗时 200.293 秒；该时间包含唤醒、客户端重试等待、网络、推理与下载，不是纯 GPU 启动时间。云端 handler 为 1.635 秒，见 真实冷启动验收（NFS：`/mnt/home/huang/cst_ai/archive/basicsr-review-20261118/docs/deploy/vertex-cold-start-20261117.json`）。

使用 Endpoint 返回的专属 `dedicatedEndpointDns` 对应完整预测 URL，不把管理 API 地址或网页地址填入客户端。无流量而持续缩至零超过 30 天的模型会自动 undeploy，届时需要重新部署才能继续使用。

配置与实际缩容已分别验证：v1beta1 返回上述 300 秒配置；Cloud Monitoring 在同一 Endpoint/DeployedModel 上记录 11:08:14 UTC 副本为 1、11:15:14 UTC 副本为 0，后者晚于最后一次成功页面推理 360.529 秒。首次结果见 闲置缩容验收记录（NFS：`/mnt/home/huang/cst_ai/archive/basicsr-review-20261118/docs/deploy/vertex-idle-20261117.json`）。冷启动测试后再次确认同一 DeployedModel 在 11:22:14 UTC 为 1、11:28:14 UTC 为 0，后者晚于冷请求完成 366.712 秒，见 最终闲置归零记录（NFS：`/mnt/home/huang/cst_ai/archive/basicsr-review-20261118/docs/deploy/vertex-idle-after-cold-20261117.json`）。观测已结束，未追加预测。配置限制参见 [官方 Scale To Zero 说明](https://docs.cloud.google.com/vertex-ai/docs/predictions/autoscaling)。

验收后若不继续使用，undeploy 模型以停止在线 GPU 资源计费；只删除本地容器不能停止 Vertex Endpoint 的费用。训练 2 iter 产物只用于验证训练链路，在线推理继续使用既有正式 final step 30000。

## work1 CPU 内网页面

正式入口：http://192.168.3.11:7861 。项目目录为共享 NFS 上的
`/mnt/home/huang/code/cst_ai/BasicSR`，直接复用代码，未复制或移动权重/数据。
页面身份为本机合法 `authorized_user` ADC（`huang.pengbin@johnan.com`），
本地私有配置在 `/home/huang/.config/basicsr-vertex/ui.env`，ADC 同目录 0600。
不要将凭据放入 Git、NFS、镜像或浏览器。

统一在仓库根目录管理：

```bash
make -f deploy/vertex/Makefile ui-check
make -f deploy/vertex/Makefile ui-start   # 已有容器时拒绝覆盖
make -f deploy/vertex/Makefile ui-status
make -f deploy/vertex/Makefile ui-logs
bash deploy/vertex/read-cloud.sh         # 只读状态与实际副本，不预测
```

已存在的正确容器在人工停止后恢复时，先检查 `ui-status`，再使用
`docker start denoise-vertex-ui`；需要停止新入口时使用
`docker stop denoise-vertex-ui`。`ui-start` 不重建现有容器。

容器 `denoise-vertex-ui` 以 UID/GID `1003:1003` 运行，只读挂载当前
`internal_ui.py` 和本机 ADC，设置 `NVIDIA_VISIBLE_DEVICES=void`、
清空 `CUDA_VISIBLE_DEVICES`。实测无 `/dev/nvidia*`，DeviceRequests 为空，
仅发布到 work1 实际内网 IP，`share=False`。保留英文界面和 11 个模型目录；
未部署的 10 个选项禁用。当前云端只支持 Cylinderhead final。

2026-11-18 真实页面验收：64×64 冷请求全流程 231.861 秒，云端 1.835 秒；
1518×1518 TIFF 全流程 51.755 秒，云端 47.361 秒。均成功下载尺寸一致的
uint16 TIFF，返回模型标识匹配。测试从 work1 经其内网 URL 发起；另外从
物理主机 gpu01 实测新页面 HTTP 200 和 11 个预设。Mac 浏览器尚未代测。
本轮新采样已确认 14:18:14 JST 副本为 0，距最后成功预测 370.562 秒；
详细证据见 [迁移记录](../../docs/deploy/work1-migration-20261118.md)。

旧 work1 `denoise-demo` 与旧 gpu01 `denoise-vertex-ui` 均在新页面验收后停止，
未删除容器、镜像或 NFS 文件。回退时先停止 work1 新入口，确认旧配置后
可在 gpu01 使用 `docker start denoise-vertex-ui`；需要旧本地 GPU demo 时
另行明确选择，不能将其误认为 CPU→Vertex 页面。

## 完整训练与大文件上传

[训练操作说明](training/README.md) 提供显式手动提交的 8 步真实数据验收、
数据预检、归档、work1 直接上传 GCS、按实际显存选择单卡/多卡、周期备份与恢复。
当前固定训练镜像为：

```text
asia-northeast1-docker.pkg.dev/hallowed-pipe-497802-i0/denoise/denoise-vertex-training@sha256:458483e3a2fb73d5c3e5083b5e39217b970de69c1c994c7936072ed4cdc2adb3
```

镜像默认只显示帮助，本地检查均禁止 GPU 注入。2026-11-18 的实际云端短训练
已成功完成，见[真实短训练验收及手动复现](../../docs/deploy/training-acceptance-20261118.md)
及 `docs/deploy/training-readiness-20261118/`；不要将其等同于正式模型训练
完成。正式长训练未启动，完整数据未上传，线上推理模型未替换。
原四张 V100 并非固定需求；单卡 batch 1 的优化行为与原 batch 16 不等价。
现有服务账号和私有桶继续复用，未修改 IAM。

[东京 GPU 与配额申请草案](../../docs/deploy/training-gpu-quota-request-20261118.md)
区分训练支持、配额和实际库存；申请尚未提交。
[同事内网上传设计](../../docs/deploy/internal-xray-workflow-design-20261118.md)
说明断点接收、审核、GCS 同步和手动训练流程；新上传门户尚未实施。

## 本地适配验证

本地适配验证使用独立 `--rm` 容器和一张空闲 GPU，不发布宿主端口。训练 smoke 运行两次更新并确认 checkpoint 权重变化；服务测试检查 `/health`、`/predict`、正式 final 预设，以及 base64 输出确为尺寸一致的 16 位 TIFF。本地检查用于发现适配问题；真实 Vertex 训练、部署和页面推理结果分别以上方对应的云端验收记录为依据。

依据：[创建 CustomJob](https://docs.cloud.google.com/vertex-ai/docs/training/create-custom-job)、[自定义预测容器协议](https://docs.cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements)。

## 旧界面详情整合

当前页面恢复模型/checkpoint 元数据、图片详情与云端 GPU 配置及副本监控。
详见 [界面、目录、环境与 Git 状态](../../docs/deploy/work1-ui-consolidation-20261118.md)。
旧本机 GPU Compose 仅保留显式历史 profile；当前入口仍为 work1:7861。
