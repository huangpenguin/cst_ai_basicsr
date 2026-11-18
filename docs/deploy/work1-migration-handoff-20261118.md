# work1 内网页面迁移交接

## 目标与当前状态

将内网 CPU Gradio 入口由 gpu01 转到 work1，继续调用现有 Vertex GPU。迁移不要求重新构建镜像、不要求复制 NFS 仓库，也不要求新建云端模型、Endpoint 或训练任务。2026-11-18 本轮仅准备交接；未连接 work1，gpu01 入口仍运行。

代码位于 `/mnt/home/huang/code/cst_ai/BasicSR`。准备时源仓库 HEAD 是 `ee5f33dedb31658dbbff8d54b355cf65b3e6d2cd`，分支 master，部署代码已纳入该提交。本轮新增迁移文件留在共享工作区；不要仅从旧远端 clone 或强制 checkout 后假设得到全部文件，也不要覆盖其他会话的改动。本文不据此断言远端 push 状态。

| 项目 | 已知值 / work1 上的要求 |
|---|---|
| 旧主机 / URL | gpu01 / `http://192.168.3.18:7860` |
| 页面容器 | `denoise-vertex-ui`，只读挂载当前 `internal_ui.py` |
| 新主机 / IP | work1；**需在新主机检查实际内网 IP，不沿用 gpu01 IP** |
| 共享代码 | `/mnt/home/huang/code/cst_ai/BasicSR`，应在实际 NFS 上 |
| 原权重 | `/mnt/home/huang/cst_ai` 及仓库的 `experiments/pretrained_models`，已核对 11 个预设存在 |
| 完整数据 | `/mnt/data/cst_ai`，也是 NFS；仅迁移页面不需要挂载进容器 |
| 旧主机 ADC | `/home/huang/.config/pai_ai_control/gcloud/application_default_credentials.json`，本地 XFS，0600，不是共享文件 |
| 新主机 ADC | 优先用 work1 上已有、获准的 **authorized_user** ADC；当前页面不支持直接换成 service_account 类型 JSON |
| 后台身份 | 需能使用项目 `hallowed-pipe-497802-i0`、调用当前 Vertex Endpoint；拉取镜像还需仓库读取权限 |

## 已有云资源与镜像

- 项目：`hallowed-pipe-497802-i0`；编号：`580558088344`；区域：`asia-northeast1`。
- Endpoint：`8911645097217490944`；Model：`6657078912276758528`；原 DeployedModel：`1239217774224474112`。
- 页面实际使用下列冻结的**训练/页面镜像**，不是精简预测镜像：

```text
asia-northeast1-docker.pkg.dev/hallowed-pipe-497802-i0/denoise/denoise-vertex@sha256:fac4b16b9454f8b4685743477959cff89175467d001d3407ecefcf2b4747a73f
```

其本机 image ID 应为 `sha256:c5a18e2e2fdda2d1c0b4795c0aba58a11581d268edc4bcf4360853ccbb31efc4`，大小约 28.42 GB；Registry 报告约 14.71 GB。新主机拉取前检查 Docker 数据盘，预留镜像和解压空间。无需将约 28 GB 镜像另打 tar 放到共享 NFS。

专属预测 URL：

```text
https://8911645097217490944.asia-northeast1-580558088344.prediction.vertexai.goog/v1/projects/580558088344/locations/asia-northeast1/endpoints/8911645097217490944:predict
```

当前云端推理使用另一个约 7.12 GB 的精简镜像，只有 Cylinderhead final。英文页面列出 11 个预设，但其余 10 个明确标注未部署并禁止执行。这是现有能力边界，迁移不包括恢复全部模型、全部导出格式或正式长训练。

## 执行顺序

1. 在 work1 核对 `hostname`、内网 IP、`findmnt -T /mnt/home` 与仓库的可写性。确认挂载类型为 nfs/nfs4；未挂载时不要创建同名本地目录或另拷整个仓库。按路径复用当前共享工作区，保留所有提交和本地改动。
2. 检查 Docker、其数据盘可用空间、目标内网端口和现有容器。查找已有合法用户 ADC，只检查类型和读权限，不输出 token、refresh_token 或文件内容。若没有可用身份，先完成不依赖身份的准备，向用户说明需要在 work1 完成 Google 用户登录。不要把凭据写入 NFS 交接目录、Git、镜像或日志。
3. 复用已缓存的精确 digest 镜像；没有时从 Artifact Registry 拉取。若 work1 有已登录且有仓库权限的 gcloud，可使用下方命令。仅 CLI 登录不代表 ADC 已存在；两者分别检查。
4. 用新启动脚本 `--check` 先检查，再创建页面。脚本不拉镜像、不交换 token、不修改云资源、不替换现有容器。默认用当前用户 UID/GID，不硬编码 gpu01 的 1003。
5. 检查页面 HTTP 200、英文控件、11 个模型选项、未部署选项禁用、实际 `/dev/nvidia*` 为空。页面不需要本机 GPU，即使 Docker 默认 NVIDIA runtime 也要保留 `NVIDIA_VISIBLE_DEVICES=void`。
6. 完成一次小图的真实页面→Vertex→下载测试，核对模型 ID、尺寸、16 位 TIFF；再用已验收的 1518×1518 TIFF 验证代表图。记录耗时与结果。这会唤醒 GPU、产生新的可计费用量；不重复进行长时间压力测试或训练测试。
7. 新页面成功后，若能合法访问 gpu01，再经已验证 SSH 连接读取旧容器配置；确认旧容器仍指向相同 Endpoint，且本机新入口已验收后停止 gpu01 的旧页面容器。不要停止同名但用途已改变的容器，不删除镜像或数据。若不能访问旧机，保留旧入口，明确报告待切换项，不关闭其他服务。
8. 将实际新 URL、镜像 ID、无 GPU 证明、预测结果、身份类型/主体（不含凭据）、旧入口是否已停止写到新的迁移记录。另台物理电脑的内网访问若无法代测，明确标记未验证，不宣称跨主机已验收。最终停止额外测试请求，等待副本回零并保存新样本；不把昨天的零点作为今天的证据。

## 拉取与启动命令

以下 gcloud 命令要求 work1 有已登录的 CLI；若没有，参考 `deploy/vertex/README.md` 的 Cloud SDK 容器方式。不要为迁移更换 BasicSR 的 Python/Torch 栈。

```bash
gcloud auth configure-docker asia-northeast1-docker.pkg.dev
docker pull asia-northeast1-docker.pkg.dev/hallowed-pipe-497802-i0/denoise/denoise-vertex@sha256:fac4b16b9454f8b4685743477959cff89175467d001d3407ecefcf2b4747a73f

cd /mnt/home/huang/code/cst_ai/BasicSR
# 用 work1 实际 IP 和本机已有 ADC 路径填写以下两行。
export UI_BIND_IP='<work1 LAN IPv4>'
export UI_ADC_PATH='<absolute path to work1 authorized_user ADC>'

bash deploy/vertex/start-internal-ui.sh --check
bash deploy/vertex/start-internal-ui.sh
docker logs --tail 100 denoise-vertex-ui
```

如目标端口已有其他服务，可显式设置 `UI_PORT`；不自动停止该服务。容器已存在时脚本会拒绝覆盖，应先检查并复用或有针对性地调整。

## 已知问题与验收输入

2026-11-17 已验收：热状态 1518 TIFF 全流程 61.912 秒、云端 46.364 秒；单次 64×64 冷请求全流程 200.293 秒；闲置后约 6 分钟归零。它们是历史证据，不保证今天的可用性。

历史最新控制面读取出现通用 `System error`，当时副本为 0；2026-11-18 英文界面修正没有再触发云端预测。work1 agent 应先读取当前 Endpoint 再验证真实预测。429 冷启动由页面自动重试；遇到持续 5xx、库存或配额异常时记录实际错误，不伪装为迁移成功，也不未经分析重复重建云资源。需要云端重新部署时将其作为独立故障项，使用现有部署说明定位。

代表性输入在共享 NFS：

```text
/mnt/home/huang/cst_ai/experiments/cylinderhead_300kv_mean3_20261104_173035_JST/resource_review/benchmark_input/input_validation.tif
```

原始验收 JSON 已归档：`/mnt/home/huang/cst_ai/archive/basicsr-review-20261118/docs/deploy/`。不要继续假定它们全部位于 Git 仓库 `docs/deploy/`。本机 `--check` 与页面 HTTP 检查都不能代替真实云端推理验收。

## 回退

在新入口验收前保持 gpu01 页面运行。切换后若发现问题，暂停 work1 的页面，确认 gpu01 原配置仍正确后恢复旧容器即可；云资源和 NFS 均保留，无需搬回权重或回滚训练。页面启动/停止不等于云 GPU 停止，云端副本和费用应单独确认。
