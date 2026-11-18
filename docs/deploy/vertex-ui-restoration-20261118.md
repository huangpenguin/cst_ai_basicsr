# 内网 Gradio 修正与迁移准备（2026-11-18）

2026-11-18 源码整理：文中未提交／未推送的描述是验证当时的状态。源码与运维说明纳入 Git，原始运行记录保存在 NFS 的 `/mnt/home/huang/cst_ai/archive/basicsr-review-20261118/`；本次整理未重新部署或发起云端预测。

## 本轮修正

`deploy/vertex/internal_ui.py` 恢复英文标题、Preset model、Upload image、Denoise、Input、Denoised、下载文字，以及等待、状态和错误信息。模型下拉框显示原来的 11 个预设；默认使用当前云端可用的 Cylinderhead mean3 FINAL step 30000。

当前 Vertex 推理镜像仍只有这个正式缸盖模型，其他预设明确标为 `[not deployed to Vertex]`，选择后显示英文原因并禁用 Denoise。后台同时拦截未部署选择，并核对响应中的模型 ID，避免界面显示一个模型而实际执行另一个。

预览恢复原界面的线性映射：8 位输入原样显示，16 位输入按预设归一化范围映射至 8 位。移除了本轮部署时额外加入的百分位拉伸。下载仍是云端返回的 16 位 TIFF。

**本轮恢复了英文和模型选择目录，并没有声称所有原功能已经可用。** 多模型云端分发、自定义 checkpoint 路径、原版全部导出格式、云 GPU 状态展示和完整 checkpoint 元数据面板，仍需后续适配。原应用 `apps/denoise_demo/app.py` 保留，没有被本轮改写。

页面容器 `denoise-vertex-ui` 已在 gpu01 重启。检查包括英文组件配置、11 个模型选项、未部署选项禁用、已部署选项启用、模拟预测下载、响应模型核对和原始灰度映射。真实 Vertex 预测没有在本轮重新执行；旧记录中的平台系统错误也未通过本轮界面检查消除。

## 代码和权重的实际位置

| 位置 | 已确认内容 |
|---|---|
| `/mnt/home/huang/code/cst_ai/BasicSR` | NFS4 共享代码，源为 `192.168.3.14:/mnt/storage/home`，可写 |
| `/mnt/home/huang/cst_ai` 与仓库内 pretrained_models | 原 11 个预设的 checkpoint 全部存在，均位于共享 NFS |
| `/mnt/data/cst_ai` | 训练数据位于另一共享 NFS，源为 `192.168.3.14:/mnt/storage/data`，不是 gpu01 本地盘 |
| Artifact Registry 的 `denoise-vertex:20261117` | 已上传的训练/页面镜像实际包含 11 个预设权重和代码；不是仅上传了缸盖权重 |
| 当前在线 `denoise-vertex-inference:20261117` | 精简推理镜像只包含缸盖 final，Endpoint 也只有默认单预设协议 |
| GCS 的 `vertex-smoke/20261117-run1/` | 两步真实训练生成的 latest、日志、有效配置和 summary；不是全部正式模型权重库 |
| Git | 当前部署改动仍未提交或推送；上传 Docker 镜像不等于提交 Git |

因此，恢复全部原有模型不能只增加下拉框。还需将其他预设权重纳入运行中的推理镜像或获准访问的云存储路径，并使预测接口按照所选 preset/family 分发。原来的权重已经在上传过的训练镜像中，可复用，不必先重新从 NFS 上传一遍到 GCS。

云端训练已经具有上传过的执行代码，但仅验证了 4 对真实样本、2 次更新。完整数据、正式 GPU/batch/DDP 设置、周期产物持久化和恢复仍需适配，不能把本次测试当作正式训练可直接开始。

## 部署位置的理解

```text
浏览器 → 实验室服务器上的 CPU Gradio → HTTPS / IAM → Google Vertex GPU
```

页面服务在实验室服务器；GPU 推理服务和训练 worker 在 Google Cloud。Google 不直接读取实验室 NFS，使用的是上传的镜像和请求传来的图像。IAM 负责核对服务器后台代表哪个 Google 身份调用，浏览器不会取得该用户凭据。当前后台用 Huang 的用户 ADC，未另外实现独立网页登录。

## work1 迁移准备

用户指定目标为 work1，并说明稍后再通知。本轮**没有连接、修改 work1，没有停止 gpu01 入口，也没有复制或删除共享代码、数据或凭据**。

Vertex Endpoint、Model、Artifact Registry 和 GCS 均在云端，不绑定 gpu01，无需因为更换内网页面主机而重建这些云资源。

| 可复用部分 | 需在 work1 单独确认的部分 |
|---|---|
| 共享 NFS 代码、原始权重和训练数据 | 确认两处 NFS 真正挂载且权限正确；不使用未挂载的同名空目录 |
| 已上传 Docker 镜像 | work1 的 Docker 可用性、磁盘、拉取权限和本机镜像缓存 |
| 已有 Vertex 预测地址 | work1 到 Google OAuth 和预测 HTTPS 地址的网络访问 |
| 原页面实现与启动参数 | work1 内网 IP、端口占用、防火墙、文件 UID 和进程运行用户 |
| 云资源与已有任务记录 | 合法可用的后台认证；gpu01 的 `/home/huang/.config/pai_ai_control/gcloud/` 在本地 XFS 上，不是共享 NFS |

收到迁移通知后，先在 work1 启动新入口，检查英文界面、无本机 GPU、云端实际预测与另一台电脑的访问，再切换使用地址并停止 gpu01 页面。保留 NFS 工作区、未提交改动和云资源，不需要移动或删除原目录。凭据应使用获准的认证方式，不写入 Git、镜像或共享公开目录。
