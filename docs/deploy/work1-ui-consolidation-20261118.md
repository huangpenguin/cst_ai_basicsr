# work1 界面整合、目录与环境（2026-11-18 JST）

当前入口：http://192.168.3.11:7861 。旧界面未丢失，源码一直保留在
`apps/denoise_demo/`；当前 `deploy/vertex/internal_ui.py` 已整合其元数据展示能力。

## 已合并并部署

- 保留英文界面、11 个模型目录，以及未部署模型禁用规则。
- 模型详情：复用 `checkpoint_info.inspect_preset`，在 CPU 上读取模型族、架构、权重大小、时间、部分哈希、参数键和训练配置元数据。明确标注这是前端镜像中的目录副本，不代表云端运行状态。
- GPU 详情：读取实际 Vertex Endpoint 的机器型号、GPU 类型/数量、模型、扩缩容配置、控制面状态和最新副本监控样本。显示样本时间与是否过时；无样本时标记未知，不伪造零副本。
- GPU 状态刷新不调用预测，也不读取 work1 的本机 NVIDIA 设备。现有 Endpoint 未暴露每进程显存详情，界面明确说明这个限制。
- 图片详情：上传即可显示预览、尺寸、位深/dtype、格式、页数、文件大小、像素最小/最大/均值；推理完成后显示输出详情。预览为 8 位，原始输入不改写，下载继续为 16 位 TIFF。
- 新上传会清除上一次输出，避免混淆；多页输入明确要求选择切片，不静默只取第一页。

旧界面的任意本机 checkpoint 路径、模型族切换和本机 GPU 调用未接入云端：当前服务端只部署 Cylinderhead final。旧 8 位 PNG/float TIFF 导出选择未恢复；云端返回 uint16 TIFF，不能把量化后的文件伪称为原生浮点推理结果。以后扩展模型/格式需单独修改服务端契约并验收。

## 目录归并与旧部署退役

主 Git 工作副本：`/mnt/home/huang/code/cst_ai/BasicSR`，确认位于可写 NFS：
`192.168.3.14:/mnt/storage/home`。会话在应用中仍关联 `lab-docker`，本轮实际文件操作均显式指定 BasicSR，未修改 Docker 项目。

`/home/huang/scripts` 下旧 `basicsr-demo`、三个 `cylinderhead-*` 兼容链接已在此前移除；原代码与历史分析产物保留在主仓库和 NFS 归档。此次更新了过时 README，避免继续指向已退役的 7860。
旧 README 完整备份在 `/mnt/home/huang/cst_ai/archive/scripts-closeout-20261118/scripts-readme-before-ui-consolidation.md`。

旧 work1 `denoise-demo` 继续停止，并将 restart policy 设为 `no`。
历史 Compose 增加 `legacy-local-gpu` profile、关闭自动重启；默认 Compose 启动不包含该服务。
容器和镜像保留为历史回退材料；共享权重和数据未删除。其他 scripts 项目保持原状。

## Python / Docker 环境

| 用途 | 实际位置 / 环境 |
|---|---|
| 宿主管理脚本 | work1 `/home/huang/miniconda3/bin/python3`，Python 3.13.2；JSON/路径/子进程等标准库，用于生成配置、调用 Docker/CLI |
| 当前 Gradio | work1 Docker 容器 `denoise-vertex-ui`，`/workspace/.venv/bin/python`，Python 3.11.11、Gradio 6.25.0、Torch 2.6.0+cu124；CUDA available=False |
| 云 CLI | work1 上的固定 Cloud SDK Docker 镜像，读取本地只读用户 ADC |
| 模型推理 | 东京 Vertex GPU 的固定服务镜像 |
| 真实短训练 | 东京 Vertex GPU 的固定训练镜像；已完成的一次短训练详见验收报告 |

所以命令里的 `python3` 不表示拿宿主普通环境直接训练。主机负责提交，Torch 与 CUDA 依赖由 Docker 镜像固定。Docker 镜像与容器文件系统在 work1 本地；源码在 NFS；数据/权重在各自 NFS 路径或私有 GCS；ADC 保存在 work1 本地私有目录，不进入 Git。

页面使用原固定镜像 digest `fac4b16b9454f8b4685743477959cff89175467d001d3407ecefcf2b4747a73f`，只读挂载当前 `internal_ui.py`。本轮未重建云资源或镜像，只重启 CPU 前端加载改动。

## 验证与 Git 状态

六项 CPU 回归检查覆盖原始 uint16 不变、多页拒绝、未部署模型拒绝、状态读取不预测/不伪造零值、元数据不误作文件下载、原 infer API 下载契约。
真实运行页面验证了上传详情、CPU checkpoint 元数据、Vertex 配置/副本读取和禁用模型状态；未额外发送预测或启动训练。
实际证据：`work1-20261118/ui-consolidation.json`、`work1-20261118/ui-consolidation-tests.txt`。

控制面读取仍返回已知 `System error`，新版面板保留原消息；近期副本监控样本为 0。
因此本次验证的是界面及只读状态集成，不是新的付费推理验收。此前真实推理结果见迁移记录。

本轮通过 `git ls-remote` 验证 GitHub 与 GitLab 的 master 都为
`fcca7f5a210659df4cd8324af09faee611afe22f`。这是初次界面整合时的远端快照，当时新增改动尚未 commit/push。之后用户已明确要求双端发布；发布审查与故障调查见 `release-review-20261118.md`，当前版本以 Git 提交记录为准。
