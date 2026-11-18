# Gradio 基础镜像与历史本地验证

## 历史验证：gpu01 本地服务（已停止）

2026-11-17 用户已澄清：训练和推理都必须使用 Vertex AI 云端资源，只供内部使用。本地运行是对需求的误解，临时创建的 `denoise-demo` 容器现已停止；以下仅保留已完成的镜像验证记录。当前 Vertex 适配见 [deploy/vertex/](../vertex/README.md)，具体部署结果以对应运行记录为准。此前本机验证地址为 `http://192.168.3.18:7860`，现不提供服务。

| 项目 | 当前值 |
| --- | --- |
| 容器 | `denoise-demo`（已停止） |
| 镜像 | `denoise-demo:local`，ID `sha256:0a51e440d8efa9cfbc4b54860a2efddfd34b081a0dbf8d12dd7aaafdae3f41a0` |
| GPU | 宿主机 GPU 0，Tesla V100 32GB，容器只可见这一张卡 |
| 端口 | 仅发布到 `192.168.3.18:7860`，容器内为7860 |
| 默认预设 | 缸盖 final step 30000（`swinir_cylinderhead_mean3_20261104_173035`） |
| 重启策略 | `unless-stopped`，Docker 启动时恢复；手动停止后保持停止 |

本机内网地址的页面与配置接口均 HTTP 200。已通过 HTTP 上传、推理并下载64×64小图及历史1518×1518的16位TIFF；大图请求耗时17.222秒，下载文件尺寸和16位格式正确。详细配置及验证结果见 NFS 归档 `/mnt/home/huang/cst_ai/archive/basicsr-review-20261118/docs/deploy/local-deployment-20261117.json`。跨物理机访问尚未实测。

维护命令：

```bash
docker logs --tail 100 denoise-demo
docker stop denoise-demo
docker start denoise-demo
```

需要重新创建容器时，先核实同名容器及端口状态，不要删除其他工作。实际启动命令的等价形式如下（已有容器运行时无需重复执行）：

```bash
docker run -d --name denoise-demo \
  --gpus device=0 \
  --restart unless-stopped \
  -p 192.168.3.18:7860:7860 \
  denoise-demo:local
```

应用监听 `0.0.0.0:7860`，Gradio `share=False`，宿主仅绑定上述内网地址。该地址是 gpu01 的新增部署；仓库其他文档中 `192.168.3.11:7860` 指向 work1 上的既有部署。

## 构建入口

此镜像为 `deploy/vertex/Dockerfile` 提供应用、依赖和权重。先在 gpu01 的仓库根目录准备预训练权重及 NFS 实验权重，再构建：

```bash
bash deploy/serve/stage-weights.sh
docker build -f deploy/serve/Dockerfile -t denoise-demo:local .
```

`experiments/pretrained_models/` 中的官方模型需先按仓库下载脚本准备；Git 只保存下载入口，不包含权重。`deploy/serve/weights/` 为构建暂存区，已忽略。

当前云端部署入口见 [Vertex 说明](../vertex/README.md)。未采用的 Compute Engine 方案和 `vm-startup.sh` 保存在 `/mnt/home/huang/cst_ai/archive/basicsr-review-20261118/deploy/serve/`，不作为当前部署方式维护。
