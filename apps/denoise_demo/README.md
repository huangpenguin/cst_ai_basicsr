# CT 灰度去噪演示（Gradio）

独立网页界面，目录：`apps/denoise_demo/`。上传灰度图 → GPU 上用进程内缓存的模型去噪 → 并排显示 **输入 | 去噪结果**。模型族为 `swinir_m`（SwinIR-M）和 `tsfdan`（TSFDANSwinIR），二者的 `state_dict` 不能混用。

## 启动（本机 / SSH 终端，不必 Docker）

在仓库根目录：

```bash
uv sync
uv run python apps/denoise_demo/app.py
```

监听 `0.0.0.0:7860`，`share=False`（只走局域网/VPN，不是 Gradio 公网分享）。浏览器打开 `http://<GPU机器内网IP>:7860`。

### 用 tmux 保持服务不随 SSH 断开而退出

**tmux 本身几乎不占资源**（几 MB 内存）。占 CPU / GPU / 显存的是里面的 `python apps/denoise_demo/app.py`：空闲时大约占一两 GB 显存（已加载的权重），有人点「去噪」时才会打满算力。关掉 Python 进程就释放；只关 SSH、tmux 还在跑时，资源占用不变。

```bash
tmux new -s denoise-demo
cd /path/to/BasicSR
uv run python apps/denoise_demo/app.py
# 脱离会话（服务继续跑）：Ctrl-b 然后按 d
# 回来看日志：tmux attach -t denoise-demo
```

## Docker 部署是什么意思（本仓库语境）

Docker **不是**再装一套操作系统给你点网页。它是：把「CUDA + 系统库 + 本项目 Python 环境」打成一个**可重复的盒子（镜像）**，在盒子里跑同一个 `uv run python apps/denoise_demo/app.py`。

| 概念 | 白话 |
|------|------|
| 镜像 (image) | 只读模板。本仓库是 `basicsr-mlops-gpu:2.6.0-cu124`，和 GitLab CI / Dev Container 同一套。 |
| 容器 (container) | 用镜像跑起来的一个进程隔离环境。代码通过 volume 挂进容器（compose 里是 `.:/workspace`）。 |
| 端口映射 `-p 7860:7860` | 容器内 Gradio 听 7860；宿主机同一端口转进去，浏览器才能访问。 |
| GPU | compose 里 `gpus: all`，容器才能用 NVIDIA 卡。 |

**你现在已经在 GPU 机器上能 `uv run` 的话，不必为了 Demo 专门「部署 Docker」。** README 写 Docker，是因为有人只在 Dev Container / `docker compose` 里开发：容器默认 **没有** 把 7860 暴露出去（见仓库根目录 [`docker-compose.yml`](../../docker-compose.yml) 没有 `ports:`），所以浏览器打不开，除非：

- 启动时加 `-p 7860:7860`，或
- compose 的 `services.dev` 下增加 `ports: ["7860:7860"]`，或
- `network_mode: host`（和 TensorBoard 一样，容器直接用宿主机网卡，端口就是机器上的 7860）

容器里 checkpoint 路径：YAML 里写的 `/mnt/home/huang/cst_ai/...` 若不存在，会自动试 `/home/huang/cst_ai/...`（宿主机 NFS ↔ 容器挂载）。

环境变量（可选）：

- `DENOISE_DEMO_MODELS_YAML`：换一份模型注册表 YAML
- `DENOISE_DEMO_DEFAULT_ID`：覆盖默认预设 id
- `DENOISE_DEMO_IDLE_SECONDS`：模型空闲多少秒后自动卸载并释放显存（默认 `600`，`0` 关闭）。后台线程每 15 秒检查一次，与浏览器是否还开着无关；下次点「Denoise」会自动重新加载（多花几秒）。每次推理结束也会 `empty_cache()`，空闲时只占权重显存。

## 增加一条权重

编辑 `apps/denoise_demo/models.yaml`：增加 `id`、`label`、`family`、`ckpt`（绝对路径或相对仓库根）。可选 `param_key`（`params` / `params_ema` / 省略则由 loader 自动探测）、`train_yaml`、`normalization`（`normalization.json` 路径或内联 JSON）、`notes`。改完后**重启** Demo。

文件不存在仍会列在下拉框，界面显示 `exists: false`，不会静默换成别的权重。

## 图像约定

- 8-bit PNG/JPEG：灰度，除以 255 → `[0, 1]`
- uint16 TIFF：`(x - raw_min) / (raw_max - raw_min)`，裁剪到 `[0, 1]`（默认 `0` / `65535`）
- RGB（`ndim == 3`）：取 **第 0 通道**（与 `scripts/infer/infer_swinir_graydn_tiff.py` 一致）

网页预览始终为 8-bit：`(clip * 255).round().astype(uint8)`。

## 输出格式与下载

界面「Download format」单选后点「Denoise」，结果文件出现在下方「Download denoised image」，文件名 `<输入名>_denoised.<png|tif>`。编码由 `scripts/infer/output_formats.py` 的 `OutputFormat` 提供（与命令行 `--output-format` 共用）：

- `png8`：8-bit PNG
- `tiff16`（默认）：uint16 TIFF，按预设的 `raw_min`/`raw_max` 还原到输入原始值域
- `tiff32`：float32 TIFF，`[0, 1]` 归一化的模型原始输出

官方 SwinIR-M 灰度去噪权重预设有 noise15 / noise25 / noise50 三档（数值越大去噪越强）。
