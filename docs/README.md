# 文档索引

| 文档 | 内容 | 什么时候看 |
|------|------|------------|
| [workflow.md](workflow.md) | 目录速查、存储布局、数据准备、训练（本地 / CI / 续训 / TensorBoard）、评测、Demo | 日常操作 |
| [experiments.md](experiments.md) | 历次实验的参数、路径、命令、评测结果与当时的需求原话 | 查某次实验怎么做的、结论是什么 |
| [infrastructure.md](infrastructure.md) | GPU 镜像、Dev Container / compose、数据挂载契约、GitLab Runner、代码质量、Agent 规则 | 搭环境、改 CI、排查 Runner |
| [research-notes.md](research-notes.md) | 数据与伪影特性、评价标准、3D 切片划分、过度平滑的改进方向、开源方案调研 | 规划下一步实验 |
| [upstream/](upstream/) | BasicSR 上游原版文档（中英）与 Sphinx 配置 | 查 BasicSR 框架本身的用法 |

其他就近放置的说明：

- [`apps/denoise_demo/README.md`](../apps/denoise_demo/README.md)：Gradio 演示部署与模型注册
- [`.devcontainer/README.md`](../.devcontainer/README.md)：Dev Container 操作手册
- [`experiments/pretrained_models/README.md`](../experiments/pretrained_models/README.md)：预训练权重
- `.cursor/project-context/decisions.md`：给 Agent 读的稳定决策短记（英文）

新增内容时：操作流程写进 `workflow.md`，实验记录追加到 `experiments.md` 末尾，不要再新建平行文档。
