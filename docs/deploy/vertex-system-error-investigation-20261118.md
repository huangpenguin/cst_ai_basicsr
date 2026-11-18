# Vertex `System error` 调查（2026-11-18 JST）

## 可确认的结论

报错来自 Vertex API 的 `deployedModels[].status.message`，不是 work1 或 Gradio
生成的错误。该字段按官方定义是最新部署状态消息，不是逐请求预测结果。
[DeployedModel 状态定义](https://docs.cloud.google.com/dotnet/docs/reference/Google.Cloud.AIPlatform.V1/latest/Google.Cloud.AIPlatform.V1.DeployedModel.Types.Status)

本轮合成 64×64 uint16 TIFF 的真实冷启动、热请求均成功；就本次观测而言，
它不是持续阻断推理的模型/ADC 故障。回零后在没有新增预测的情况下，通用消息又出现，复现了“零副本报错 → 就绪后清空 → 回零后再次报错”的关联；这仍不等于已获 Google 确认的内部根因。平台在冷启动末期还曾报告模型服务未就绪，
随后实际预测成功，控制面消息清空。因此不能把此通用状态文案直接解释成
“权重已坏”“登录失效”或“必须重建 Endpoint”。也不能反过来将所有 System error
忽略为正常：持续 5xx、长时间无法就绪等情况仍应独立处理。

## 实际时间线

| 时间（JST） | 实测结果 |
|---|---|
| 18:10:33 | 控制面返回 System error；最新监控样本副本为 0 |
| 18:12:04 | 首次小图请求返回 429，正文明确说明正在从零扩容 |
| 18:12–18:15 | 每约 15 秒进行有上限重试，返回相同的冷启动 429 |
| 18:15:08 | 控制面短暂出现 Model server never became ready；其消息时间为 18:14:52 |
| 18:15:25 | 冷启动请求 HTTP 200；总等待 201.754 秒，云处理 1.635 秒；模型、CUDA、尺寸和 uint16 均匹配 |
| 18:15:26 | 控制面错误消息已清空；第二次热请求 HTTP 200，云处理 0.130 秒；随后停止全部预测 |
| 18:21:14（18:22:23 读到） | 新监控样本确认副本回到 0，距最后请求完成约 347 秒；该次控制面采样仍没有错误消息 |
| 18:25:23 | 在没有新增预测的情况下，System error 再次出现（消息更新时间 18:24:54），最新 18:24:14 副本样本仍为 0 |

本轮只完成两个被接受的合成小图预测，没有上传公司实际扫描、启动训练、
重建模型、修改最小副本数或改变现有 GPU 配置。429 重试请求被平台丢弃。
诊断没有保存 base64 像素或 Authorization header。

Google 文档明确说明 Scale To Zero 下首次请求会收到 429，触发从零扩容，
调用方需要等待再试；没有预约容量时，重新扩容也可能遇到库存不足。
文档没有声明通用 `System error` 文案是零副本的标准状态，不能把我们的
现象观察上升为 Google 已确认的产品缺陷或内部根因。
[官方 Scale To Zero 行为与限制](https://docs.cloud.google.com/vertex-ai/docs/predictions/autoscaling)

## 日志与诊断边界

模型配置仍是已验收的固定推理镜像，`/predict`、`/health` 与 8080 端口匹配当前服务源码；没有配置自定义 startup/health probe。仅凭这些配置不能确定哪一次底层探测触发了通用消息。

当前部署设置 `disableContainerLogging=true`。按 Endpoint ID 与审计资源名
查询当天日志，没有找到匹配记录。没有日志不能证明底层没有错误，且事后
开启日志也无法恢复此前的启动日志。本轮没有改动线上日志或自动扩缩容设置。

因此已查明错误的来源、此次与启动状态的关联及实际预测是否可用；
**Google 内部为何在这些阶段产生通用消息、是否属于平台状态呈现缺陷，仍不能
仅凭当前可见信息确定。** 若需要 Google 给出内部原因，可用本文时间、区域、
Endpoint/DeployedModel 和脱敏时间线提交支持工单；本轮没有代发工单。

资源定位：项目 `hallowed-pipe-497802-i0`，区域 `asia-northeast1`，
Endpoint `8911645097217490944`，DeployedModel `1239217774224474112`。

前端继续原样显示平台消息，并明确提示“状态刷新没有测试预测”。
保留已有 429 冷启动重试，不以状态消息单独禁用正常推理，不隐藏实际错误。
需要进一步复现持续启动失败时，应先评审启用容器日志的局部配置变更，
采集一轮启动日志；不要直接重建 Endpoint 或把 min replicas 改为 1 以掩盖现象。

完整脱敏时间线见 [机器可读证据](vertex-system-error-evidence-20261118.json)。监控 response_count 在本轮区间记录了 2 个 HTTP 200，与诊断程序的两次成功预测一致。
