# 内网扫描交付与配对审核操作说明

日期：2026-11-19（日本时间）。运行入口：http://192.168.3.11:7862 。去噪入口继续使用 http://192.168.3.11:7861 。代码与部署统一属于 BasicSR，不再建立独立实验项目。

## 同事怎样上传

1. 点击 **New scan batch**，填写批次名称、提交人、工件 ID 和扫描组。同一个工件、同一方向的对应扫描使用同一个扫描组，不把电流写成不同扫描组。
2. 多选 TIFF 或 ZIP。推荐每个 ZIP 对应一种拍摄条件，GT 独立一包；每项填写真实条件，并选择 **Noisy (LQ)** 或 **Reference (GT)**。必要时展开工件/扫描组字段。
3. 点击 **Upload selected files**。浏览器先分块计算完整 SHA256，再开始续传；初次计算大文件校验值需要时间。可暂停，关闭页面后用批次 ID 和访问码恢复，再选择同一个源文件。先保存访问码，不能发给无关人员。
4. 收齐文件后点击 **Check files and request review**。`Uploaded` 仅表示传输完成；后台核对源 SHA256、解包 CRC 和 TIFF 格式后才进入 **Awaiting review**。原件保留在 NFS；此时不会同步 GCS，也不会训练。

推荐的交付形状如下，文件夹名不要求与示例相同，但切片编号必须真实可对应：

```text
300kV_10uA.zip       XY/data_0001.tif, XY/data_0002.tif, ...
300kV_40uA.zip       XY/data_0001.tif, XY/data_0002.tif, ...
Reference_GT.zip    XY/data_0001.tif, XY/data_0002.tif, ...
```

也可将一个条件分成多个 ZIP，保持相同条件、工件和扫描组标记。目录中的 XY/XZ/YZ 用于区分方向；未知方向保留为 unspecified，不猜测坐标含义。JSON/CSV/TSV 等 sidecar 保留但不自动解释，页面会提示审核采集字段。加密/嵌套归档、路径逃逸、链接、损坏 ZIP 不进入可用图像集合。

## 你怎样审核

点击 **Reviewer sign in**。密码只保存在 work1 的 `/home/huang/.config/basicsr-xray-portal/admin-password.txt`，通过你自己的服务器终端读取；不要把密码提交到 Git 或聊天。此期是已选择的内网 HTTP 试用，尚无传输加密及公司单点登录。

审核分三步：

1. **核对输入**：查看各条件的可用切片数量、缺失与重复项、图像格式、尺寸、方向和元信息。错误上传可以 **Exclude**，填原因后补传正确文件；原件不删除，校验和审核结果会失效，需要重新校验。
2. **核对配对**：逐对查看 LQ/GT、灰度范围、饱和比例和拍摄条件，检查方向、边缘及结构是否对应。自动匹配按工件、扫描组、方向、切片编号提出候选，不按 ZIP 顺序，也不把高电流自动认定为 GT。候选不唯一时手选 GT；位置/尺寸不符合的映射仍会被后台拒绝。一个 GT 可以对应多个条件的 LQ。确认后勾选 **Pair checked and approved**；异常对保持 Exclude。512 像素预览使用共同的 0..65535 灰度窗口，细节和精确对齐应下载 **Original TIFF** 检查。系统不会自动证明空间配准正确。
3. **核对划分**：优先将完整工件放入同一个 train/monitor/test。仅有一个体数据时选择 **Single volume: continuous slice blocks**，按连续切片范围分配全部条件，留出隔离带；默认 16 层需要结合实际扫描审核，不代表获得独立工件测试。每个纳入数据的 pair 仍须批准。填写审核理由，确认异常项已处理，点击 **Save review**。

**Freeze approved dataset** 会再次检查每个候选的决定、配对有效性、源文件是否改变、内容重复、跨集泄漏、隔离带、train/monitor 非空和磁盘空间。失败时显示原因，可修改审核后重试。成功后版本不可修改；审核页面保持只读，可以继续查看原图、映射及划分。

最后由你单独点击 **Sync approved version to GCS**。只有出现 **Cloud Ready** 才代表远端大小、CRC32C 和清单已验证。上传失败可对同一版本重试；不需要复制数据到另一个版本。原始 ZIP 留在 NFS，云端只接收训练归档和溯源清单；test 图像不进入训练归档。页面没有训练提交和模型替换接口。

## 数据管理与运行边界

- 原始上传、派生预览和冻结版本位于真实共享 NFS `/mnt/data/cst_ai/intake/<batch_id>/`。
- 云端续传会话单独放在 work1 本地 `/home/huang/.local/state/basicsr-xray-portal-cloud`，仅 worker 可见；不会挂载给网页 API。
- 数据库、任务、上传偏移及锁在 work1 本地 `/home/huang/.local/state/basicsr-xray-portal`；ADC 仍在已有本地私有路径，只挂载给云同步 worker。
- NFS 不可用时拒绝创建批次或开始写入，不创建同名本地目录。已有 NFS hard mount 网络故障可能使进行中的 I/O 等待，恢复后继续；HTTP 超时不等于数据已提交。
- 默认单文件 32 GiB、单批原件 64 GiB、解包 128 GiB、50,000 文件、单图 1 亿像素；导出逻辑大小另限 128 GiB，计入共享 GT 的重复副本。单 worker、每个任务最多两小时，CPU/内存受容器限制。
- 支持单页二维 uint16 TIFF；多页、其他位深、RAW 转换、配准及 GT 平均不在本期。接收文件不等于其内容能用于训练。
- 不自动删除原件。任务强制中断可能留下 `.partial` 派生目录，不会被当作成功结果；管理员可核对任务后清理明确的派生临时项。不要清理 original、数据库或续传状态来“修复”失败。
- 备份应同时覆盖 NFS 原件/版本与本地 SQLite、tus 元数据。数据库使用 SQLite backup API 得到一致快照；私有状态备份也必须保存在受限位置，不放 Git、公开共享目录或浏览器。

启动、状态、重启和 GCS 续传验收命令见 [部署说明](../../deploy/internal/README.md)。正式训练继续由管理员按 [训练操作说明](../../deploy/vertex/training/README.md) 手动提交。

本次已执行的真实验收、性能观察与边界见 [验收报告](xray-portal-acceptance-20261119.md)。
