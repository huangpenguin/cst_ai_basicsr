# 实验记录

> 用途：以后查阅参数 / 输入输出 / 脚本命令 / 当时用户要求。  
> 存储约定：共享数据 `/mnt/data/cst_ai`（只读倾向）；个人产出 `/mnt/home/huang/cst_ai`（可写）。  
> 第 0–8 节为 2026-07 SwinIR 配对 TIFF 系列；第 9 节起为 2026-08 的 ALcast 与 TSFDAN。新实验请在末尾追加章节，不要改写历史结论。  
> 对话归档：Cursor agent transcript 052c56e5…（本地，仓库外）。

## 0. 总览时间线

| 阶段 | 约日期 | 产物 |
|------|--------|------|
| Phantom FT 加长训练 | ~07-22 | `aligned_drive` + `…_bs8_30k` YAML 系 |
| 早期模型对比上传 | ~07-23 | Drive `cylinder_train10_model_compare` |
| 固定评测 v1 + Cylinder90 训练 | 07-24 ~ 07-26 | `eval_benchmark_v1`、Drive `eval_benchmark_v1_3models_20260726` |
| Mix 训练 + 评测 v2 | 07-27 ~ 07-29 | `cylinder_phantom_mix`、`eval_benchmark_v2`、Drive `eval_benchmark_v2_4models_20260729` |
| 传统 CV 基线 | 07-29 | `eval_benchmark_cv` |
| 脚本单一化规则 | 07-29 | `.cursor/rules/singular-scripts.mdc` |
| EPI + 残差 panel 增强（无 CNR） | 07-29 | `enrich_eval_benchmark_panels.py`；训练 loss=**MSE 非 L1** |

---

## 1. 公共约定

### 1.1 PreFT 权重

```text
experiments/pretrained_models/SwinIR/004_grayDN_DFWB_s128w8_SwinIR-M_noise25.pth
```

### 1.2 推理面板

- DL：`LQ | PreFT | FT | GT`（`scripts/infer/infer_swinir_graydn_tiff.py --baseline-model …`）
- 默认 `--tile 256 --tile-overlap 32`（与训练 val 一致）
- CV：`LQ | <method> | GT`

### 1.3 脚本入口（单一化后，2026-07-29）

| 能力 | 唯一脚本 |
|------|----------|
| 建固定评测树 | `scripts/eval/build_eval_benchmark.py` |
| 跑某模型全列推理 | `scripts/eval/run_eval_benchmark_model.sh` |
| 填对比表 | `scripts/eval/fill_eval_benchmark_table.py --root … --preset v1\|v2` |
| 传统 CV 基线 | `scripts/eval/run_eval_benchmark_cv.py` |
| 建 mix 数据集 | `scripts/data/build_cylinder_phantom_mix.py` |
| EPI/残差/大字体 panel（不重推） | `scripts/eval/enrich_eval_benchmark_panels.py` |

规则：`.cursor/rules/singular-scripts.mdc`（同功能脚本不复数；用 flag / `--root` 扩展）。

### 1.4 CI 训练

- Job：`train_paired_tiff_swinir`（手动）
- 换实验：改 `.gitlab/ci/train.yml` 里 `TRAIN_OPT` 指向的 YAML
- 07-29 写本节时指向 mix（2026-08 起改为 ALcast，见第 9 节）：

```yaml
TRAIN_OPT: "options/train/SwinIR/train_SwinIR_grayDN_cylinder_phantom_mix_bs4_30k.yml"
```

---

## 2. 数据集

### 2.1 `aligned_drive`（Phantom）

| 项 | 值 |
|----|-----|
| Flat LQ/GT | `/mnt/data/cst_ai/datasets/aligned_drive/` |
| Meta | `/mnt/home/huang/cst_ai/datasets/aligned_drive/` |
| 来源 | Google Drive 整理；`folder_roles.yaml` 切片范围过滤 |
| 备注 | `split_mode=tail` → `val.txt` 几乎全是 group **5-4（40kV）** |

### 2.2 `cylinder_260520_gt90ua`

| 项 | 值 |
|----|-----|
| Flat（hardlink） | `/mnt/data/cst_ai/datasets/cylinder_260520_gt90ua/` |
| Meta + SPLIT_RECORD | `/mnt/home/huang/cst_ai/datasets/cylinder_260520_gt90ua/` |
| GT | `1-110kV_90uA` |
| 排除 | `0-300kV_100uA`（对不齐） |
| 切片 | Z **130–750**（仅 260520） |
| Split | train **3912** / val **435** |
| Holdout（不训练） | `cylinder_260527_eval_gt60ua` |

### 2.3 `cylinder_phantom_mix`

| 项 | 值 |
|----|-----|
| Flat（hardlink） | `/mnt/data/cst_ai/datasets/cylinder_phantom_mix/` |
| Meta | `/mnt/home/huang/cst_ai/datasets/cylinder_phantom_mix/` |
| 前缀 | `cyl__` ← Cylinder90；`ph__` ← aligned_drive |
| Split | stratified **domain×µA**；train **11213** / val **1242**；共 12455 |
| 排除 | `cylinder_260527_*` 不进 mix |
| 建集命令 | `uv run python scripts/data/build_cylinder_phantom_mix.py` |

---

## 3. 训练实验

### 3.A Phantom FT（既有，本系列对比行之一）

**用户相关要求（摘录，07-22）：**

> 验证集…想和微调前的对比一下效果…输出在一起的并列图，注意之前有类似的推理验证脚本如果不复用记得删了；  
> 复制 yaml：`val_freq`/`save_checkpoint_freq` 1000 或 2000；`batch_size_per_gpu: 8`；`total_iter: 30000`；`milestones: [15000, 25000]`。

| 项 | 值 |
|----|-----|
| 权重（对比用） | `/mnt/home/huang/cst_ai/experiments/aligned_drive/train_SwinIR_grayDN_paired_tiff_noise25_P128W8/models/net_g_latest.pth` |
| 标签 | `phantom_aligned_drive` |

### 3.B 旧 Cylinder archive（只推理，不重训）

| 项 | 值 |
|----|-----|
| 权重 | `/mnt/home/huang/cst_ai/archive/cylinderblock_2673001900/net_g_latest.pth` |
| 标签 | `old_cylinder_archive` |
| 注意 | 当时 GT≈300kV + 几何变换；相对现 90/60µA 无变换评测会吃亏 |

### 3.C Cylinder90（新训）

**用户要求（摘录，07-24 / 07-24 补充）：**

> 复制 yaml 改进 cylinder：换用 **90µA 作 GT**，其余电流作 LQ；**300kV 不再加入**；硬链接与 output **不要覆盖**上次；留下分割记录。  
> 训练数据只用 **260520 slice 130–750**；推理从 260520 val 与 260527 各选 **20**；battery 有多少用多少、不合并；表格三模型行。

| 项 | 值 |
|----|-----|
| YAML | `options/train/SwinIR/train_SwinIR_grayDN_cylinder_gt90ua_bs8_30k.yml`（文件名含 bs8，**实际 bs=4**） |
| `name` | `train_SwinIR_grayDN_cylinder_gt90ua_noise25_P128W8_bs4_i30k` |
| `batch_size_per_gpu` | **4**（曾试 8 → ~31.65GiB/GPU OOM；见 `.cursor/lessons-learned/2026-07-25-swinir-bs8-oom-32gib.md`） |
| `total_iter` | 30000 |
| `gt_size` / `window_size` | 128 / 8 |
| `val_freq` / `save_checkpoint_freq` | 2000 |
| `scheduler.milestones` | [15000, 25000] |
| `experiments_root` | `/home/huang/cst_ai/experiments/cylinder_260520_gt90ua` |
| 权重 | `…/train_SwinIR_grayDN_cylinder_gt90ua_noise25_P128W8_bs4_i30k/models/net_g_latest.pth` |
| 标签 | `new_cylinder_gt90ua` |
| 启动 | 改 `TRAIN_OPT` → 手动跑 CI `train_paired_tiff_swinir` |

### 3.D Cylinder+Phantom Mix（第四行）

**用户要求（摘录，07-27）：**

> 追加第四行：和 Cylinder90 一样，但 **把 phantom 也加进来一起学**；结果上传 Drive。  
> 想知道为何评测 cylinder 都是同一电流；希望多电流，但 **不能把训练集选进来**。  
> 若不占额外空间就 **hardlink 建混合集**、起好名字；评测 **泄漏优先、不强求多电流**；旧三模型不重训，换了测试集就重跑推理。

| 项 | 值 |
|----|-----|
| YAML | `options/train/SwinIR/train_SwinIR_grayDN_cylinder_phantom_mix_bs4_30k.yml` |
| `name` | `train_SwinIR_grayDN_cylinder_phantom_mix_noise25_P128W8_bs4_i30k` |
| 超参 | 与 Cylinder90 同档（bs=4, 30k, milestones 15k/25k, val/save 2k） |
| `experiments_root` | `/home/huang/cst_ai/experiments/cylinder_phantom_mix` |
| 权重 | `…/models/net_g_latest.pth` |
| 标签 | `cylinder_phantom_mix` |

```bash
# 建集（示例）
uv run python scripts/data/build_cylinder_phantom_mix.py

# CI：TRAIN_OPT 指向 mix YAML 后手动触发 train_paired_tiff_swinir
```

---

## 4. 固定评测集

### 4.1 `eval_benchmark_v1`（三模型交付）

**用户要求（摘录，07-24）：**

> 固定推理验证数据：battery(5/20,5/27)、cylinder(5/20,5/27)、phantom；b/c 必须从 **val** 抽，勿用 train；报告指标给拍 X 光客户解释；表格填三模型行。

| 列 | 内容 | 张数 |
|----|------|------|
| `battery_260520` | 最高 µA 作 GT，其余 LQ | 9（有多少用多少） |
| `battery_260527` | 同上 | 6 |
| `cylinder_260520_val20` | cyl90 `val.txt` spread | 20 |
| `cylinder_260527_holdout20` | 260527 holdout | 20 |
| `phantom_aligned_drive_val20` | aligned_drive val | 20 |

| 项 | 路径 |
|----|------|
| 根目录 | `/mnt/home/huang/cst_ai/outputs/eval_benchmark_v1/` |
| pairs（hardlink/子集） | `…/pairs/<column>/{LQ,GT}/` |
| 建集（历史；现并入单一脚本） | 原 `build_eval_benchmark_v1.py` → 现请用仓库历史或按需扩展 `build_eval_benchmark.py` |
| Drive 包 | `eval_benchmark_v1_3models_20260726`（约 677 MiB） |

**为何 v1 cylinder 几乎单电流：** `split_mode=tail` + 字典序 → val 几乎全是 `9-110kV_80uA_*`；`pick_spread` 只在该池内抽。

### 4.2 `eval_benchmark_v2`（泄漏优先 + 四模型）

**用户要求（摘录，07-27 / 07-29）：**

> 换测试集则重跑推理；格式与上次一致；上传新 Drive 文件夹、不覆盖旧包。  
> 「继续补完并上传，注意保持脚本单一性…并把规则写进 rule」。

| 相对 v1 的差异 | 说明 |
|----------------|------|
| cylinder 260520 | cyl90/mix **train 排除**后的安全 val 池；不强求多 µA |
| cylinder 260527 | 全 holdout LQ 上 **µA 层化** |
| phantom | val 减去 phantom/mix train |
| 模型行 | 旧三行 + **mix** |

| 项 | 路径 / 命令 |
|----|-------------|
| 根目录 | `/mnt/home/huang/cst_ai/outputs/eval_benchmark_v2/` |
| 建集 | `uv run python scripts/eval/build_eval_benchmark.py` |
| 推理（单模型全列或指定列） | 见下 |
| 填表 | `uv run python scripts/eval/fill_eval_benchmark_table.py --root …/eval_benchmark_v2 --preset v2` |
| 暂存交付 | `/mnt/home/huang/cst_ai/outputs/eval_benchmark_v2_4models_20260729/` |
| Drive | `gpu_server_huang:eval_benchmark_v2_4models_20260729`（约 797 MiB，350 objects） |

```bash
BENCH=/mnt/home/huang/cst_ai/outputs/eval_benchmark_v2
PRE=experiments/pretrained_models/SwinIR/004_grayDN_DFWB_s128w8_SwinIR-M_noise25.pth

# 示例：mix 全 5 列
bash scripts/eval/run_eval_benchmark_model.sh "$BENCH" \
  /mnt/home/huang/cst_ai/experiments/cylinder_phantom_mix/train_SwinIR_grayDN_cylinder_phantom_mix_noise25_P128W8_bs4_i30k/models/net_g_latest.pth \
  cylinder_phantom_mix cuda:0 "$PRE"

# 只补一列
bash scripts/eval/run_eval_benchmark_model.sh "$BENCH" <ckpt> <tag> cuda:0 "$PRE" phantom_aligned_drive_val20

uv run python scripts/eval/fill_eval_benchmark_table.py --root "$BENCH" --preset v2
```

**四模型权重对照**

| tag | ckpt |
|-----|------|
| `old_cylinder_archive` | `…/archive/cylinderblock_2673001900/net_g_latest.pth` |
| `phantom_aligned_drive` | `…/experiments/aligned_drive/…/net_g_latest.pth` |
| `new_cylinder_gt90ua` | `…/experiments/cylinder_260520_gt90ua/…_bs4_i30k/models/net_g_latest.pth` |
| `cylinder_phantom_mix` | `…/experiments/cylinder_phantom_mix/…_bs4_i30k/models/net_g_latest.pth` |

---

## 5. 传统 CV 基线（`eval_benchmark_cv`）

**用户要求（原文，07-29）：**

> 请你复用上面的提取出来的推理验证集，应该你有单独作为一个硬连接数据集保存吧，我现在想请大致判断一下（我能提供的情报就是图像是3d重建x光影像的xoy-2d切片），尝试用几种传统的cv方法进行处理，生成相应脚本处理同时简单记录时间，同样输出结果对比图到cst_ai的outputs文件夹下的一个对应cv的文件夹。

| 项 | 值 |
|----|-----|
| 输入 | **复用** `eval_benchmark_v2` 的 `pairs/` + `manifests/`（不另拷 TIFF） |
| 输出 | `/mnt/home/huang/cst_ai/outputs/eval_benchmark_cv/` |
| 脚本 | `scripts/eval/run_eval_benchmark_cv.py` |
| 方法 | `gaussian`, `median`, `bilateral`, `nlm`, `tv`, `wavelet` |
| 面板 | `LQ \| <method> \| GT` → `panels/<method>/<column>/*_compare.png`（共 450） |
| 计时 | 每图 `seconds` 写入 metrics；汇总 `timing_summary.csv` |
| 依赖 | 新增 `PyWavelets`（wavelet） |

**默认方法参数（粗基线，未 per-column 搜索）**

| 方法 | 参数要点 |
|------|----------|
| gaussian | `cv2.GaussianBlur` ksize=5, σ=1.0 |
| median | `medianBlur` ksize=5（uint8） |
| bilateral | d=9, sigmaColor=50, sigmaSpace=50 |
| nlm | `fastNlMeansDenoising` h=10, template=7, search=21 |
| tv | `denoise_tv_chambolle` weight=0.08 |
| wavelet | BayesShrink, soft, `rescale_sigma=True` |

```bash
# 全量（5 列 × 6 方法）
uv run python scripts/eval/run_eval_benchmark_cv.py \
  --bench-root /mnt/home/huang/cst_ai/outputs/eval_benchmark_v2 \
  --output-root /mnt/home/huang/cst_ai/outputs/eval_benchmark_cv

# 试跑
uv run python scripts/eval/run_eval_benchmark_cv.py \
  --columns cylinder_260520_val20 --max-per-column 1 \
  --output-root /tmp/eval_benchmark_cv_smoke
```

**粗结论（默认参数）：** bilateral / TV 在多数列 ΔSNR 最好；phantom 上约 +7 dB；wavelet 几乎无效；算法本身毫秒～约 1.5 s/图（总墙钟含 SNR/PSNR/SSIM）。

---

## 6. Drive 交付清单（勿互相覆盖）

| 文件夹名 | 内容 |
|----------|------|
| `cylinder_train10_model_compare` | 早期 10 张对比（保留） |
| `eval_benchmark_v1_3models_20260726` | v1 三模型包 |
| `eval_benchmark_v2_4models_20260729` | v2 四模型包（含 mix `net_g_latest.pth`） |

上传示例：

```bash
rclone copy /mnt/home/huang/cst_ai/outputs/eval_benchmark_v2_4models_20260729 \
  gpu_server_huang:eval_benchmark_v2_4models_20260729 \
  --transfers 8 --checkers 16 --retries 10
```

---

## 7. 用户原始提示词汇编（本系列关键句）

以下按时间摘自对话，便于还原「当时要什么」；完整上下文见 Cursor transcript。

### 07-22 — PreFT 并列图 + 加长 YAML

> 1.TRAIN_OPT是啥；  
> 2.验证集虽然得到了一个gt和输出和lq的对比图但是我想和微调前的对比一下效果，请你随便从验证集中挑几张帮我做一下推理分析，并输出在一起的并列图，注意之前有类似的推理验证脚本如果不复用记得删了；  
> 3.此外我已经决定好了下一次实验的改进方向了，请你拷贝一份yaml并帮我修改name和相关参数：  
> `val_freq`/`save_checkpoint_freq` 1000或2000；`batch_size_per_gpu: 8`；`total_iter: 30000`；`milestones: [15000, 25000]`。

### 07-24 — 固定评测 + Cylinder90 + 客户向指标说明

> 现在这个文件夹的组织方式和你验证的流程很不错可以固定一下…  
> 1.报告文件里所有指标请带上详细切生动的解释，给拍x光照的客户解释用；  
> 2.…三个来源 battery / cylinder / phantom…b和c一定要从val集里选择…  
> 3.请复制一个yaml…换用90ua的图作为gt…300kv…不再加入…改ci再训练一次…  
> 4.…填入表格…  
> 5.…硬链接和output…不要覆盖上一次内容…留下记录。

补充：

> 1.cylinder训练数据请只使用260520的slice 130–750…验证…260520的val…和260527…都选20；  
> 2.phantom…展开成为5列…  
> 3.…旧 cylinder + phantom + 尚未执行的 cylinder 三行；  
> 4.battery不够…有多少用多少，不要合并；  
> 5.phantom val 全是 5-4 也没关系。

### 07-27 — 第四行 Mix + 多电流评测疑问

> 那现在我要追加第四行训练数据和结果，然后基本上和最新 Cylinder90 训练集一样，只不过把phantom的数据也加进来一起学习然后推理验证即可，然后像之前一样生成结果文件并传到谷歌盘，注意…为什么之前…都选的同一个电流的数据，能不能多选几个电流…不能把训练集选进来。

确认方案：

> 如果不额外占用空间的话，请你新建一个混合集吧，注意命好名字便于区分；  
> 2.评测的时候建议B，之前训练过的模型就不要再训练了，但是注意重新挑选过程中如果实在没有多电流的非训练集也不要强求了，力求测试集合不要数据泄漏；模型不用重跑，但是如果你换了测试集合就重跑一下测试的图片即可，然后输出的格式仍和上次一致即可。

### 07-29 — 补完上传 + 脚本单一性

> 继续补完并上传，注意保持始终脚本的单一性，我不想同一个功能的脚本在仓库里复数存在，同时请把这个规则写进rule里面去。

### 07-29 — 传统 CV

> 现在请你复用上面的提取出来的推理验证集，应该你有单独作为一个硬连接数据集保存吧，我现在想请大致判断一下（我能提供的情报就是图像是3d重建x光影像的xoy-2d切片），尝试用几种传统的cv方法进行处理，生成相应脚本处理同时简单记录时间，同样输出结果对比图到cst_ai的outputs文件夹下的一个对应cv的文件夹。

### 07-29 — 本实验记录文档

> 请你把上面一系列的实验对应的参数改变，输入输出和使用脚本指令等等全都记录到一个md里，便于以后查看，现在实验次数越来越多了，然后还可以附上我当时对于该实验的要求和提示词便于理解。

### 07-29/07-30 — EPI + 残差 panel（不做 CNR）

> 实现 EPI / 残差图；panel 加大字体并把指标写上；不重跑推理；CNR 因无缺陷标注暂不做；上传到原 Drive 文件夹。

```bash
uv run python scripts/eval/enrich_eval_benchmark_panels.py \
  --bench-root /mnt/home/huang/cst_ai/outputs/eval_benchmark_v2
# → Drive: eval_benchmark_v2_4models_20260729（覆盖更新）
```

要点：训练 loss 为 **MSE**；panel 布局 `LQ|PreFT|FT|GT|ResPreFT|ResFT`；metrics 含 `*_epi`。

---

## 8. 相关仓库内指针

| 文件 | 作用 |
|------|------|
| `.cursor/project-context/decisions.md` | 稳定路径/决策短记 |
| `.cursor/rules/singular-scripts.mdc` | 脚本不复数 |
| `.cursor/lessons-learned/2026-07-25-swinir-bs8-oom-32gib.md` | bs=8 OOM |
| `options/train/SwinIR/train_SwinIR_grayDN_cylinder_gt90ua_bs8_30k.yml` | Cylinder90 |
| `options/train/SwinIR/train_SwinIR_grayDN_cylinder_phantom_mix_bs4_30k.yml` | Mix |

---

## 9. ALcast 150kV（2026-08，当前默认 `TRAIN_OPT`）

| 项 | 值 |
|----|-----|
| 原始数据 | `/mnt/data/cst_ai/datasets/alcast_150kv_raw/`（Drive 文件夹 `1U1yvwC1AQoa2AzRGtPDf5rk8mJZ-rP8W`） |
| Flat（hardlink） | `/mnt/data/cst_ai/datasets/alcast_150kv_gt66ma/` |
| Meta | `/mnt/home/huang/cst_ai/datasets/alcast_150kv_gt66ma/` |
| 角色 | GT=`1_ALcast_150kV66mA_data`；LQ=4/7/17/33mA（`scripts/data/folder_roles_alcast_150kv_gt66ma.yaml`） |
| 切片 | 用户 QC 范围 **70–820**（含端点），范围外空白片排除 |
| 划分 | 每电流分层留出 **8 test + 8 val**（各 32 张），其余 train |
| YAML | `options/train/SwinIR/train_SwinIR_grayDN_alcast_gt66ma_bs4_30k.yml` |
| `name` | `train_SwinIR_grayDN_alcast_gt66ma_noise25_P128W8_bs4_i30k` |
| 超参 | 4 GPU × bs4，P128，lr 1e-4，30k iter，milestones [15000, 25000]，MSE，每 2k 存档 |
| `experiments_root` | `/home/huang/cst_ai/experiments/alcast_150kv_gt66ma` |
| 在线 val | **`val: ~`**（避免 DDP val 阻塞训练 CI）；`val.txt` 保留但训练不用 |
| 训练后评测 | ALcast `test.txt` + 既有评测集（aligned_drive / cylinder / mix） |
| CI | `train_paired_tiff_swinir_debug` → `train_paired_tiff_swinir` |

## 10. TSFDAN 双分支 + LoRA（2026-08）

### 10.1 上游 PreFT 基线（不训练）

用 [Wasserdawn/TSFDAN](https://github.com/Wasserdawn/TSFDAN) 原仓库代码 + 官方 `SwinTransformer_288800iter.ckpt`，在 `eval_benchmark_v2` 五列上直接推理（面板 `LQ | Upstream | GT`，各列 `normalization.json` uint16→[0,1]，tile=512，约 1 分钟/张 1280²）。

- 独立目录：`/mnt/home/huang/cst_ai/experiments/tsfdan_upstream_infer/`（含 clone、`run_bench.sh`、`outputs/`）
- 复现：`bash /mnt/home/huang/cst_ai/experiments/tsfdan_upstream_infer/run_bench.sh [bench_root] [device] [tile]`

| 列 | n | resid_mean | PSNR | LQ PSNR | ΔPSNR |
|----|---|------------|------|---------|-------|
| battery_260520 | 9 | 0.005199 | 25.182 | 24.881 | +0.300 |
| battery_260527 | 6 | 0.003728 | 31.565 | 30.854 | +0.711 |
| cylinder_260520_val20 | 20 | 0.004574 | 34.932 | 34.176 | +0.756 |
| cylinder_260527_holdout20 | 20 | 0.007200 | 31.829 | 30.511 | +1.318 |
| phantom_aligned_drive_val20 | 20 | 0.010155 | 29.300 | 27.776 | +1.524 |

结论：官方预训练权重在本数据上改动很小（残差均值约 0.005），与 BasicSR 移植版 PreFT「几乎不变」一致，需要在本域微调。

### 10.2 LoRA 微调（mix 数据集）

| 项 | 值 |
|----|-----|
| Arch / Model | `TSFDANSwinIR` / `TSFDANSwinIRModel`（非 SwinIR-M 权重） |
| 预训练 | `experiments/pretrained_models/TSFDAN/SwinTransformer_288800iter.ckpt`（`scripts/download/download_tsfdan_pretrain.py`） |
| 网络 | embed_dim 60，depths [4,4,4]，heads [6,6,6]，window 8 |
| LoRA | r=16，alpha=16，target `qkv/proj/fc1/fc2/conv_after_body/conv_last` |
| YAML | `options/train/SwinIR/train_TSFDAN_lora_cylinder_phantom_mix_bs4_10k.yml` |
| 超参 | 4 GPU × bs4，P128，lr 1e-4，10k iter，milestones [5000, 8000]，MSE，**`val: ~`** |
| `experiments_root` | `/home/huang/cst_ai/experiments/cylinder_phantom_mix_tsfdan_lora` |
| CI | `train_tsfdan_lora_mix_debug` → `train_tsfdan_lora_mix` → **`eval_tsfdan_lora_mix`**（唯一正式指标/面板来源） |
| 推理 | `scripts/infer/infer_tsfdan_tiff.py`、`scripts/eval/run_eval_benchmark_tsfdan.sh`；tag `tsfdan_lora_mix` |

`eval_tsfdan_lora_mix` 优先用 `TSFDAN_CKPT_NAME`，找不到时回退到最新的 `net_g_<iter>.pth`。
