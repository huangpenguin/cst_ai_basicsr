# 调研与方法笔记

本文整合了原 `cst-ai/` 目录下的个人调研笔记（数据特性、预处理、TensorBoard、训练方向、SwinIR 改进、3D 切片、ADRnet/ACDNet、MONAI、AutoML、产业 CT PoC 技术调查等），去掉对话式重复内容，只保留结论和可执行建议。操作性内容（路径、命令、CI）已并入 [workflow.md](workflow.md)。

## 1. 任务与数据特性

对象是工业 X 光 CT 重建后的 **2D XOY 横截面切片**（电池、cylinder 金属件、phantom、ALcast 铸件），uint16 灰度 TIFF。任务是用低管电流（低剂量）图预测同位置高剂量图。

### 1.1 低剂量下的伪影来源

| 现象 | 物理原因 | 图像表现 |
|------|----------|----------|
| 射线硬化（beam hardening） | 高密度金属（铜箔、铝箔、外壳）吸收低能光子，只剩高能部分穿透 | 黑色条带、放射状阴影、边缘异常亮 |
| 光子饥饿（photon starvation） | 到达探测器的光子数不足，泊松统计噪声大 | 雪花噪点，SNR 极低 |
| 散射 | 金属周围散射光子 | 白色雾状晕影，遮挡细微结构 |

与相机的高斯噪声不同，CT 重建噪声是**结构性噪声**（投影不完整、金属伪影、泊松计数、重投影误差），常表现为贯穿全图的放射线。通用去噪器容易把它当成边缘保留甚至强化。只做 2D 去噪还会忽略 Z 向连续性，层间切换时可能闪烁。

理想（高剂量）图像：条带与散射消失、极片呈清晰的高对比平行细条纹、背景干净。

### 1.2 什么算"好"

业务标准（优先）：

1. 伪影被去除，背景均匀，被遮挡的极片纹理露出来。
2. 极片边缘**零涂抹**：Overhang（正负极片对齐）检测对几个像素的模糊都敏感。
3. 无幻觉：不能凭空画出极片，也不能把断裂的极片"脑补"连上。

量化指标（用于报告和选模型）：

- **PSNR**：整体像素误差，越高越好，30 dB 以上算不错。
- **SSIM**：结构 / 纹理 / 对比度相似度，0–1，越接近 1 越好（≥0.95 较理想），最契合极片条纹结构。
- 本项目另报告参考 **SNR** 与 SNR 增益、EPI（边缘保持指数）和残差图。

指标高不等于成功：可能过度平滑或生成假结构，最终要看图。

## 2. 3D 切片数据怎么用

| 视角 | 结论 |
|------|------|
| 网络输入 | 不同 Z 切片结构不同，可以当作不同 2D 样本来切 patch、增加迭代 |
| 数据划分 | **严禁按切片随机划分**。相邻切片空间高度相关，会造成泄漏、指标虚高。按体数据（volume / 样本 / 扫描条件）划分 |
| 真实多样性 | 1 个体数据的 100 张切片 ≠ 100 个独立样本；共享设备伪影和噪声分布，容易过拟合 |

实操建议：

- 剔除首尾纯背景切片（本项目用 `folder_roles*.yaml` 的 `slice_ranges`）。
- 在结构丰富的切片上多采 patch，平滑区域少采。
- 2.5D：把 Z-1/Z/Z+1 三张切片拼成 3 通道输入，利用 Z 向连续性（仅在 2D 基线到瓶颈时再做）。

## 3. 过度平滑（输出变糊）与改进方向

**原因**：纯 MSE / L1 监督会让模型输出所有可能解的平均值，噪声被抹掉的同时边缘和纹理也被抹平。MSE 的 PSNR 通常最高，但视觉上偏软。

按成本从低到高：

| 层级 | 手段 | 说明 |
|------|------|------|
| 数据 / 配置 | 检查配对、归一化、预训练是否正确加载、训练是否足够长 | 80% 问题出在这里；只改 YAML |
| 损失函数 | `L1Loss` / `CharbonnierLoss` 替代 MSE | 通常保留更多边缘；本项目当前仍用 MSE 作基线 |
| | Gradient / Edge Loss（Sobel/Laplacian 后算 L1） | 专治边缘变糊 |
| | FFT / Frequency Loss | 放射状伪影在频域特征明显，有助于抑制条纹并保留高频纹理 |
| | Perceptual（VGG）/ GAN Loss | 强制学习纹理，但有幻觉风险，工业检测场景慎用 |
| 超参 | 更大 patch（`gt_size` 与 `img_size` 同步改为 192/256，须为 `window_size` 的整数倍，可能要降 batch） | 更多上下文区分噪声与纹理 |
| | 延长训练（15k → 30k）、降低 `val_freq` 以便观察 | 验证指标仍在上升时 |
| 网络结构 | 全局残差：网络只预测噪声 / 伪影图 | 降低学习难度，天然保留低频结构 |
| | 浅层特征跨层融合（1×1 conv / channel attention 接到重建层前） | 保留浅层的边缘信息 |
| | 边缘感知双分支（Sobel 或小 CNN 分支 + cross-attention 融合） | 本项目 TSFDAN 双分支即属此类思路 |
| 后处理 | `α·SwinIR(X) + (1-α)·X` 混合、USM 锐化 | 简单但"盲目"：会把噪声按比例带回；网络内残差优于外部混合 |

架构替换（Restormer、NAFNet 等）只在 SwinIR 基线稳定、盲测 SNR 到平台或结构明显变糊时再考虑。Restormer 的 MDTA / GDFN 擅长全局依赖，适合跨越全图的放射状伪影；若采用应在本域配对数据上从头训练，并配合旋转 / 仿射增强以覆盖不同方向的条纹。

## 4. 配对监督的现实问题

- **像素级错位**：物理配对很难完全对齐，带错位的 L1/L2 监督会教出发糊的模型。本项目的数据源已像素配准，准备阶段禁止任何几何变换。
- **仿真 domain gap**：人工加高斯 / 泊松噪声（或 Radon 变换模拟低剂量）容易生成配对，但与真实设备的射线硬化、散射等非均匀噪声存在差距。这也是 TSFDAN 等论文引入自监督微调的原因。

## 5. 开源方案调研

| 方向 | 项目 | 要点 |
|------|------|------|
| 降噪 / 超分框架 | [MONAI](https://github.com/Project-MONAI/tutorials) | 医疗影像标准框架，含 2D/3D 网络、Diffusion/GAN、Auto3DSeg 等自动化组件；[Autoencoder MedNIST 教程](https://github.com/Project-MONAI/tutorials/blob/main/modules/autoencoder_mednist.ipynb) 可直接改成 2D 灰度去噪（`spatial_dims=2, in_channels=1`，可换 `UNet`） |
| 降噪 / 超分框架 | OpenMMLab（MMEditing / MMagic） | 大量预训练恢复模型（Real-ESRGAN、SwinIR），配置化组合 |
| 自动化 | NAS for Image Restoration | 自动搜索 U-Net / Transformer 变体，暂列为 TODO |
| 金属伪影（MAR） | [ACDNet](https://github.com/hongwang01/ACDNet) | 自适应卷积字典网络，显式嵌入金属伪影物理先验，可解释性和跨数据集泛化好；适合 cylinder 等强金属伪影 |
| MAR 训练数据 | [metal_artifact_simulation](https://github.com/NAIST-ICB/metal_artifact_simulation) | MATLAB 生成含金属伪影的投影数据，内置水等效射线硬化校正，可批量造配对数据 |
| 有限角重建 | [DOLCE](https://github.com/wustl-cig/DOLCE) | 条件扩散模型 + 物理测量模型，SOTA；用 2D 先验合成一致的 3D 体 |
| 有限角重建 | [DL-limited-angle-CT-reconstruction](https://github.com/WangStephen/DL-limited-angle-CT-reconstruction) | FDK + U-Net，轻量，含正弦图生成工具 |
| 底层算子 | [RTK](https://github.com/RTKConsortium/RTK) | ITK 系锥束 CT 重建库（SART/SIRT 等），提供正投影 / 反投影 |

PoC 建议顺序：先用依赖少、参数量小的库（ACDNet、DL-limited-angle-CT）在少量自有切片上快速验证可行性。

### ACDNet 类 MAR 网络的 mask 处理

ACDNet 等 MAR 网络也是逐层处理 2D XOY 切片，与本项目数据形态一致。对大实心金属圆柱：

- **整个内部都应 mask 为 1**：实心金属在正弦图上留下很宽的无信号带，只 mask 边缘会漏掉中间的盲区，重建仍满屏条纹；整体 mask 才能正确计算并插值 / 预测整条带。
- 内部没有需要保留的结构，整体 mask 等于让网络专注修复圆外被污染的区域。
- 阈值分割自动生成的本来就是实心圆盘，直接使用即可。
- 若是**空心金属管**（内部是水或其他材料），mask 应为圆环，只标管壁。

## 6. 历史：Cylinderblock 几何预处理（已废弃）

早期 cylinderblock 实验的 GT 为 300kV/100µA（1530×1530），LQ 为 1280×1280 且方向不同，因此预处理做了：

- 旋转：260520 批 193°、260527 批 91°（`scipy.ndimage.rotate`，`reshape=False`，双线性，`mode="nearest"` 以避免四角黑三角）。
- 放大：1280 → 1530（×1.195，双线性），带来轻微模糊。
- 按 `data_XXXX.tif` 与 GT 同切片配对，slice 130–750，GT 用 hardlink 复用。

该流程与相关脚本已删除；结果归档在 `/mnt/home/huang/cst_ai/archive/cylinderblock_2673001900/`。现行流程要求数据源本身已配准，准备阶段不做任何几何变换，并改用 90µA（260520）/ 60µA（260527）作 GT。旧模型在新评测集上吃亏的原因即在于此。

## 7. 为什么选择"clone BasicSR 原仓库"

- 路线 A（clone 全仓库 + editable install）适合要改 cfg、数据集、网络的微调；路线 B（`pip install basicsr` + 自建项目）适合结构差异大的新工程。本项目用路线 A。
- 先只换数据、YAML 和预训练权重，用官方 `basicsr/train.py -opt xxx.yml` 跑通；效果不佳时按第 3 节的层级排查，最后才考虑换框架。
- 默认 SR/去噪模型不需要编译 C++ 扩展（`BASICSR_EXT=True` 仅 EDVR/StyleGAN2 等需要）。
