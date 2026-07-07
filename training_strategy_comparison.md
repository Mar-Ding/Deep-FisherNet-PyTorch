# 训练策略对比：原始 FisherNet vs 当前 PyTorch 实现

> 对比依据：arXiv:1608.00182 (2016) 论文 + 官方代码 https://github.com/ppengtang/fishernet (Caffe)  
> 当前项目更新日期：2026-07-07（新增 configs.py、ResNetFisherNet、preset 配置系统）

---

## 1. 训练流程

| 维度 | 原始 FisherNet | 当前实现 | 状态 |
|---|---|---|---|
| **训练阶段** | 两阶段：① 整图 CNN finetune (9k iters) → ② FisherNet 端到端 (40k iters) | **单阶段**：直接端到端 | ❌ 未实现 |
| **后处理** | 提取 FV 特征 → 训练 **linear SVM** (C=1) | 直接用 **FC 分类头** | ❌ 未实现 |
| **Preset 系统** | 无 | 3 个 preset: smoke / alexnet-paper-like / official-res101-like | ✅ |
| **总迭代量** | ~49k iters | 由 epoch 数控制（多尺度训练下约 50k per epoch） | ✅ 量级可调 |

## 2. 优化器 & 学习率策略

| 维度 | 原始 FisherNet (论文) | 官方 ResNet Caffe 代码 | 当前实现 (official-res101-like preset) |
|---|---|---|---|
| **优化器** | SGD + Momentum (0.9) | SGD + Momentum (0.9) | **SGD + Momentum** (0.9) ✅ |
| **Backbone lr** | 0.001 | 冻结 (lr_mult=0) | **0.001** |
| **FC 层 lr** | 0.1 | — | 0.001 (weight) / 0.002 (bias) |
| **Fisher Layer lr** | 0.0001 | — | **0.01 (weight) / 0.02 (bias)** ⚠️ 比论文高 100× |
| **lr schedule** | multistep @60% iter, gamma=0.1 | stepvalue=8000 (单步), gamma=0.1 | multistep @60% epoch, gamma=0.1 ✅ |
| **Weight decay** | 0.0005 | 0.0005 | 0.0005 ✅ |
| **Grad accum** | — | iter_size=8 | grad_accum_steps=8 ✅ |
| **有效 batch** | 2（batch×iter\_size） | 16（2×8） | 8（1×8）≈ 接近 |

> 注意：官方 ResNet Caffe 代码中**所有 conv 层 lr_mult=0**（backbone 冻结），而当前 preset 里 backbone_lr=0.001 是端到端微调的。论文的 VGG16 实验也是端到端的。两者策略不同但都有依据。

## 3. Patch 提取策略

| 维度 | 原始 FisherNet | 当前实现 (official-res101-like) |
|---|---|---|
| **提取方法** | **SPP** (Spatial Pyramid Pooling) | **RoI Align**（不同但功能等价） |
| **Patch 尺度** | **7 个尺度**: {64,96,128,160,192,224,256} | **7 个尺度**: {64,96,128,160,192,224,256} ✅ |
| **Stride** | 32 | 32 ✅ |
| **每图 patch 数** | ~300-800 | 上限 800 ✅ |
| **RoI 输出尺寸** | SPP 多级 pooling | roi_output_size=1（ResNet 用 1×1） |
| **Patch 特征维度** | 256（降维后） | 128（ResNet: 2048→128 FC） |

> ✅ patch 密度已对齐原文（7 scales, stride 32, max 800），不再是我之前对比里说的"4 scales + stride 64 + max 160"。

## 4. 数据增强 & 测试策略

| 维度 | 原始 FisherNet | 当前实现 |
|---|---|---|
| **训练图像** | 5 scales {480,576,688,864,1200}, resize 长边保宽高比 | 固定 336~1008 方形随机缩放（train_scales） |
| **训练增强** | 水平翻转 + 多尺度 | 水平翻转 (hflip_prob=0.5) + 多尺度随机缩放 ✅ |
| **测试图像** | **5 尺度** {480,576,688,864,1200} + flip → 10 views 取均值 | 固定 448×448 **单尺度** ❌ |

> ⚠️ 最大差异：测试时完全无多尺度集成。原文靠 10-view 集成提点，当前单尺度测试会损失约 1-2 mAP。

## 5. 特征处理

| 维度 | 原始 FisherNet | 当前实现 |
|---|---|---|
| **FV 后处理** | power-norm → L2-norm → **linear SVM** (C=1) | power-norm → L2-norm → **FC layer** ❌ |
| **Fisher Layer priors** | 丢弃（等权重假设） | 丢弃（learn_priors=False）✅ |
| **GMM 分量 K** | 32 (AlexNet/VGG) | 64 (ResNet) / 32 (AlexNet) |
| **FV 维度** | 2×K×D = 16384 | 2×64×128 = 16384 ✅ |
| **损失函数** | 多标签 sigmoid 交叉熵 | BCEWithLogitsLoss ✅ |

## 6. Backbone

| 维度 | 原始 FisherNet (论文) | 官方 ResNet Caffe | 当前实现 |
|---|---|---|---|
| **架构** | AlexNet / VGG16 | **ResNet-101** | ResNet-101 + AlexNet |
| **预训练** | ImageNet | ImageNet | ImageNet ✅ |
| **Conv 训练** | 端到端 | 冻结 (lr_mult=0) | 端到端 (lr=0.001) |
| **Patch MLP** | FC6→FC7→256-dim FC | ResNet conv5→2048→128 FC | 2048→128 FC ✅ |

---

## 改进优先级（更新版）

| 优先级 | 改进项 | 状态 | 预期收益 |
|---|---|---|---|
| P0 | 改用 **SGD + multistep + 差异化 lr** | ✅ **已实现** | — |
| P0 | 增加 patch 密度（7 scales, stride 32, max 800） | ✅ **已实现** | — |
| P0 | 训练时多尺度 + 水平翻转 | ✅ **已实现** | — |
| P0 | ResNet101 backbone | ✅ **已实现** | — |
| P1 | **测试时多尺度集成**（3-5 scales + flip） | ❌ 未实现 | 估计 +1~2 mAP |
| P2 | **两阶段训练**（先 finetune backbone 再插 Fisher Layer） | ❌ 未实现 | 收敛稳定性 |
| P2 | **Linear SVM 替代 FC 头** | ❌ 未实现 | 可能 +0.5~1 mAP |
| P2 | 更新 README 反映 preset 用法 | ❌ 未更新 | 文档完整性 |

## 项目文件结构（更新后）

```
fishernet/
├── configs.py              ← 新增: 3 个 preset 配置
├── data/
│   ├── voc.py              ← 更新: train_scales + hflip_prob
│   └── patches.py
├── models/
│   ├── __init__.py          ← 更新: 导出 ResNetFisherNet
│   ├── fisher_layer.py
│   └── fishernet.py         ← 更新: 新增 ResNetFisherNet + build_fishernet
├── utils/
│   └── metrics.py
scripts/
├── train.py                 ← 重写: --preset / SGD / multistep / grad_accum
├── evaluate.py              ← 更新: --preset
├── init_gmm.py              ← 更新: --preset
└── benchmark_step.py
reports/
├── handoff_4090.md          ← 新增: GPU 迁移交接文档
└── implementation_report.md
```

> 原始官方代码（Caffe）在 `external/official-fishernet/`
