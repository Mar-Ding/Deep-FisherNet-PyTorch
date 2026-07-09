# Deep FisherNet — 当前实验状态

## 项目信息

- **论文：** Deep FisherNet for Object Classification (NIPS 2016)
- **本地路径：** `F:/paper_ws/Deep FisherNet for Object Classification/`
- **当前最佳：** B1 VGG16 corrected-patches + full SVM C=10 = **80.25% mAP**
- **更新日期：** 2026-07-09

---

## 已完成实验矩阵

| 方案 | 描述 | 500-sample SVM | Full SVM | 结论 |
|------|------|----------------|----------|------|
| **B1** | VGG16 corrected dense patches, legacy Fisher, C=10 | 75.28% | **80.25%** | ✅ 当前最佳 |
| B2 | VGG16 official-like Fisher math, 4 epoch | 67.36% | - | ❌ gate failed |
| C1 | ResNet101 spatial Fisher, random init | 55.13% | - | ❌ gate failed |
| C2 | ResNet101 spatial Fisher + frozen BN | 62.53% | - | ❌ gate failed |
| C3 | ResNet101 spatial Fisher + frozen BN + self-fit PCA/GMM init | 61.02% | - | ❌ gate failed |
| D1 | B1 + longest-resize GMM init + 8 epoch | 73.13% | - | ❌ below 78% stop line |

## B1 最佳结果

B1 full SVM C=10: **80.25% mAP**

Per-class AP:

| 类别 | AP | 类别 | AP | 类别 | AP | 类别 | AP |
|------|----|------|----|------|----|------|----|
| aeroplane | 95.86 | bicycle | 87.02 | bird | 92.83 | boat | 90.75 |
| bottle | 56.68 | bus | 81.65 | car | 87.15 | cat | 85.66 |
| chair | 63.14 | cow | 71.73 | diningtable | 71.30 | dog | 86.70 |
| horse | 87.84 | motorbike | 82.11 | person | 92.39 | pottedplant | 50.88 |
| sheep | 82.88 | sofa | 67.20 | train | 94.52 | tvmonitor | 76.67 |

## 工程判断

ResNet101 spatial 路线已经测过三个关键变量：

- C1: 官方空间 Fisher 架构本身，55.13%
- C2: 冻结 BN 后升到 62.53%，说明 BN 是问题之一
- C3: 补 PCA/GMM 初始化后降到 61.02%，说明主要瓶颈不只是初始化

因此继续烧 ResNet101 spatial 的性价比很低。它离 B1 的 75.28% 500-sample gate 还差 12-14 mAP，离 full SVM 80.25% 更远。

D1 又验证了 B1 主线的训练平台问题：

- 修复 GMM init 的 `resize_mode=longest` 后，8 epoch 500-sample SVM 为 73.13%
- 训练从 epoch4 起平台，val_mAP 约 0.10
- 比 B1 4 epoch 的 75.28% 低 2.15 mAP

这说明继续延长当前 end-to-end 训练不是通向 86.2% 的有效路径。

## 下一步建议

1. **停止 ResNet101 spatial 路线。**
   除非拿到真正官方 Caffe `.caffemodel` + `.mat`，否则不再建议上卡跑 C 系列。

2. **把 B1 作为最终可交付主线。**
   80.25% 已经是当前代码/数据/评估链路下最稳的结果。

3. **如果还要追分，必须换成诊断/对齐路线。**
   下一步优先检查评估实现、特征缓存、SVM solver、patch head/stage1 权重继承、Caffe 预处理差异。不要继续简单加 epoch。

4. **若目标必须接近论文 91.2%，需要改目标定义。**
   需要找回官方模型/参数，或完整复刻 Caffe 数据预处理、PCA/GMM、网络权重和评估脚本。当前 PyTorch 复现路线靠挤牙膏不太可能补 11 mAP。

## 已补工程能力

- `scripts/evaluate.py` 支持 `--feature-cache-dir`，可缓存 B1 FV 特征后反复扫 SVM。
- `scripts/evaluate.py` 支持 `--svm-solver sgd|liblinear`，用于排查 SGD SVM 是否欠拟合。
- `scripts/init_gmm.py` 已修复 GMM init 未传 `resize_mode=longest` 的问题。
- `scripts/train.py` 的 `--stage1-weights` 已补 VGG `fc6/fc7` 到 `patch_mlp` 的映射，后续 stage1->stage2 才是可信变量。
- `AGENT_E_SVM_DIAGNOSTIC.md` 给出下一轮 agent 只跑评估/缓存/SVM sweep 的指令。
