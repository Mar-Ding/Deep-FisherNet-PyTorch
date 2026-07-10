# Deep FisherNet 复现考核任务总结报告

## 1. 任务目标

这次考核任务的主要目标是复现论文 **Deep FisherNet for Object Classification** 在 PASCAL VOC 2007 分类任务上的结果，并尽量接近论文中报告的 **91.2% mAP**。

一开始我对这个任务的理解比较直接：把论文里的 Deep FisherNet 结构用 PyTorch 搭起来，训练后看 mAP 能不能接近论文结果。但实际做下来发现，这个工作不是简单复现一个网络结构，而是一个比较完整的工程复现任务。Deep FisherNet 依赖多阶段训练、dense patch 采样、Fisher Vector 编码、GMM/PCA 初始化、多尺度测试和外部 SVM 分类器，其中很多细节都来自原始 Caffe 实现。如果这些细节没有对齐，即使主干网络和大体结构相同，结果也会有明显差距。

因此，后面的工作重点逐渐从“单纯跑模型”转为“建立可靠 baseline，然后通过一组组实验判断差距到底来自哪里”。

最终当前最好的结果是：

> **VGG16 corrected-patches + full SVM C=10：80.25% mAP**

这个结果距离论文的 91.2% 还有约 **10.95 mAP** 差距。虽然没有完全达到论文指标，但我围绕这个差距做了多轮调参、结构迁移、初始化、Stage1 迁移和评估链路排查，基本判断出继续简单加 epoch、随意换 backbone 或只补 Stage1 迁移的收益都很低，差距更可能来自原始 Caffe 训练链路和实现细节。

## 2. 我完成的主要工作

这次任务中，我主要完成了以下几类工作：

1. 搭建并调通 Deep FisherNet 的 PyTorch 训练与评估流程。
2. 实现 VOC2007 多标签分类数据读取、dense patch 采样和多尺度输入。
3. 实现 VGG16/ResNet101 backbone 接入和 Fisher 编码层。
4. 实现 Fisher Vector 特征抽取和外部 SVM 评估。
5. 设计多组实验，对不同可能瓶颈进行验证。
6. 根据实验结果修改代码，补充 feature cache、SVM sweep、GMM 初始化和 Stage1 权重继承等功能。
7. 记录每组实验结果，并根据 gate 规则决定是否继续跑 full SVM 或更长训练。
8. 对官方 Caffe ResNet101 backbone 做逐层数值对齐，定位并修复 PyTorch 与 Caffe 的结构差异。

我觉得这次任务中比较重要的一点是，没有把所有结果简单归结为“没调好参数”，而是尽量把每一步实验设计成有明确目的的验证。例如：换 ResNet101 是为了验证 backbone 是否限制性能；冻结 BN 是为了验证小 batch 下 BN 统计是否造成退化；重新做 GMM 初始化是为了验证 Fisher 初始化是否是瓶颈；扫 SVM 的 C 和 solver 是为了验证最终分类器是否欠拟合。

## 3. Baseline 建立过程

最开始尝试时，端到端训练的 FC 分类头 mAP 很低，val mAP 长期在 0.1 左右。如果只看这个指标，很容易认为模型完全没有学到东西。但论文实际采用的是 Fisher Vector 特征加外部 SVM 的评估方式，所以我后面把重点放到了 SVM 评估上。

经过修正 patch 采样和 resize 流程后，我建立了 B1 baseline：

- backbone：VGG16
- 输入方式：多尺度 dense patches
- resize：longest-side resize
- Fisher：legacy Fisher 实现
- 后处理：SVM

B1 的结果是：

| 设置 | mAP |
|------|-----|
| 500-sample SVM C=10 | 75.28% |
| full SVM C=1 | 79.76% |
| full SVM C=10 | **80.25%** |

B1 full SVM C=10 的 per-class AP 如下：

| 类别 | AP | 类别 | AP | 类别 | AP | 类别 | AP |
|------|----|------|----|------|----|------|----|
| aeroplane | 95.86 | bicycle | 87.02 | bird | 92.83 | boat | 90.75 |
| bottle | 56.68 | bus | 81.65 | car | 87.15 | cat | 85.66 |
| chair | 63.14 | cow | 71.73 | diningtable | 71.30 | dog | 86.70 |
| horse | 87.84 | motorbike | 82.11 | person | 92.39 | pottedplant | 50.88 |
| sheep | 82.88 | sofa | 67.20 | train | 94.52 | tvmonitor | 76.67 |

这个结果说明当前实现不是完全失败的。它在 aeroplane、bird、boat、person、train 等类别上已经有比较高的 AP，但在 bottle、chair、pottedplant、sofa 这类小目标或背景复杂的类别上偏弱。这也提示后续差距可能和 patch 覆盖、尺度对齐、局部区域特征质量有关。

## 4. 后续实验设计和结果

为了判断 80.25% 到 91.2% 的差距来自哪里，我后面做了几条路线的实验。每条路线都设置了 500-sample SVM gate，先用较低成本判断是否值得继续。

### 4.1 B2：尝试更接近官方的 Fisher 参数化

我首先怀疑 PyTorch 里的 Fisher 层实现和原始 Caffe 实现不一致，所以做了 B2。B2 尝试引入更接近官方的 Fisher 设置，包括：

- Caffe 风格参数化。
- log-det 项。
- learned priors。
- sum pooling。

结果是：

| 方案 | 500-sample SVM |
|------|----------------|
| B1 corrected baseline | 75.28% |
| B2 official-fisher 1 epoch | 68.80% |
| B2 official-fisher 4 epoch | 67.36% |

这个结果低于 B1，而且 4 epoch 还比 1 epoch 更差。因此我判断，仅仅把 Fisher 公式改得更像官方，并不能直接补上差距。可能原因是参数化、初始化、学习率和当前 patch 特征分布没有同时对齐。于是 B2 停止，没有继续 full SVM。

### 4.2 C1/C2/C3/C4：尝试迁移并对齐 ResNet101

接下来我尝试换更强的 backbone。直觉上 ResNet101 比 VGG16 更强，也许可以提升特征质量。但 Deep FisherNet 很依赖 dense patch 局部描述子，所以这个迁移并不一定稳定。为了避免一次性试太多变量，我把 BN、初始化和 Caffe/PyTorch 结构差异逐步拆开验证：

| 方案 | 设置 | 500-sample SVM | 观察 |
|------|------|----------------|------|
| C1 | ResNet101 spatial Fisher, random init | 55.13% | 明显失败 |
| C2 | ResNet101 spatial + frozen BN | 62.53% | 有提升，但仍低 |
| C3 | frozen BN + PCA/GMM init | 61.02% | 初始化没有带来提升 |
| C4 | Caffe-v1 ResNet101 + frozen BN + PCA/GMM init | 67.48% | backbone 对齐有效，但仍未过 gate |

C1 很低，说明 ResNet101 spatial 不能直接替代 VGG dense patch 路线。C2 冻结 BN 后提升了 7.4 mAP，说明小 batch 训练下 BN 确实是一个问题。但即使冻结 BN，结果还是远低于 B1。C3 加入 PCA/GMM 初始化后也没有继续提升，说明问题不只是初始化。

后来为了确认是不是 PyTorch ResNet101 和官方 Caffe ResNet101 本身不一致，我又做了 Caffe/PyTorch 逐层对齐。用同一张图导出 `input -> conv1_relu -> pool1 -> res2c -> res3b3 -> res4b22 -> res5c` 后，定位到两处差异：Caffe 的 `pool1` 是 `pad=0` 并依赖 ceil 输出，而 torchvision 默认是 `pad=1`；另外 Caffe ResNet 的下采样 stride 在 bottleneck 的 `branch2a/conv1`，torchvision ResNet v1.5 默认在 `conv2`。修正后，backbone 7 个可比层全部通过，`res5c/layer4` 的 relative L2 约为 `3.75e-7`。

基于这个修正，我重新跑了 C4。C4 从 C2 的 62.53% 提升到 67.48%，说明 Caffe-v1 backbone 对齐确实有作用。但它仍然低于 70% 停止线，也明显低于 B1 的 75.28% 500-sample SVM。因此这组实验的结论更清楚：ResNet101 低分不只是一个 backbone 实现 bug，但即使修掉这个 bug，spatial ResNet101 路线也没有追上 VGG dense-patch 主线。

### 4.3 D1：修正 GMM 初始化并延长训练

另一个可能原因是 B1 的 GMM 初始化和训练数据不对齐。检查代码后发现，`init_gmm.py` 原本没有正确传入 `resize_mode=longest`，这意味着 GMM 初始化看到的 patch 分布和训练时的 patch 分布可能不同。

我修复了这个问题，然后重新用 longest-resize 的 patch 描述子估计 GMM，并训练 8 epoch。D1 的结果是：

- GMM init：200K descriptors，resize_mode=longest
- 训练：8 epoch
- train_loss：0.2760
- val_mAP：约 0.1037，从 epoch4 后基本平台
- 500-sample SVM：73.13%

这个结果反而比 B1 的 75.28% 低了 2.15 mAP。因此我判断，当前主线不是简单训练不够，也不是修复 GMM resize 后就能提升。继续加 epoch 的收益不高。

这个实验虽然结果不好，但它排除了一个很常见的想法：模型没到论文结果，不一定是因为训练轮数不够。至少在当前设置下，训练更久没有改善 Fisher 表征的 SVM 结果。

### 4.4 E1：检查 SVM 后处理是不是瓶颈

Deep FisherNet 最终分类依赖外部 SVM，所以我又检查了是否是 SVM 的 C 值或 solver 没调好。为此我修改了 `scripts/evaluate.py`，加入：

- Fisher Vector feature cache。
- `--svm-solver sgd|liblinear`。
- `--svm-max-iter`。
- `--svm-tol`。
- C 值扫描。

这样可以先抽一次 FV 特征，然后反复扫 SVM 参数，不需要每次都重新跑 GPU forward。

E1 中 SGD solver 的 500-sample sweep 结果是：

| C | 500-sample mAP |
|---|----------------|
| 0.1 | 72.40% |
| 1 | 74.25% |
| 3 | 75.14% |
| 10 | **75.53%** |
| 30 | 75.02% |
| 100 | 75.03% |

最好结果是 75.53%，只比 B1 的 75.28% 高 0.25 mAP。liblinear 的第一项在加载缓存后被终止，没有得到完整结果，但 SGD sweep 已经足够说明 SVM 调参不是主要瓶颈。

所以我没有继续跑 E2 full SVM sweep。我的判断是，即使 full SVM 有一点提升，也大概率只能从 80.25% 到 80.3%-80.6%，无法接近 86.2%，更不用说 91.2%。

### 4.5 F1：验证完整 Stage1→Stage2 迁移

最后我又验证了一个之前认为最有希望的变量：Stage1 到 Stage2 的权重继承。论文流程中先做 whole-image CNN finetune，再把权重迁移到 FisherNet。之前代码里主要加载 VGG 的卷积层，但 `fc6/fc7` 没有完整映射到 Stage2 的 patch MLP，所以我先修复了这个映射，再让 agent 重新跑 F1。

F1 中 Stage1 权重加载是成功的：

- 成功加载 30 个兼容 tensor。
- `fc6/fc7` 已经按预期映射到 `patch_mlp`。
- 500-sample SVM C=10：75.34% mAP。

对比结果如下：

| 方案 | 500-sample mAP | 说明 |
|------|----------------|------|
| B1 | 75.28% | no Stage1 baseline |
| E1 | 75.53% | SVM sweep 最佳 |
| F1 | 75.34% | Stage1 transfer |

F1 和 B1/E1 基本没有差别，说明在当前实现下，补上 Stage1 transfer 并没有带来预期提升。因此我也停止了 F1，没有继续跑 full SVM。

## 5. 实验汇总

| 方案 | 目的 | 500-sample SVM | Full SVM | 结论 |
|------|------|----------------|----------|------|
| B1 | VGG16 corrected-patches baseline | 75.28% | **80.25%** | 当前最佳 |
| B2 | 官方风格 Fisher 参数化 | 67.36% | - | 低于 B1，停止 |
| C1 | ResNet101 spatial | 55.13% | - | 迁移失败 |
| C2 | ResNet101 spatial + frozen BN | 62.53% | - | 有提升但不够 |
| C3 | ResNet101 + frozen BN + PCA/GMM | 61.02% | - | 初始化不是主要问题 |
| C4 | Caffe-v1 ResNet101 + frozen BN + PCA/GMM | 67.48% | - | backbone 对齐有帮助，但仍不够 |
| D1 | longest-resize GMM init + 8 epoch | 73.13% | - | 延长训练无收益 |
| E1 | SVM solver/C sweep | **75.53%** | - | SVM 不是主要瓶颈 |
| F1 | Stage1 transfer | 75.34% | - | Stage1 迁移也不是主要瓶颈 |

从这些实验可以看出，我尝试的方向不是随机调参，而是围绕几个可能的差距来源逐步验证：

- 如果是 Fisher 公式不一致，B2 应该提升，但没有。
- 如果是 backbone 不够强，ResNet101 应该提升；实际只有 Caffe-v1 对齐后回升到 67.48%，仍低于 B1。
- 如果是 BN 问题，C2 应该明显接近 B1，但仍差很多。
- 如果是初始化问题，C3/D1 应该提升，但没有。
- 如果是训练轮数问题，D1 8 epoch 应该提升，但没有。
- 如果是 SVM 调参问题，E1 应该有明显提升，但只有 0.25 mAP。
- 如果是 Stage1 权重继承问题，F1 应该提升，但结果和 B1 基本相同。

所以当前结论是：差距不是某一个简单超参数造成的，而是更可能来自完整官方实现链路没有完全对齐。

## 6. 对实验路线合理性的复盘

我也重新检查了整个尝试过程，重点看有没有明显违背论文本身意图、或者属于低质量试错的实验。整体判断是：主要实验都不是无目的乱试，但有几条需要在报告里明确写成“诊断性扩展”，不要写成严格复现论文 VGG16 91.2% 指标的主线。

比较贴合论文意图的部分包括：

- B1 的 VGG16 corrected dense patches + Fisher Vector + SVM，是当前最接近论文 VGG 路线的主线。
- B2 尝试官方风格 Fisher 参数化，是为了对齐 Caffe 自定义 Fisher 层。
- D1 修复 GMM 初始化的 resize 对齐，是必要的工程排错。
- E1 检查 SVM solver 和 C 值，是因为论文最终评估依赖外部 SVM。
- F1 修复 Stage1 到 Stage2 的 `fc6/fc7 -> patch_mlp` 权重继承，是为了补齐论文多阶段训练逻辑。

需要克制表述的是 C1/C2/C3/C4 的 ResNet101 spatial 路线。它不是论文 VGG16 91.2% 结果的直接复现路线，而是基于官方仓库中可见的 `Res-101` prototxt 做的迁移和诊断实验。它的作用是验证“更强 backbone + 官方 spatial Fisher 架构是否能补足差距”，以及排查小 batch BN、PCA/GMM 初始化和 Caffe/PyTorch backbone 差异。因此它不是无依据尝试，但应明确属于辅助诊断路线。C4 修正 backbone 后仍未过 gate，也说明继续在这条线上烧卡的收益不高。

另外，B2 只能说明“当前 PyTorch 里的公式级近似没有带来提升”，不能说明官方 Fisher 设计本身无效。因为我们没有拿到官方 `.mat` 和 `.caffemodel`，所以 B2 还不是数值级官方 Fisher 复现。

所以整个过程里没有明显违背论文意图的主线错误；真正要注意的是报告措辞：把论文主线、工程排错、诊断性扩展区分清楚，不把负结果写成对论文方法本身的否定。

## 7. 我在工程上做的修改

除了跑实验，我也根据实验中暴露的问题改了一些代码。

### 7.1 评估脚本支持特征缓存

为了减少重复抽特征的成本，我给 `scripts/evaluate.py` 增加了 `--feature-cache-dir`。这样同一个 checkpoint 的 FV 特征可以缓存下来，后续只需要反复训练 SVM。

这对 E1 很重要。如果没有 feature cache，每扫一个 C 都要重新跑一遍模型，成本会高很多。

### 7.2 支持不同 SVM solver 和参数

我在评估脚本里增加了：

- `--svm-solver`
- `--svm-max-iter`
- `--svm-tol`
- `--no-standardize`

这样可以更系统地检查 SVM 后处理，而不是只能用固定默认值。

### 7.3 修复 GMM 初始化的 resize 对齐

我给 `scripts/init_gmm.py` 增加了 `--resize-mode`，确保 GMM 初始化和训练时的数据 resize 方式一致。虽然 D1 没有提升，但这个修改排除了一个真实存在的工程偏差。

### 7.4 修复 Stage1 到 Stage2 的 VGG 权重继承

论文里的训练流程包含 Stage1 whole-image finetune，然后迁移到 Stage2 FisherNet。检查代码后发现，原来的 `--stage1-weights` 主要加载 VGG 的 `features.*`，但没有把 VGG 的 fc6/fc7 映射到 Stage2 的 patch MLP。

我补充了下面的映射：

- `classifier.1.weight` -> `patch_mlp.0.weight`
- `classifier.1.bias` -> `patch_mlp.0.bias`
- `classifier.4.weight` -> `patch_mlp.3.weight`
- `classifier.4.bias` -> `patch_mlp.3.bias`

这个修改很关键，因为它让后续真正验证 Stage1→Stage2 迁移成为可能。F1 结果显示，即使这个映射成功加载，当前链路的 500-sample SVM 也只有 75.34%，因此 Stage1 迁移本身不是主要差距来源。

### 7.5 生成 ResNet101 spatial 的 PCA/GMM 初始化

为了测试 C3，我实现了 `scripts/init_res101_spatial_params.py`，可以生成 ResNet101 spatial Fisher 所需的 PCA/GMM 初始化参数，并导出 `.pt` 和与 Caffe 字段兼容的 `.mat`。虽然 C3 没有提升，但这个工具为后续和官方参数对齐提供了基础。

### 7.6 对齐 Caffe 和 PyTorch 的 ResNet101 backbone

后面我又补了一个更细的排查：用同一张图片分别导出 Caffe 和 PyTorch 的中间层结果，逐层比较误差。这个过程发现 torchvision ResNet101 和官方 Caffe ResNet101 有两个结构细节不同：

- `pool1` 的 padding/ceil 输出方式不同。
- bottleneck 下采样 stride 所在卷积不同。

我把 PyTorch 里的 ResNet101 改成 Caffe-v1 风格后，单图 backbone 对齐通过。这个修改随后用于 C4 实验。C4 的 500-sample SVM 是 67.48%，比 C2 有提升，但仍然没有接近 B1。因此这一步的价值主要是把“backbone 数值不一致”这个问题从黑箱里拆出来，而不是直接带来最终高分。

## 8. 我对差距的理解

现在最好的结果是 80.25%，距离 91.2% 仍然有明显差距。结合这些实验，我认为差距主要不在“再多调几个超参数”，而在下面几个更深的地方。

### 8.1 官方 Caffe 训练链路没有完全复刻

Deep FisherNet 不是单阶段训练。论文中的 Stage1/Stage2 权重继承、Fisher 初始化、多尺度 patch 流程和 SVM 评估是连在一起的。目前我已经修了 Stage1 权重映射，并通过 F1 验证了迁移可以成功加载，但它没有带来 mAP 提升。因此差距更可能来自 Stage1 之外的官方 Caffe 参数、预处理和 patch/Fisher 细节。

### 8.2 Caffe 和 PyTorch 的图像预处理可能有差异

Caffe 模型常见的 BGR/RGB 顺序、mean subtraction、resize 插值、坐标取整方式，都可能影响局部 patch 特征。Deep FisherNet 又特别依赖 dense patch，所以这些细节会被放大。当前只修了 longest resize，还不能保证和官方完全一致。

### 8.3 官方 Fisher/PCA/GMM 参数缺失

Fisher 编码对 GMM means、sigmas、priors 和 PCA 投影比较敏感。自己估计的参数不一定等价于官方训练资产。B2/C3/D1 的结果说明，表面上补 Fisher 公式或初始化，不一定能复现官方行为。

### 8.4 patch 采样细节可能仍未完全一致

B1 已经证明 dense patches 是正确方向，但弱类别集中在小目标和背景复杂类别上，这说明 patch 覆盖、尺度选择、边界处理或 ROI 对齐仍可能有偏差。

### 8.5 SVM 后处理不是主要矛盾

E1 扫过 C 后只提升 0.25 mAP，所以 SVM 参数不是当前最主要问题。后面如果还要提升，重点应该回到特征提取和训练链路，而不是继续扫 SVM。

## 9. 后续如果继续做，我会优先做什么

F1 跑完后，训练侧主要方向基本都已经验证过了。如果这个课题后续还继续推进，我不会再优先开新的大规模训练，而会把重点放到官方实现对齐上：

1. 找官方 Caffe `.caffemodel`、Fisher `.mat`、PCA/GMM 参数和 mean file。
2. 对齐 Caffe 的 BGR/RGB、mean subtraction、resize 插值和坐标取整。
3. 对齐 patch 采样边界、ROI pooling、Fisher 参数化和 SVM 评估脚本。
4. 针对 bottle、chair、pottedplant、sofa 等弱类别做 patch 可视化，检查小目标覆盖是否和论文实现有差异。

也就是说，后续的关键不是继续“试一个新训练配置”，而是尽量复原官方 Caffe 工程细节。只有这些资产或细节对齐后，才有可能解释 80.25% 到 91.2% 之间的主要差距。

## 10. 总结

这次考核任务最终没有把 Deep FisherNet 完全复现到论文的 91.2% mAP，但我认为这个过程还是比较完整地完成了一个复现课题应该做的工作。

当前最好的结果是 **80.25% mAP**。这个结果说明 PyTorch 实现已经获得了有效的 Fisher 表征，但和论文仍有明显差距。为了分析这个差距，我没有只做简单调参，而是围绕多个可能瓶颈设计了 B/C/D/E/F 系列实验：

- B1 建立了可靠 baseline。
- B2 验证 Fisher 参数化不是单独瓶颈。
- C1/C2/C3/C4 验证 ResNet101 迁移、BN/初始化和 Caffe-v1 backbone 对齐问题。
- D1 验证 GMM resize 修复和延长训练没有带来收益。
- E1 验证 SVM solver/C 不是主要瓶颈。
- F1 验证 Stage1→Stage2 权重继承不是主要瓶颈。

通过这些实验，我逐步排除了几条低收益路线，也明确了后续最值得继续做的是官方 Caffe 细节对齐，而不是继续盲目训练。

如果从课题组考核角度总结，我觉得这次工作体现了三个方面：

1. **复现能力**：能够把论文中的复杂结构和评估流程落到可运行代码中。
2. **实验设计能力**：不是随机调参，而是针对假设逐步设计实验。
3. **工程判断能力**：能够根据结果及时停止低收益路线，把问题收敛到更可能的差异来源。

所以这次任务虽然没有完全追到论文数字，但已经形成了一个清晰的复现 baseline、完整的实验记录和后续可执行方向。我认为这对于课题组继续推进该论文复现，或者基于 Fisher 编码做后续改进，都是有参考价值的。
