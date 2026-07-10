# Deep FisherNet 复现考核任务报告

## 1. 任务和最终结果

考核任务是复现论文 **Deep FisherNet for Object Classification** ，在 PASCAL VOC 2007 上做训练和测试。论文的结果是 **91.2% mAP**。

一开始我以为主要工作是把网络结构用 PyTorch 搭出来，然后训练到接近论文结果。实际做下来发现，Deep FisherNet 不只是一个普通 CNN 分类网络，它的结果依赖一整套流程：图像预处理、dense patch 采样、CNN patch 特征、Fisher Vector 编码、PCA/GMM 参数、多阶段训练，最后还要用外部 SVM 做分类评估。任何一段没有和官方实现对齐，最后 mAP 都可能差很多。

当前我跑到的最好结果是：

> B1 VGG16 corrected-patches + full SVM C=10：**80.25% mAP**

和论文的 91.2% 相比，还差 **10.95 mAP**。这个差距不小，所以后面我没有只继续加 epoch 或者随便扫参数，而是把可能出问题的地方拆开检查。最后跑了 B/C/D/E/F 几组实验，分别看 Fisher 公式、ResNet101 迁移、BatchNorm、PCA/GMM 初始化、训练时长、SVM 参数和 Stage1 到 Stage2 权重继承。

目前的判断是：80.25% 说明 PyTorch 版本已经学到有效的 Fisher 表征，但还没有和论文的官方 Caffe 实验链路对齐到同一个水平。差距更可能来自官方权重、`.mat` 参数、预处理和 patch 采样细节，而不是某一个 C 值或者 epoch 没调好。

## 2. 官方代码和公开实现情况

我查到的官方仓库是：

- [ppengtang/fishernet](https://github.com/ppengtang/fishernet)

这个仓库里有官方 Caffe 实现，主要包括：

- `external/official-fishernet/caffe-fishernet-conv`
- `external/official-fishernet/models/Res-101`
- `external/official-fishernet/tools`

但是它不是一个可以直接复现论文 91.2% 的完整实验包。比较关键的问题是：

- `README.md` 只有很简单的说明，没有完整训练命令。
- `tools/load_parameters.py` 需要外部 `.mat` 参数，字段包括 `weights`、`bias`、`priors_new`、`pca_w`、`pca_b`，但仓库没有给出论文使用的那份参数。
- 仓库里主要能看到 `Res-101` 相关模型，没有完整的 VGG16 复现实验资产。
- 没有找到论文实验对应的 `.caffemodel`、Fisher/PCA/GMM `.mat`、mean file、训练日志和 SVM 评估脚本。

我也看了网上有没有现成 PyTorch 复现。结果基本是：

- 公开 Caffe 架构还是官方 `ppengtang/fishernet` 为主。
- 没找到可靠的第三方 PyTorch 复现。
- 官方 fork 里也没看到新增权重、`.mat` 或 PyTorch 实现。

所以这个任务不是“下载官方代码跑一下”，而是在官方只给了部分 Caffe 工程的情况下，把能复现的部分迁到 PyTorch，再通过实验看差距到底卡在哪里。

## 3. 我实现和整理的工程内容

当前工程里已经有一条完整的 PyTorch 训练和评估链路，包括：

- VOC2007 多标签分类数据读取；
- 多尺度输入和 dense patch 采样；
- VGG16 / ResNet101 backbone；
- patch MLP；
- Fisher Vector 编码层；
- 多标签 BCE 训练；
- Fisher Vector 特征抽取；
- 外部 SVM 评估；
- per-class AP 和 mAP 统计；
- 4090 远程训练 runbook 和日志回传流程。

前期比较容易误判的一点是：训练时端到端 FC 分类头的 val mAP 长期只有 0.1 左右。如果只看这个数，会觉得模型完全没学到东西。但论文最终不是直接用 FC 头评估，而是用 Fisher Vector 特征接 SVM。后面改用 SVM 结果作为主要判断后，B1 能到 80.25%，说明模型并不是完全失败，问题更多在复现细节和官方对齐上。

## 4. 实验策略

因为 full SVM 和长时间训练都比较耗时，我没有让每个方案都跑完整评估，而是用了一个比较简单的 gate：

- 先跑 500-sample SVM，看方向有没有希望；
- 只有明显接近或超过 B1 的方案，才继续 full SVM；
- 明显低于停止线的方案直接停掉。

这样做主要是为了省训练时间。比如 ResNet101 那几组在 500-sample 上已经明显低于 B1，如果继续跑 full SVM，基本只是确认一次已经很明显的负结果。

## 5. B1：VGG16 corrected-patches baseline

B1 是当前最好的结果，也是后面所有实验的参照。

配置：

- backbone：VGG16
- 输入：多尺度 dense patches
- resize：longest-side resize
- patch sizes：64, 96, 128, 160, 192, 224, 256
- stride：32
- max patches：800
- Fisher：legacy Fisher 实现
- 评估：SVM

结果：

| 设置 | mAP |
|------|-----|
| 500-sample SVM C=10 | 75.28% |
| full SVM C=1 | 79.76% |
| full SVM C=10 | **80.25%** |

B1 full SVM C=10 的 per-class AP：

| 类别 | AP | 类别 | AP | 类别 | AP | 类别 | AP |
|------|----|------|----|------|----|------|----|
| aeroplane | 95.86 | bicycle | 87.02 | bird | 92.83 | boat | 90.75 |
| bottle | 56.68 | bus | 81.65 | car | 87.15 | cat | 85.66 |
| chair | 63.14 | cow | 71.73 | diningtable | 71.30 | dog | 86.70 |
| horse | 87.84 | motorbike | 82.11 | person | 92.39 | pottedplant | 50.88 |
| sheep | 82.88 | sofa | 67.20 | train | 94.52 | tvmonitor | 76.67 |

这个结果说明目前的实现有可用的表征能力。强的类别主要是 aeroplane、bird、boat、person、train；弱的类别是 bottle、chair、pottedplant、sofa 这类小目标或者背景干扰比较大的类别。差距可能和 patch 覆盖、尺度、局部特征分布有关。

## 6. B2：官方风格 Fisher 参数化

B2 想检查的是 Fisher 层本身。官方 Caffe 里有自定义 Fisher layer，如果 PyTorch 里的 Fisher 公式和官方有差异，可能会影响很多。

我尝试加入更接近官方的设置：

- Caffe 风格 Fisher 参数化；
- log-det 项；
- learned priors；
- sum pooling。

结果：

| 方案 | 500-sample SVM |
|------|----------------|
| B1 corrected baseline | 75.28% |
| B2 official-fisher 1 epoch | 68.80% |
| B2 official-fisher 4 epoch | 67.36% |

B2 没有涨，4 epoch 还比 1 epoch 低。这个结果说明，只把公式写得更像官方是不够的。Fisher 的参数、初始化、学习率和输入特征分布要一起对齐，否则可能还不如 B1 里的 legacy 实现。

所以 B2 没有继续跑 full SVM。

## 7. C1/C2/C3：ResNet101 spatial Fisher

ResNet101 这条线不是论文 VGG16 91.2% 的直接复现路线。我跑它主要是因为官方仓库里能看到 `Res-101` 相关 prototxt，而且想检查更强 backbone 和 spatial Fisher 架构能不能补差距。

三组结果：

| 方案 | 设置 | 500-sample SVM | val_mAP ep4 |
|------|------|----------------|-------------|
| C1 | ResNet101 spatial, random init | 55.13% | 0.0940 |
| C2 | ResNet101 spatial + frozen BN | 62.53% | 0.1805 |
| C3 | frozen BN + self-fit PCA/GMM init | 61.02% | 0.1018 |

C1 很低。C2 冻结 BN 后涨了 7.4 mAP，说明小 batch 下 BN 统计确实会拖后腿。但 62.53% 还是远低于 B1。C3 加了 PCA/GMM 初始化也没有继续涨。

这条线的结论比较明确：在当前实现里，ResNet101 spatial 不是追分的好方向。更强 backbone 没有自动变成更高 mAP，可能还因为没有保留 VGG dense patch 那种局部采样方式，反而损失了 Deep FisherNet 需要的局部描述子。

## 8. D1：GMM 初始化修正和延长训练

D1 检查两个很具体的问题：

1. GMM 初始化看到的 patch 分布是否和训练一致；
2. B1 是不是训练太短。

检查代码时发现，`scripts/init_gmm.py` 原来没有正确把 `resize_mode=longest` 传进数据集。也就是说，训练用 longest resize，但 GMM 初始化可能不是同一个 resize 方式。这个问题需要修，因为 Fisher 对 patch 分布比较敏感。

修完后，我用 longest-resize 的 patch descriptors 重新估计 GMM，并训练 8 epoch。

D1 结果：

- GMM init：200K descriptors，resize_mode=longest
- 训练：8 epoch
- train_loss：0.2760
- val_mAP：约 0.1037，从 epoch4 后基本平台
- 500-sample SVM C=10：73.13%

D1 比 B1 4 epoch 的 75.28% 还低 2.15 mAP。这个结果有点反直觉，但也说明一个问题：继续加 epoch 或者修一个单独的 GMM resize 细节，并不会自然把分数推上去。

## 9. E1：SVM solver 和 C 值

论文最终评估用 SVM，所以我检查了一下 SVM 后处理是不是限制了分数。

我改了 `scripts/evaluate.py`，增加：

- `--feature-cache-dir`
- `--svm-solver sgd|liblinear`
- `--svm-max-iter`
- `--svm-tol`
- `--no-standardize`

这样可以把 Fisher Vector 特征先缓存下来，再重复扫 SVM 参数，不用每次都重新跑 GPU forward。

E1 的 500-sample SGD SVM sweep：

| C | 500-sample mAP |
|---|----------------|
| 0.1 | 72.40% |
| 1 | 74.25% |
| 3 | 75.14% |
| 10 | **75.53%** |
| 30 | 75.02% |
| 100 | 75.03% |

最好的 C=10 是 75.53%，只比 B1 的 75.28% 高 0.25 mAP。这个提升太小，不可能解释 80.25% 到 91.2% 的差距。liblinear 第一项加载缓存后被中止，没有完整产出，但 SGD sweep 已经能说明 SVM 参数不是主要瓶颈。

## 10. F1：Stage1 到 Stage2 权重继承

论文里有 Stage1 whole-image CNN finetune，再迁移到 Stage2 FisherNet 的过程。检查代码时发现，原来的 `--stage1-weights` 主要加载 VGG 的 `features.*`，但 VGG 的 `fc6/fc7` 没有正确映射到 Stage2 的 `patch_mlp`。

我补了下面的映射：

- `classifier.1.weight` -> `patch_mlp.0.weight`
- `classifier.1.bias` -> `patch_mlp.0.bias`
- `classifier.4.weight` -> `patch_mlp.3.weight`
- `classifier.4.bias` -> `patch_mlp.3.bias`

F1 结果：

- Stage1 checkpoint 成功加载 30 个兼容 tensor；
- `fc6/fc7 -> patch_mlp` 映射成功；
- 500-sample SVM C=10：75.34%。

对比：

| 方案 | 500-sample mAP | 说明 |
|------|----------------|------|
| B1 | 75.28% | no Stage1 baseline |
| E1 | 75.53% | SVM sweep 最佳 |
| F1 | 75.34% | Stage1 transfer |

F1 和 B1/E1 基本一样。也就是说，在当前代码和训练条件下，补上 Stage1 transfer 没有带来明显收益。

## 11. 实验汇总

| 方案 | 目的 | 500-sample SVM | Full SVM | 结论 |
|------|------|----------------|----------|------|
| B1 | VGG16 corrected-patches baseline | 75.28% | **80.25%** | 当前最佳 |
| B2 | 官方风格 Fisher 参数化 | 67.36% | - | 低于 B1，停止 |
| C1 | ResNet101 spatial | 55.13% | - | 迁移失败 |
| C2 | ResNet101 spatial + frozen BN | 62.53% | - | 有提升，但仍远低于 B1 |
| C3 | ResNet101 + frozen BN + PCA/GMM | 61.02% | - | 初始化不是主要问题 |
| D1 | longest-resize GMM init + 8 epoch | 73.13% | - | 延长训练无收益 |
| E1 | SVM solver/C sweep | **75.53%** | - | SVM 不是主要瓶颈 |
| F1 | Stage1 transfer | 75.34% | - | Stage1 迁移没有明显收益 |

这些实验基本是在按差距来源逐个排查：

- 如果 Fisher 公式是主因，B2 应该涨，但没有；
- 如果 backbone 是主因，ResNet101 应该涨，但没有；
- 如果 BN 是主因，C2 应该接近 B1，但仍然差很多；
- 如果 PCA/GMM 初始化是主因，C3 或 D1 应该涨，但没有；
- 如果训练不够，D1 8 epoch 应该涨，但没有；
- 如果 SVM 是主因，E1 sweep 应该明显涨，但只涨 0.25 mAP；
- 如果 Stage1 transfer 是主因，F1 应该涨，但基本没变化。

所以现在我不太倾向于继续做小范围调参。继续扫 C、加 epoch、换随机种子可能还能挤出一点点，但很难补上 10 mAP 以上的差距。

## 12. 对实验路线的复盘

我回头检查了一下这些尝试，主要想确认有没有明显偏离论文意图，或者只是为了跑实验而跑实验。

比较贴近论文主线的部分有：

- B1：VGG16 dense patches + Fisher Vector + SVM，是当前最接近论文 VGG 路线的方案；
- B2：尝试官方风格 Fisher 参数化，是为了对齐 Caffe 自定义 Fisher 层；
- D1：修 GMM 初始化和训练 resize 不一致的问题，这是实际代码里发现的偏差；
- E1：检查 SVM，因为论文最终结果本来就依赖外部 SVM；
- F1：补 Stage1 到 Stage2 的 `fc6/fc7 -> patch_mlp` 权重继承，对应论文里的多阶段训练逻辑。

需要单独说明的是 C1/C2/C3。它们不是论文 VGG16 91.2% 结果的直接复现，而是基于官方仓库里能看到的 `Res-101` prototxt 做的扩展排查。它们的作用是看“更强 backbone + spatial Fisher”有没有可能补差距，以及小 batch BN、PCA/GMM 初始化是不是问题。结果不好之后及时停掉是合理的，但报告里不能把这条线说成论文 VGG 主线。

B2 也要注意表述。因为拿不到官方 `.mat` 参数，所以 B2 只能说明“公式级近似加自生成参数没有带来提升”，不能说明官方 Fisher 设计本身无效。

总的来说，这些实验不是乱试；但在写报告时要把“论文主线复现”和“工程诊断扩展”分开写，避免把负结果说得过头。

## 13. 工程上改过的地方

这次不只是跑实验，也改了几处会影响判断的代码。

### 13.1 feature cache

`scripts/evaluate.py` 增加了 `--feature-cache-dir`。这样同一个 checkpoint 的 train/test Fisher Vector 可以缓存，后面扫 SVM 参数不用重复抽特征。

### 13.2 SVM 参数可控

评估脚本支持：

- `--svm-solver`
- `--svm-max-iter`
- `--svm-tol`
- `--no-standardize`

这让 SVM 可以单独诊断，而不是只能用一个固定默认配置。

### 13.3 GMM 初始化 resize 对齐

`scripts/init_gmm.py` 增加了 `--resize-mode`，修复 GMM 初始化和训练 resize 可能不一致的问题。D1 没有涨分，但这个修复本身是必要的。

### 13.4 Stage1 权重映射

`scripts/train.py` 里补了 VGG `fc6/fc7` 到 Stage2 `patch_mlp` 的映射。这样 F1 的 Stage1 transfer 才算真的测到了。

### 13.5 ResNet101 spatial 参数初始化

为了跑 C3，我实现了 `scripts/init_res101_spatial_params.py`，可以生成 ResNet101 spatial Fisher 需要的 PCA/GMM 参数，并导出 PyTorch `.pt` 和 Caffe 字段兼容的 `.mat`。这次没有涨分，但以后如果拿到官方参数，可以继续用这个方向做对齐。

## 14. 为什么有官方代码还是对不齐

这次最容易困惑的一点是：官方代码明明能看到，为什么结果还是对不齐？

我的理解是，能看到代码结构，不等于能复现当年的实验状态。官方仓库能告诉我们网络大概怎么连、Fisher layer 怎么写、参数加载脚本要哪些字段；但它没有给出完整实验所需的状态，比如：

- 论文使用的 `.caffemodel`；
- 论文使用的 Fisher/PCA/GMM `.mat`；
- VGG16 路线的完整 prototxt 和训练权重；
- mean file、resize 插值、BGR/RGB、patch 坐标等预处理细节；
- SVM 评估脚本和参数；
- 能直接跑到 91.2% 的 runbook。

Deep FisherNet 对这些细节很敏感。Fisher 里的 priors、sigma、PCA 投影稍微不一样，最后 FV 分布就会变；patch 坐标、resize、mean subtraction 不一样，也会改变局部描述子。现在我能做到的是结构级复现，也就是流程和模块大体对应；但论文数字更像是需要数值级对齐，也就是同一张图经过 Caffe 和 PyTorch 后，中间特征都要基本一致。缺少官方权重和 `.mat` 参数时，这一步很难保证。

## 15. 我对差距来源的判断

结合实验结果，我觉得差距主要可能来自下面几类。

### 15.1 Caffe 预处理差异

Caffe 里常见的 BGR/RGB 顺序、mean subtraction、resize 插值、坐标取整方式都可能影响结果。Deep FisherNet 又依赖 dense patch，这些小差异会被放大。现在只对齐了 longest resize，还不能说完全复刻了 Caffe 行为。

### 15.2 官方 Fisher/PCA/GMM 参数缺失

Fisher 编码对初始化很敏感。我们可以自己估 PCA/GMM，也可以按官方字段导出 `.mat`，但这不等于论文使用的那份参数。B2/C3/D1 都说明，自己生成的“看起来合理”的 Fisher 参数不能自然恢复论文性能。

### 15.3 patch 采样仍可能有差异

B1 已经说明 dense patches 是正确方向，但 bottle、chair、pottedplant、sofa 这些类比较弱。这些类别很容易受小目标覆盖、背景 patch 比例和尺度选择影响，所以 patch 采样细节仍然值得怀疑。

### 15.4 Stage1/Stage2 训练还没有完全等价

F1 说明补上 `fc6/fc7` 映射后没有涨分，但这不代表官方训练链路已经完全复刻。Stage1 的数据预处理、训练周期、优化器、初始化资产和迁移方式仍可能不同。

### 15.5 SVM 不是主要问题

E1 里 SVM sweep 只带来 0.25 mAP 左右的变化，所以 SVM 后处理最多是小因素，不太可能解释 10.95 mAP 的差距。

## 16. 后续如果继续做

如果后面还要继续追论文结果，更值得做的是官方对齐：

1. 找官方 `.caffemodel`、Fisher/PCA/GMM `.mat`、mean file 和训练日志；
2. 尝试直接编译并跑通官方 Caffe 工程；
3. 用同一张图片做 Caffe 和 PyTorch 的逐层对比，包括输入、backbone feature、PCA 输出、Fisher assignment、Fisher sum 和最终 FV；
4. 对 bottle、chair、pottedplant、sofa 等弱类别可视化 patch 覆盖，看看小目标是否被采样策略稀释；
5. 如果拿不到官方资产，就在报告里说明：官方代码能看到，但关键权重和参数缺失，所以只能做到结构级复现，不能保证论文数字级复现。

## 17. 总结

这次没有把结果追到论文的 91.2%，最后最好是 **80.25% mAP**。这个结果不算达到目标，但也不是一个完全失败的复现：至少说明当前 PyTorch 版本能得到有效的 Fisher 表征。

我后面的主要工作是围绕差距做排查，而不是只继续调参。B1 建了 baseline；B2 看 Fisher 参数化；C1/C2/C3 看 ResNet101、BN 和初始化；D1 修 GMM resize 并延长训练；E1 看 SVM；F1 补 Stage1 到 Stage2 的权重继承。这些实验排除了一些看起来可能有用、但实际收益不大的方向。
