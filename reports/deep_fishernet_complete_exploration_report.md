# Deep FisherNet 复现与实验报告

## 1. 任务目标

​	本项目的目标是复现论文 **Deep FisherNet for Object Classification**，在 PASCAL VOC2007 上实现论文报告的 **91.2% mAP**。目前工程的最佳结果是：

> **VGG16 corrected-patches + Fisher Vector + SVM：80.25% mAP**

​	在尝试复现中，我的尝试思路是：

​	本次复现先建立可用基线，再进行 Caffe/PyTorch 对齐；随后围绕公式、backbone、初始化、训练、SVM 和权重迁移逐项排查。初版 PyTorch 结果与论文有差距后，我又编译并运行了官方 Caffe 工程，对统一输入在两个版本中进行逐层比较，先排除实现错误，再判断问题主要来自哪一部分。最后通过健康度诊断检查 Fisher assignment 是否塌缩，并据此调整训练策略。

​	项目重点包括：

- Caffe 与 PyTorch backbone 对齐；
- dense patch 提取；
- PCA 降维；
- PCA/GMM/Fisher 参数初始化；
- Fisher Vector 编码；
- 端到端训练；
- 外部 SVM 评估；
- VOC2007 全量测试集验证。

当前工程形成了两条主要路线：

1. VGG16 corrected-patches 主路线；
2. ResNet101 spatial Fisher 诊断与改进路线。



下面按实验顺序说明具体过程。

## 2. 实验环境

- 数据集：PASCAL VOC2007
- 训练集：5011 张
- 测试集：4952 张
- GPU：NVIDIA GeForce RTX 4090 24GB
- Python：3.12.3
- PyTorch：2.5.1+cu124

## 3. 工程实现与 Caffe/PyTorch 对齐

当前 PyTorch 工程已经实现：

- VOC2007 多标签数据读取；
- 多尺度图像预处理；
- dense patch 采样；
- VGG16 和 ResNet101 backbone；
- patch descriptor 提取；
- PCA 投影；
- Fisher Vector 编码；
- 多标签 BCE 训练；
- FV+SVM 评估；
- per-class AP 和 mAP 统计；
- descriptor/FV 缓存机制；
- Fisher assignment 健康度与参数漂移监控。

### 3.1 G1：官方 Caffe 工程编译

完成内容：

- 成功编译官方 `caffe-fishernet-conv`；
- Caffe binary 和 pycaffe 均可运行；
- 官方 Fisher 自定义层可解析；
- `test_image.prototxt` 的 238 个 blob 可以导出。

同时确认官方仓库没有提供完整论文训练资产：

- 未找到完整 Deep FisherNet `.caffemodel`；
- 未找到论文对应的 Fisher/PCA/GMM `.mat`；
- 未找到完整 VGG16 训练日志和评估脚本。

因此后续采用结构级和数值级对齐相结合的方式进行复现。

### 3.2 G2/G3：Fisher 全链路数值对齐

通过 Caffe 和 PyTorch 对同一输入逐层比较，定位到：

- Caffe 的 Fisher prior 实际通过 `fisher_weight -> Softmax` 注入；
- PyTorch 默认使用 `softmax(log(priors))`；
- 两种 prior 语义会导致不同的 Fisher assignment；
- 二阶项需要使用 `0.7071` 缩放。

完成修正后：

- `input` 到 `res5c_pca` 对齐；
- PCA L2 输出对齐；
- Fisher assignment 对齐；
- Fisher first-order/second-order 对齐；
- `fisher_sum` 对齐；
- **12/12 个可比较层全部通过**。

这说明在初始化状态下，Caffe 与 PyTorch 的 Fisher 前向实现已经完成对齐。

## 4. 实验设计与消融结果总览

下表先汇总各组实验的目的、主要改动和结果。前期大量结果使用 500 张样本进行快速筛选；表中注明“全量”的结果使用 VOC2007 test 全部 4952 张图像。500 张结果用于比较方向，不能与全量结果直接等同。

| 实验 | 主要改动 | 评估方式 | 结果 | 对下一步的判断 |
|---|---|---|---:|---|
| B1 | VGG16 corrected-patches + Fisher Vector | 全量 FV+SVM | **80.25%** | 建立当前工程主基线 |
| B2 | Caffe 风格 Fisher 参数化、learned prior、log-det | 500 张 FV+SVM | 67.36% | 仅改公式不能带来提升 |
| C1 | ResNet101 spatial，随机初始化 | 500 张 FV+SVM | 55.13% | 更换 backbone 后结果下降 |
| C2 | ResNet101 + frozen BN | 500 张 FV+SVM | 62.53% | 小 batch 下 BN 统计是重要因素 |
| C3/C4 | PCA/GMM 初始化、Caffe-v1 结构对齐 | 500 张 FV+SVM | 61.02% / 67.48% | 结构对齐有效，但仍低于 VGG 主线 |
| D1 | 修复 GMM resize，训练至 8 epochs | 500 张 FV+SVM | 73.13% | 初始化和训练时长不是主要差距来源 |
| E1 | SVM solver、C 值和标准化扫描 | 500 张 FV+SVM | **75.53%** | 评估器只能带来小幅改善 |
| F1 | 补全 Stage1 到 Stage2 的 fc6/fc7 迁移 | 500 张 FV+SVM | 75.34% | 权重迁移不是主要瓶颈 |
| G9 | GAP、assignment 熵、参数漂移诊断 | GAP+SVM | 7.71% | C4 训练破坏了 backbone 和 Fisher 分配 |
| G10A | 固定 encoder，只训练 FC | FC / FV+SVM | 16.23% / 30.86% | FV 中有信息，但 FC 没有有效利用 |
| G11 | 扫描 sigma、temperature 和 prior | 500 张 FV+SVM | 43.07% / 56.71% | assignment 对结果影响很大，需区分 SVM 和端到端训练配置 |
| G12B | sigma=1,T=2，微调 backbone | 全量 FV+SVM | **65.81%** | 微调改善 FV，但 FC 仍明显落后 |
| G13 | 低学习率 warmup Fisher 参数 | FC | 29.11% | Fisher 已开始学习，尚未证明能达到论文结果 |

下面是按这个顺序展开。每组实验都分别说明设计原因、实际执行、结果，以及结果如何影响下一步选择。

## 5. B1：VGG16 主基线

B1 是当前全工程最高结果，也是后续实验的主要参照。

设计 B1 的目的，是先固定输入、backbone、patch 提取和评估方法，建立一个可以反复比较的主基线。后续每次只改变一个主要因素，才能判断性能变化来自哪里。

配置：

- VGG16 backbone；
- corrected dense patches；
- patch sizes：64、96、128、160、192、224、256；
- stride：32；
- max patches：800；
- Fisher Vector；
- 外部线性 SVM；
- SVM `C=10`。

结果：

| 评估方式 | mAP |
|---|---:|
| 500-sample SVM | 75.28% |
| Full SVM，C=1 | 79.76% |
| Full SVM，C=10 | **80.25%** |

类别表现方面，aeroplane、bird、boat、person、train 较强；bottle、chair、pottedplant、sofa 较弱。弱类别主要是小目标、背景复杂或目标区域容易被 patch 稀释的类别。

## 6. B2：官方风格 Fisher 参数化

为了验证 Fisher 公式是否是主要问题，加入了：

- Caffe 风格参数化；
- log-det；
- learned priors；
- sum pooling；
- 官方风格二阶项缩放。

结果：

| 实验 | 500-sample mAP |
|---|---:|
| B1 baseline | 75.28% |
| B2，1 epoch | 68.80% |
| B2，4 epochs | 67.36% |

结果低于 B1，因此不再沿这条配置继续训练。

从这个结果看，仅把公式改得更接近官方并不能自动提升性能。后续需要把 Fisher 参数、PCA/GMM 初始化、输入特征分布和训练策略放在一起检查。

由于没有拿到官方 `.mat` 参数，B2 只能说明当前公式级近似没有带来收益，不能否定论文中的官方 Fisher 设计。

## 7. C1～C4：ResNet101 spatial 路线

ResNet101 路线属于诊断性扩展，不是论文 VGG16 91.2% 结果的直接复现路线。实验目的包括：

- 验证更强 backbone 是否能提升结果；
- 检查小 batch 下 BatchNorm 的影响；
- 检查 Caffe/PyTorch ResNet 结构差异；
- 检查 PCA/GMM 初始化是否为主要瓶颈。

结果：

| 实验 | 配置 | 500-sample mAP |
|---|---|---:|
| C1 | ResNet101 spatial，random init | 55.13% |
| C2 | ResNet101 spatial，frozen BN | 62.53% |
| C3 | frozen BN + PCA/GMM init | 61.02% |
| C4 | Caffe-v1 ResNet101 + frozen BN + PCA/GMM | 67.48% |

主要发现：

- frozen BN 带来明显提升，说明小 batch BN 统计会破坏训练；
- 修正 Caffe-v1 stride、pooling 后继续提升；
- PCA/GMM 初始化没有带来额外收益；
- 即使完成 backbone 结构对齐，仍然低于 VGG16 主线。

## 8. D1：GMM 初始化和训练时长

检查发现，GMM 初始化时的 resize 配置可能与训练阶段不一致。因此修复了 GMM 初始化的 `resize_mode`，重新提取 patch descriptors、估计 PCA/GMM，并训练到 8 epochs。

结果：

- 500-sample SVM：73.13%；
- 低于 B1 的 75.28%；
- epoch4 后基本进入平台；
- FC val_mAP 约 10%。

从这个结果看，单独修复 GMM resize 或延长训练不能解决主要差距，下一步应把注意力转向评估器和训练过程。

## 9. E1：SVM solver 和 C 值扫描

评估脚本增加了 feature cache、SGD/liblinear solver、C 参数、最大迭代次数、tolerance 和 standardization 开关。

结果：

| C | 500-sample mAP |
|---:|---:|
| 0.1 | 72.40% |
| 1 | 74.25% |
| 3 | 75.14% |
| 10 | **75.53%** |
| 30 | 75.02% |
| 100 | 75.03% |

SVM 参数优化只带来约 0.25 mAP 的提升。因此，SVM solver 和 C 值不是 80.25% 与论文 91.2% 差距的主要来源，后续没有必要继续在这一方向投入大量训练时间。

## 10. F1：Stage1 到 Stage2 权重迁移

论文包含 Stage1 whole-image CNN finetune，再迁移到 Stage2 FisherNet 的过程。原来的迁移逻辑只加载 VGG convolution 层，没有完整映射 fc6/fc7。

补充映射：

```text
classifier.1 -> patch_mlp.0
classifier.4 -> patch_mlp.3
```

结果：

- 成功加载 30 个兼容 tensor；
- fc6/fc7 映射成功；
- 500-sample SVM：75.34%。

| 实验 | 500-sample mAP |
|---|---:|
| B1 | 75.28% |
| E1 | 75.53% |
| F1 | 75.34% |

从 B1、E1 和 F1 的差异看，Stage1 权重迁移不是当前主要瓶颈，后续应优先检查特征和 Fisher 训练状态。

## 11. G9：健康度诊断与 backbone 损坏定位

在 ResNet101 C4 checkpoint 上进行健康度分析：

- GAP+SVM mAP：7.71%；
- gamma 最大值均值：0.941；
- gamma entropy：0.190；
- 有效 Fisher 分量：2.16/64；
- means 严重漂移；
- sigmas 大幅变化；
- priors 明显漂移。

这表明 C4 训练过程同时破坏了 backbone 特征和 Fisher assignment，不能只把问题归因于 Fisher 层本身。

随后使用原始 torchvision pretrained ResNet101 做对照：

- GAP+SVM mAP：86.20%；
- backbone 特征本身具有很强分类信号；
- Fisher 参数冻结时，means/sigmas/priors 几乎零漂移。

从这组诊断可以判断：

> backbone 初始化不是问题，主要问题来自 Fisher assignment 和训练过程。

86.20% 只是 raw backbone 的 GAP+SVM 诊断基线，不是 Deep FisherNet 最终结果。

## 12. G10：固定 encoder 和 FV+SVM

G10A 固定 backbone 和 Fisher，仅训练 FC：

- FC val_mAP：16.23%；
- FV+SVM 500-sample mAP：30.86%。

这表明 FV 特征中仍然保留了可用信息，而当前 FC 优化没有充分利用这些信息；在这一阶段，外部线性 SVM 的效果明显好于 FC。

G10B 尝试 backbone 微调：

- backbone learning rate：`1e-5`；
- Fisher 冻结；
- 4 epochs；
- FC val_mAP：24.03%。

早期结果说明 backbone 微调可能有帮助，但 Fisher assignment 仍然需要单独修正。

## 13. G11：Fisher assignment 系统扫描

G11 对 sigma scale、assignment temperature 和 prior mode 进行了扫描。

| sigma | Temperature | Prior | FV+SVM mAP | gamma_eff |
|---:|---:|---|---:|---:|
| 1.0 | 1.0 | learned | 24.78% | 1.56 |
| 1.0 | 2.0 | learned | 43.07% | 10.2 |
| 8.0 | 4.0 | learned | 56.71% | 约 1.0 |

出现了两种策略：

### 13.1 SVM 表现较高、但 assignment 退化的配置

`sigma=8,T=4` 的 FV+SVM mAP 达到 56.71%，但 gamma_eff 约为 1，Fisher 基本退化成单分量硬分配。随后从零训练 FC 只有 17.49%。

该组合适合固定特征的 SVM 推理，但不适合端到端训练。

### 13.2 更适合训练的 assignment

`sigma=1,T=2` 的 FV+SVM mAP 为 43.07%，gamma_eff 为 10.2，多个 Fisher 分量被有效使用，适合作为训练起点。

基于这个对比，后续训练选择：

> **sigma=1,T=2**

## 14. G12：从固定 encoder 到 backbone 微调

### 14.1 G12A-Fresh

从零初始化 FC，冻结 backbone 和 Fisher：

| Epoch | val_mAP |
---:|---:|
| 1 | 17.87% |
| 2 | 25.13% |
| 3 | 25.78% |
| 4 | 26.45% |

这个结果说明，`sigma=1,T=2` 的表现不是旧 FC 头不适配造成的假象。不过 FC 在第 3、4 个 epoch 的增长已经明显放缓。

### 14.2 G12A-Hard

使用 `sigma=8,T=4` 从零训练 FC：

| Epoch | val_mAP |
---:|---:|
| 1 | 16.97% |
| 2 | 17.49% |

这进一步说明，SVM mAP 较高的极端 assignment 并不适合 FC 端到端训练。

### 14.3 G12B

在 `sigma=1,T=2` 下微调 backbone：

- FC val_mAP：28.34%；
- FV+SVM 500-sample：45.85%；
- VOC2007 test 全量 FV+SVM：**65.81%**。

完整测试集结果显著高于 500 张子集估计，说明 backbone 微调确实改善了 FV 特征质量。

## 15. G13：Fisher 参数 warmup

G13 开始以低学习率训练 Fisher 参数，backbone 暂时冻结。

结果：

- FC val_mAP：29.11%；
- means 相对漂移：0.00013；
- sigmas 相对漂移：0.00016；
- priors 相对漂移：0.00018。

这说明 Fisher 参数已经开始学习，同时没有发生大幅漂移或立即塌缩。

需要说明的是，G13 health 报告中的 gamma_eff 使用 G9 默认 T=1 统计，不能直接作为 T=2 assignment 的最终健康度结论。

## 16. 当前最佳结果的正确解释

当前有三个容易混淆的数字：

### 80.25%

VGG16 corrected-patches + Fisher Vector + SVM，VOC2007 test full，是当前全工程最高结果。

### 65.81%

ResNet101 spatial + `sigma=1,T=2` + backbone 微调 + FV+SVM，是 ResNet101/Fisher 新路线的完整测试结果。

它不是对 80.25% 的回退，而是另一条尚未完成完整训练的路线。

### 86.20%

原始 torchvision ResNet101 的 GAP+SVM 诊断结果，只用于证明 backbone 具有较强分类能力，不是 Deep FisherNet 最终结果。

## 17. 与论文 91.2% 的差距

论文的 91.2% 是完整端到端结果，而本项目的 80.25% 和 65.81% 都属于当前 PyTorch 复现链路中的阶段性结果。

剩余差距可能来自：

1. 官方 Caffe 权重和 `.mat` 参数缺失；
2. BGR/RGB、mean subtraction、resize 插值存在差异；
3. dense patch 坐标和边界处理未完全确认；
4. 官方 Stage1/Stage2 训练过程未完全复刻；
5. Fisher 参数训练尚未达到论文状态；
6. 多尺度训练和测试未完全恢复；
7. 当前 FC 优化流程没有充分利用 FV 特征。

已经通过实验排除的主要因素包括：

- 单纯 SVM C 值；
- 单纯延长训练 epoch；
- 单独 GMM resize；
- 单独 Stage1 权重迁移；
- 单纯更换 backbone；
- 单独改变 Fisher 公式。

## 18. 工程改进内容

项目期间完成了以下工程修复和工具建设：

- 增加 FV feature cache；
- 支持多种 SVM solver 和 C 值；
- 修复 GMM init 的 resize 对齐；
- 修复 VGG fc6/fc7 到 patch MLP 的权重映射；
- 实现 ResNet101 Caffe-v1 结构；
- 修正 pool1 padding 和 stride；
- 增加 Fisher 参数反推工具；
- 增加 gamma entropy 和有效分量监控；
- 增加 Fisher 参数漂移监控；
- 增加 sigma/temperature/prior 扫描脚本；
- 接入 Fisher assignment 参数；
- 完成 Caffe/PyTorch 初始化态 12/12 层对齐.

## 19. 最终结论

本项目没有完全复现论文的 91.2%，但完成了较完整的工程复现和问题定位。

当前成果可以总结为：

1. PyTorch 版本已经形成完整可运行链路。
2. Caffe/PyTorch 初始化态 Fisher 前向已实现 12/12 层数值对齐。
3. VGG16 主路线达到 **80.25% mAP**，为当前工程最高结果。
4. ResNet101/Fisher 路线在全量 VOC2007 test 上达到 **65.81% mAP**。
5. raw backbone GAP+SVM 达到 **86.20%**，证明 backbone 不是主要瓶颈。
6. Fisher assignment 塌缩是早期失败的重要原因。
7. `sigma=1,T=2` 是当前最适合训练的 assignment。
8. backbone 微调有效提升了 FV 表达质量。
9. 当前 FC 性能低于 FV+SVM，说明剩余瓶颈主要在分类器优化和端到端训练流程。
10. 论文 91.2% 仍需要更完整的官方参数、Caffe 预处理、训练流程和多尺度配置。

+++

### 以上是这几天我对考核的任务的实验研究和结果。下面是我尝试中的一些感受：

##### 复现难点与项目限制：

​	本项目从 Caffe 迁移到 PyTorch 的难点，首先不在于把网络结构重新写一遍，而在于论文所依赖的完整训练资产和实现细节并不齐全。当前公开资源主要是论文、部分 Caffe 工程代码和网络定义，缺少可以直接用于复现实验的完整模型权重、PCA/GMM 参数、Fisher 参数文件、训练日志以及最终评估脚本。因此，很多步骤只能根据代码、论文公式和实验结果逐项推断，无法简单地执行一个官方预训练模型得到 91.2%。

​	具体困难主要包括以下几方面：

1. **官方参数缺失。** 官方仓库中没有找到论文训练阶段使用的完整 `.caffemodel`，也没有找到对应的 PCA、GMM 和 Fisher `.mat` 参数。PCA 投影方向、GMM 均值和方差、分量先验都会直接影响 Fisher Vector，重新估计这些参数后，得到的实际上是一个近似复现版本。

2. **Caffe 自定义层语义不完全明确。** Fisher 层中的 prior、sigma、log-det、二阶项缩放和 pooling 方式，不能只依靠标准 PyTorch 层名来判断。部分参数是通过 Caffe 的 blob、权重层或 Softmax 间接实现的。通过 12/12 层逐层对齐已经排除了主要的前向实现错误，但初始化参数和训练状态仍无法与官方完全确认。

3. **预处理细节影响很大。** BGR/RGB 顺序、mean subtraction、resize 插值、图像边界、patch 坐标、stride、padding 和多尺度采样都会改变 patch descriptor。论文只给出了整体流程，许多细节没有完整记录。尤其是 dense patch 的边界处理和不同尺度下的 patch 数量，可能造成看似相同配置下的实际输入并不相同。

4. **Caffe 与 PyTorch 的数值环境不同。** 两个框架在卷积实现、BatchNorm 统计、插值、浮点累积顺序和 GPU kernel 上存在差异。单层误差虽然可能很小，但经过 PCA、GMM assignment 和 Fisher 聚合后会被放大。因此，本项目采用逐层比较和统计指标监控，而不是只比较最终 mAP。

5. **训练流程难以完整复原。** 论文包含 Stage1 whole-image CNN 训练、Stage2 patch/Fisher 训练、权重迁移、学习率调整和多尺度设置，但公开资料没有给出全部 epoch、batch size、数据增强、参数冻结顺序和学习率策略。当前虽然补全了 fc6/fc7 迁移并测试了 backbone/Fisher warmup，但仍不能保证与论文的完整训练轨迹一致。

6. **评估结果存在可比性限制。** 部分实验为了节省时间先使用 500 张样本筛选配置，最终结果再使用 VOC2007 test 全部 4952 张图像确认。500 张结果适合判断方向，但不能直接和全量 mAP 或论文的 91.2% 比较。raw backbone 的 GAP+SVM 86.20% 也只是诊断结果，不是完整 FisherNet 结果。

7. **训练目标和外部分类器存在差异。** 当前 FC 使用多标签 BCE 进行端到端训练，而 FV+SVM 是在提取完特征后对每个类别单独拟合线性分类器。两者优化目标、正则化方式和收敛特性不同，因此出现 FV+SVM 明显高于 FC 的情况并不矛盾，也不能仅凭 FC mAP 判断 Fisher 特征没有效果。

8. **部分现象只能通过实验间接判断。** 例如 GMM 初始化、assignment temperature、sigma scale 和 prior mode 会共同影响 Fisher 分配。实验可以证明某种配置在当前数据和实现下更好，但无法在缺少官方参数的情况下证明它就是论文中使用的唯一配置。因此报告中的结论应理解为当前工程的实验判断，而不是对官方实现细节的完全还原。

​	这次尝试基本完成了从 Caffe 自定义层到 PyTorch FisherNet 的主要结构和数值对齐，并通过多组消融实验定位了 backbone、BatchNorm、assignment、分类器和训练流程的影响。但由于确实感觉到原始项目很古早，很多信息来源很单一，当前结果更准确的表述是“基于公开代码和论文信息完成的工程复现与改进”，而不是对论文训练环境的一比一复刻。可能是当前结果与论文 91.2% 之间仍存在差距的主要原因。
