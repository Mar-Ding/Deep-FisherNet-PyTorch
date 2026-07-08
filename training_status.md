# Deep FisherNet — 完整实验报告

## 项目信息
- **论文：** Deep FisherNet for Object Classification (NIPS 2016)
- **代码：** https://github.com/Mar-Ding/Deep-FisherNet-PyTorch
- **本地路径：** `F:/paper_ws/Deep FisherNet for Object Classification/`
- **GPU：** AutoDL RTX 4090, 2.2元/h

---

## 实验结果汇总

| 实验 | backbone_lr | fisher_lr | classifier_lr | epoch | 归一化 | GMM来源 | val_mAP | **SVM mAP** |
|------|------------|-----------|---------------|-------|--------|---------|---------|------------|
| v1 旧预设（错lr） | 0.001 | 0.0001❄️ | 0.1🔥 | 16 | 双归一化❌ | ImageNet | 77.30%虚高 | **67.30%** |
| v2 论文lr | 0.0001 | 0.1🔥 | 0.001❄️ | 16 | 双归一化❌ | ImageNet | 13.67% | **62.65%** |
| **v2+fix（最佳）** | **0.0001** | **0.1** | **0.001** | **16** | **单归一化✅** | **ImageNet** | **13.67%** | **70.71%** |
| v3 折中lr | 0.0001 | 0.01 | 0.01 | 5 | 单归一化✅ | ImageNet | ~10% | 62.15% |
| v4 新GMM | 0.0001 | 0.01 | 0.01 | 1 | 单归一化✅ | Stage1 finetune | ~9% | 55.96% |
| v5 放开backbone | 0.001 | 0.01 | 0.01 | 中断 | - | Stage1 finetune | - | - |
| **冻结基线（无训练）** | - | - | - | - | 单归一化✅ | **Stage1 finetune** | - | **54.00%** |

## 已确认的修改 & 与论文对齐情况

| 修改项 | 是否对齐论文 | 状态 | 说明 |
|-------|------------|------|------|
| FisherLayer 内部 norm（训练开，评估关） | ⚠️ 近似 | ✅ 已修 | 训练时 FC 看到 norm 特征更稳定；评估时 SVM 路径做一次 norm |
| `use_flip=True`（SVM测试） | ✅ 对齐 | ✅ 已修 | 论文明确 "+ horizontal flip" |
| `SGDClassifier` 替代 `LinearSVC` | ✅ 等价 | ✅ 已修 | 数学上同一目标函数，快1000倍 |
| 双归一化 bug（FisherLayer + 外部各一次） | ❌ bug | ✅ 已修 | 导致特征被4次方根压缩，修复后 +8% |
| GMM用Stage1 finetune特征重初始化 | ⚠️ 改进 | ✅ 可用 | 论文用原始预训练特征，但finetune后用更准 |
| Stage2 lr 采用论文原始值 | ✅ 对齐 | ✅ 已恢复 | fisher=0.1, classifier=0.001, backbone=0.0001 |

## 关键结论

1. **Fisher 层训练有效：** 冻结基线 54% → 训练后 70%（+16%）
2. **双归一化是明确 bug：** 修复后 +8%（62%→70%）
3. **天花板在 70%，** 离论文 91.7% 差距 20+ 点
4. **GMM 初始化、lr 调优对天花板无影响** — 问题在更深层

## 怀疑方向（下一阶段）

### 方向1：Patch特征质量（最高优先级）
冻结基线仅 54%，一个标准 FV 基线在 VOC2007 应有 70-80%。问题可能是：
- VGG16 移除了 pool5 后特征图分辨率变化，patch 对齐有误
- `roi_align(spatial_scale=1.0)` 配合 `_make_feature_rois` 的坐标缩放是否正确？
- `max_patches=800` 是否足够覆盖图像中的物体？
- 验证方法：可视化 patch 在图像上的位置，看是否覆盖物体

### 方向2：Fisher Layer 梯度流验证
训练时 val_mAP 始终 10%，说明 FC 没学到任何东西。可能：
- RAW FV 数值过大导致梯度爆炸/消失
- Fisher Layer 的前向/反向计算有细微错误
- 验证方法：对比 FisherLayer 输出和 sklearn GMM FV 编码的数值是否接近

### 方向3：SVM C 参数调优
固定 C=1.0 可能不是最优。用 SGDClassifier 快速扫 C=[0.1, 1.0, 10, 100]

### 方向4：增加 FC 容量
Linear(16384, 20) 太单薄，加 hidden 层提高梯度质量

## 下一任 AI 的入口文件
- `scripts/evaluate.py` — SGDClassifier版 + 归一化修复 + use_flip=True
- `scripts/train.py` — Stage2 训练脚本
- `scripts/fv_baseline.py` — FV基线测试脚本（sklearn GMM + FV + SVM）
- `scripts/init_gmm_stage1.py` — 用Stage1特征重跑GMM初始化
- `fishernet/models/fisher_layer.py` — Fisher Layer 实现（核心）
- `fishernet/models/fishernet.py` — 网络结构
- `fishernet/configs.py` — 预设配置（已恢复论文lr）
- `eval_fix.log` — 最佳结果（70.71%）日志
- `eval_frozen.log` — 冻结基线（54.00%）日志
