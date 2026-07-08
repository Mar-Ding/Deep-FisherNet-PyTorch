# Deep FisherNet VGG16 — 训练总结

**远程 GPU：** AutoDL RTX 4090
**本地路径：** `F:/paper_ws/Deep FisherNet for Object Classification/`
**GitHub：** `https://github.com/Mar-Ding/Deep-FisherNet-PyTorch`

---

## 最终结果

| 版本 | backbone_lr | fisher_lr | classifier_lr | 归一化 | SVM mAP |
|------|------------|-----------|---------------|--------|---------|
| **v2+fix（最佳）** | 0.0001 | 0.1 | 0.001 | 单次（修复后） | **70.71%** |
| v1（旧预设错lr） | 0.001 | 0.0001❄️ | 0.1🔥 | 双归一化 | 67.30% |
| v2（论文lr） | 0.0001 | 0.1 | 0.001 | 双归一化 | 62.65% |
| v3（折中lr） | 0.0001 | 0.01 | 0.01 | 单次 | 62.15% |
| v4（新GMM） | 0.0001 | 0.01 | 0.01 | 单次 | 55.96% |

## 已确认的修改

| 修改 | 效果 | 说明 |
|-----|------|------|
| 修复双归一化（FisherLayer+外部双norm） | +8% | 明确bug，已修 |
| SGDClassifier 替代 LinearSVC | 数学等价，速度1000x | 已集成 |
| use_flip=True（SVM测试） | 对齐论文 | 已集成 |
| GMM重初始化 | 无明显改善 | 保留可选 |

## 训练进度

- Stage 1（VGG16 finetune, 57 epochs）: ✅ **81.93% val_mAP**
- Stage 2（FisherNet, 16 epochs）: 🔄 第5轮训练中（PID 4930）
- SVM评估: 最近一次 v2+fix = **70.71%**

## 文件说明

- `scripts/evaluate.py` — SGDClassifier版 + use_flip=True + 外部归一化
- `fishernet/configs.py` — 论文原始lr
- `training_status.md` — 本文件
- `eval_fix.log` — 最佳结果（70.71%）日志
