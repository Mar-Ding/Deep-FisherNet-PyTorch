# Deep FisherNet — 完整实验报告

## 项目信息
- **论文：** Deep FisherNet for Object Classification (NIPS 2016)
- **代码：** https://github.com/Mar-Ding/Deep-FisherNet-PyTorch
- **本地路径：** `F:/paper_ws/Deep FisherNet for Object Classification/`
- **GPU：** AutoDL RTX 4090, 2.2元/h

---

## 当前最新结果

| 实验 | 描述 | val_mAP | SVM mAP(500) | **SVM Full mAP** |
|------|------|---------|-------------|-----------------|
| **B1** | 修正VGG补丁流水线，4 epoch | 0.1140 | **75.59%** | **79.76%** ✅ |
| B2 1ep | 官方Fisher数学，1 epoch | 0.1155 | 待评估 | 待评估 |

## B1 SVM Full: **79.76% mAP**（较旧最佳70.71%提升+9%）

每类AP：
| 类别 | AP | 类别 | AP | 类别 | AP | 类别 | AP |
|------|----|------|----|------|----|------|----|
| aeroplane | 95.50 | bicycle | 85.91 | bird | 92.33 | boat | 90.55 |
| bottle | 56.61 | bus | 80.16 | car | 86.68 | cat | 86.27 |
| chair | 63.54 | cow | 70.39 | diningtable | 72.59 | dog | 85.49 |
| horse | 87.78 | motorbike | 80.99 | person | 93.96 | pottedplant | 52.05 |
| sheep | 81.84 | sofa | 62.28 | train | 93.95 | tvmonitor | 76.40 |

## B1 C参数扫描（500样本）
| C | mAP |
|---|-----|
| 0.1 | 72.46% |
| 1.0 | 74.15% |
| 10 | **75.28%** |
| 100 | 75.03% |

## 服务器实验清单
| 路径 | 说明 |
|------|------|
| `outputs/b1_vgg_corrected_4ep/best.pt` | ✅ B1最佳 (1.01GB) |
| `outputs/b2_vgg_official_fisher_1ep/best.pt` | ✅ B2 1ep (1.01GB) |
| `outputs/agent_logs/b1_svm_full.log` | ✅ B1全量SVM结果 |
| `outputs/agent_logs/b1_train.log` | ✅ B1训练日志 |
| `outputs/agent_logs/b2_train_1ep.log` | ✅ B2 1ep训练日志 |

## 下阶段
1. **P0: B2 1ep SVM评估** — 跑500样本SVM看B2是否优于B1
2. **P1: B2 4 epoch训练** — 如果B2有希望
3. **P2: C参数调优** — 对B1完整SVM用C=10重跑优化

## 连接
- `ssh -p 23701 root@connect.bjb1.seetacloud.com` / 密码 BeR8KvW4XOV8

更新日期: 2026-07-08
