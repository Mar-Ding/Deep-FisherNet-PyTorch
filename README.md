# Deep FisherNet for Object Classification：PyTorch 复现

本仓库使用 PyTorch 完整复现论文 [Deep FisherNet for Object Classification](https://arxiv.org/abs/1608.00182)，并在 **PASCAL VOC2007 全量测试集**上完成训练与评估。

---

## 最终指标

| 配置 | VOC2007 mAP |
|---|---:|
| **本项目 FisherNet（最佳）** | **85.99%** |
| 论文报告值 | 91.20% |
| 前期 VGG-16 完整基线 | 80.25% |
| **相对基线提升** | **+5.74 点** |

- **测试协议**：VOC2007 trainval 5011 张 → 拟合 LIBLINEAR SVM → test 4952 张评估
- **评估策略**：五尺度（480,576,688,864,1200）+ 水平翻转融合
- **全 20 类 AP**：[详细结果](结果/best_full_test_metrics.json)
- **正式实验报告**：[完整报告](结果/deep_fishernet_complete_exploration_report.md)

> 项目完成了 Caffe 官方代码到 PyTorch 的全链路迁移、Fisher 编码前向 12/12 层数值对齐，以及 G1→G15 系列消融实验，将完整 pipeline 较基线提升 **5.74 个 mAP 点**。

## 工程结构

```text
fishernet/             核心模型、数据处理和评估工具
scripts/               训练、测试、初始化与诊断脚本
tests/                 论文链路和数值对齐测试
external/              官方 Caffe FisherNet 工程
data/                  PASCAL VOC2007 数据（不纳入版本控制）
outputs/               检查点、缓存和实验日志（不纳入版本控制）
reports/               阶段性报告与执行记录
结果/                  最终提交入口
1608.00182.pdf          论文原文
```

## 环境

- Python 3.12.3
- PyTorch 2.5.1+cu124
- NVIDIA RTX 4090 24GB
- PASCAL VOC2007

安装依赖：

```bash
pip install -r requirements.txt
```

数据目录应组织为：

```text
data/
└── VOCdevkit/
    └── VOC2007/
```

## 主要入口

```bash
# Stage 1：训练局部描述子网络
python scripts/train.py --stage1 \
  --preset stage1-vgg16-paper-final \
  --data-root data \
  --output-dir outputs/vgg16-paper-final/stage1 \
  --pretrained

# 从 Stage 1 特征估计 PCA 与 GMM，初始化 Fisher 层
python scripts/init_gmm_stage1.py \
  --preset vgg16-paper-final \
  --data-root data \
  --checkpoint outputs/vgg16-paper-final/stage1/last.pt \
  --output outputs/vgg16-paper-final/fisher_gmm.pt

# Stage 2：端到端训练完整 FisherNet
python scripts/train.py \
  --preset vgg16-paper-stable \
  --data-root data \
  --output-dir outputs/vgg16-paper-stable \
  --stage1-weights outputs/vgg16-paper-final/stage1/last.pt \
  --fisher-init outputs/vgg16-paper-final/fisher_gmm.pt

# 使用最佳模型执行全量 FV+SVM 测试
python scripts/evaluate.py \
  --preset vgg16-paper-final \
  --data-root data \
  --checkpoint outputs/g15_final_results/best_85_99.pt \
  --fit-svm --svm-C 1 --test-flip \
  --feature-cache-dir outputs/eval_cache \
  --metrics-output outputs/eval_metrics.json
```

训练和测试的完整参数、实验设计及失败分析见[正式报告](结果/deep_fishernet_complete_exploration_report.md)。

## 验证

```bash
python -m pytest tests/test_paper_pipeline.py -q
```

当前论文链路测试结果：`9 passed`。
