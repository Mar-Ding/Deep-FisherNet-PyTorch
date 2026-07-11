# Deep FisherNet 考核任务交付说明

## 提交内容

| 文件 | 内容 |
|---|---|
| [deep_fishernet_complete_exploration_report.md](deep_fishernet_complete_exploration_report.md) | 正式复现与实验报告 |
| [best_full_test_metrics.json](best_full_test_metrics.json) | VOC2007 test 全量最佳结果及 20 类 AP |
| [../1608.00182.pdf](../1608.00182.pdf) | 复现论文原文 |

## 最终结果

- 论文结果：91.20% mAP
- 本项目完整 FisherNet：**85.99% mAP**
- 前期完整基线：80.25% mAP
- 相对基线提升：**5.74 个 mAP 点**
- 评估集：VOC2007 test 全部 4952 张图像
- 模型：VGG16 + Fisher Vector + LIBLINEAR
- 测试：五尺度 480/576/688/864/1200，加水平翻转

## 模型与证据

最佳模型保存在：

`../outputs/g15_final_results/best_85_99.pt`

- 文件大小：1,085,254,846 bytes
- SHA256：`39804F231ABBCAD586EF1ED4FDDA8539D53B6AE3B04FAF55925CCD246FCABA10`

完整日志和对照指标保存在 `../outputs/g15_final_results/`。模型文件较大，因此不在本目录重复存放。

## 代码入口

- 训练：`../scripts/train.py`
- 测试：`../scripts/evaluate.py`
- GMM/PCA 初始化：`../scripts/init_gmm_stage1.py`
- 模型实现：`../fishernet/models/`
- 数据处理：`../fishernet/data/`
- 对齐测试：`../tests/test_paper_pipeline.py`

