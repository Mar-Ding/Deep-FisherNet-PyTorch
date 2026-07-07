# Deep FisherNet 租卡训练计划

> 目标：在 4090 上复现 Deep FisherNet (NIPS 2016) 在 PASCAL VOC 2007 的 mAP

---

## 环境准备

```bash
# 1. 克隆项目
git clone <repo-url> fishernet
cd fishernet

# 2. 安装依赖
pip install -r requirements.txt

# 3. 确认 PyTorch 可用 CUDA
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

预期输出：`2.x`, `True`, `NVIDIA GeForce RTX 4090`

---

## 训练流程（Stage 1 → Stage 2 → Eval）

### Step 1: GMM 初始化（~10 min）

```bash
python scripts/init_gmm.py \
    --preset vgg16-paper-like \
    --data-root data \
    --pretrained \
    --num-workers 4
```

输出：`outputs/fishernet_voc2007/fisher_gmm.pt`

> 如果数据不在 `data/VOCdevkit/VOC2007`，加上 `--download` 会自动下载（~500MB）

### Step 2: Stage 1 — 整图 CNN finetune（~2-3h）

9k iters × batch=32 / 5011 ≈ **57 epochs**

```bash
python scripts/train.py \
    --preset stage1-vgg16-paper \
    --data-root data \
    --output-dir outputs/stage1 \
    --pretrained \
    --num-workers 4
```

输出：
- `outputs/stage1/last.pt` — 最后一轮 checkpoint（含 features.\* 权重）
- `outputs/stage1/best.pt` — 最佳 checkpoint

### Step 3: Stage 2 — FisherNet 端到端（~8-12h）

40k iters × batch=2 / 5011 ≈ **16 epochs**

```bash
python scripts/train.py \
    --preset vgg16-paper-like \
    --data-root data \
    --output-dir outputs/stage2 \
    --stage1-weights outputs/stage1/last.pt \
    --fisher-init outputs/fishernet_voc2007/fisher_gmm.pt \
    --pretrained \
    --num-workers 4
```

三层差异化 lr：
- **backbone** = 0.001（预训练卷积层）
- **classifier** = 0.1（新增 256-dim FC + 20-class 分类头）
- **Fisher layer** = 0.0001（GMM 参数）

### Step 4: 评估（~30 min）

#### 4a. 单尺度 baseline
```bash
python scripts/evaluate.py \
    --preset vgg16-paper-like \
    --data-root data \
    --checkpoint outputs/stage2/best.pt
```

#### 4b. 多尺度（论文 5 scales + flip → 10-view averaging）
```bash
python scripts/evaluate.py \
    --preset vgg16-paper-like \
    --data-root data \
    --checkpoint outputs/stage2/best.pt \
    --test-scales 480 576 688 864 1200
```

#### 4c. 完整论文流程：多尺度 + SVM（论文标配）
```bash
python scripts/evaluate.py \
    --preset vgg16-paper-like \
    --data-root data \
    --checkpoint outputs/stage2/best.pt \
    --test-scales 480 576 688 864 1200 \
    --fit-svm --svm-C 1.0
```

---

## 预期耗时

| 步骤 | 预计时间 | 说明 |
|------|---------|------|
| GMM 初始化 | ~10 min | 50K descriptor 采样 + sklearn GMM fit |
| Stage 1 训练 | **~2-3 h** | 57 epochs, batch=32, 纯卷积 |
| Stage 2 训练 | **~8-12 h** | 16 epochs, batch=1, 800 patches/image |
| 多尺度评估 | ~30 min | 10 views/image × 4952 test images |
| **总计** | **~12-16 h** | 可 overnight 跑完 |

Stage 2 瓶颈在 RoI Align + 800 patches forward，4090 上一步 ~0.1s → 16 × 5011 × 0.1 ≈ 2.2h forward，加上 backward 约 4-6x → 8-12h 合理。

---

## 监控技巧

```bash
# 实时看 loss 趋势
tail -f outputs/stage2/training.log  # 如果 stdout 已重定向

# GPU 使用率
watch -n 2 nvidia-smi

# 中间 checkpoint 评估
python scripts/evaluate.py --preset vgg16-paper-like --data-root data \
    --checkpoint outputs/stage2/last.pt --test-scales 480 576 688 864 1200
```

---

## 连接方式

晚上你发 SSH 命令给我，我会：
1. 传项目文件（`rsync` 或 `git clone`）
2. 跑一遍环境检查
3. 从 Step 1 开始执行
4. 每个步骤完成后汇报结果给终端
