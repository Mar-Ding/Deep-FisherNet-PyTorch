# Deep FisherNet 4090 Handoff Log

Date: 2026-07-07

Project: PyTorch reproduction of **Deep FisherNet for Object Classification** on PASCAL VOC 2007.

Workspace on current laptop:

```text
F:\paper_ws\Deep FisherNet for Object Classification
```

## Current Status

The project has a runnable PyTorch implementation with:

- VOC 2007 dataset loader.
- Dense patch generation.
- Differentiable Fisher layer with trainable GMM-like parameters.
- AlexNet FisherNet backbone.
- ResNet101 FisherNet backbone.
- Preset configs for smoke tests, AlexNet-like experiments, and official ResNet101-like experiments.
- Training, evaluation, benchmark, and GMM initialization scripts.

The current laptop is:

```text
NVIDIA GeForce RTX 4060 Laptop GPU, 8GB VRAM
```

It can initialize GMM and run smoke tests, but full 4 epoch official-like training is better moved to a 4090.

## Dataset

VOC 2007 data is already present:

```text
data/VOCdevkit/VOC2007
```

Expected counts already checked:

- trainval: 5011 images
- test: 4952 images
- JPEGImages/Annotations: 9963 files

Original tarballs are also present:

```text
VOCtrainval_06-Nov-2007.tar
VOCtest_06-Nov-2007.tar
```

On the 4090 machine, either copy the full project directory or recreate:

```text
data/VOCdevkit/VOC2007
```

## Important Files

Main implementation:

```text
fishernet/models/fisher_layer.py
fishernet/models/fishernet.py
fishernet/configs.py
fishernet/data/voc.py
scripts/train.py
scripts/evaluate.py
scripts/init_gmm.py
scripts/benchmark_step.py
```

Official Caffe reference repo:

```text
external/official-fishernet
```

User-provided training strategy comparison:

```text
training_strategy_comparison.md
```

Earlier implementation report:

```text
reports/implementation_report.md
```

## Official-Like Config

The main config to continue with is:

```text
--preset official-res101-like
```

It is based on the public official repository's ResNet101 strategy, not every experiment in the paper. Key settings:

- backbone: ResNet101
- ImageNet pretrained backbone: yes, use `--pretrained`
- Fisher components: `K=64`
- patch descriptor dimension: `D=128`
- Fisher vector dimension: `2 * K * D = 16384`
- train scales: `336 504 672 840 1008`
- horizontal flip probability: `0.5`
- patch sizes: `64 96 128 160 192 224 256`
- patch stride: `32`
- max patches per image: `800`
- optimizer: SGD
- base/backbone lr: `0.001`
- classifier weight/bias lr: `0.001 / 0.002`
- Fisher weight/bias lr: `0.01 / 0.02`
- momentum: `0.9`
- weight decay: `0.0005`
- grad accumulation steps: `8`

Important nuance:

The public official ResNet101 Caffe config uses `SoftmaxWithLoss` and a classifier output size shown as 8 in the released model file, while this PyTorch implementation is VOC multi-label and uses `BCEWithLogitsLoss` with 20 classes. This is intentional for PASCAL VOC classification but is not a byte-for-byte Caffe reproduction.

## Completed Checks

Static checks:

```powershell
python -m compileall fishernet scripts
python scripts\train.py --help
python scripts\evaluate.py --help
```

All passed.

Preset model instantiation:

```text
smoke                  AlexNetFisherNet  fv_dim=1024   classes=20
alexnet-paper-like      AlexNetFisherNet  fv_dim=16384  classes=20
official-res101-like    ResNetFisherNet   fv_dim=16384  classes=20
```

VOC sample read check:

```text
len = 5011
image = (3, 224, 224)
labels = (20,)
boxes = (8, 4)
first image_id = 000005.jpg
```

Smoke chain test completed:

```powershell
python scripts\train.py --preset smoke --data-root data --output-dir outputs\smoke_chain_test --epochs 1 --max-train-batches 2 --max-val-batches 2 --num-workers 0 --no-progress
```

Result:

```text
epoch=1 train_loss=0.6902 val_mAP=0.6667
```

Standalone evaluation loading checkpoint completed:

```powershell
python scripts\evaluate.py --preset smoke --data-root data --checkpoint outputs\smoke_chain_test\last.pt --max-batches 2 --num-workers 0
```

Result:

```text
mAP: 0.6667
```

This mAP is meaningless scientifically because only 2 validation batches were used. It only proves the train-save-evaluate chain works.

## Benchmark Results on RTX 4060 Laptop

Command family:

```powershell
python scripts\benchmark_step.py --preset official-res101-like --image-size <scale> --steps 5
```

Results:

| image size | avg step after warmup | max memory |
|---:|---:|---:|
| 336 | 0.096 s | 4.76 GB |
| 504 | 0.096 s | 5.10 GB |
| 672 | 0.108 s | 5.55 GB |
| 840 | 0.134 s | 6.16 GB |
| 1008 | 0.182 s | no OOM in test |

These numbers are synthetic benchmark numbers. Full training includes disk IO, VOC transforms, random multi-scale sampling, checkpointing, and validation, so real epoch time is longer.

Laptop estimate before moving to 4090:

- 1 epoch: roughly 20-45 minutes
- 4 epochs: roughly 1.5-3 hours

This is why full training should move to 4090.

## GMM Initialization

Official-like GMM initialization completed successfully.

Command:

```powershell
python scripts\init_gmm.py --preset official-res101-like --data-root data --output outputs\official_res101_like\fisher_gmm.pt --pretrained --num-workers 0
```

Output:

```text
outputs\official_res101_like\fisher_gmm.pt
```

Verified contents:

```text
means  shape = (64, 128), dtype=float32
sigmas shape = (64, 128), dtype=float32
priors shape = (64,), dtype=float32
```

This file is compatible with `official-res101-like`.

Do not use:

```text
outputs\fishernet_voc2007\fisher_gmm.pt
```

That older file is from an AlexNet/K32/D256 experiment and is not compatible with ResNet101 K64/D128.

## Existing Outputs

Useful:

```text
outputs/official_res101_like/fisher_gmm.pt
```

Smoke only:

```text
outputs/smoke_chain_test/last.pt
outputs/smoke_chain_test/best.pt
```

Old low-quality engineering run, not a formal reproduction:

```text
outputs/fishernet_voc2007/last.pt
outputs/fishernet_voc2007/best.pt
outputs/fishernet_voc2007/fisher_gmm.pt
```

Do not report old `outputs/fishernet_voc2007` mAP as the reproduction result.

## Recommended 4090 Procedure

1. Create environment and install dependencies.

```powershell
pip install -r requirements.txt
```

2. Confirm CUDA/PyTorch.

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

3. Run a quick official-like limited chain test on 4090.

```powershell
python scripts\train.py --preset official-res101-like --data-root data --output-dir outputs\official_res101_like_4090_probe --fisher-init outputs\official_res101_like\fisher_gmm.pt --pretrained --epochs 1 --max-train-batches 8 --max-val-batches 8 --num-workers 4
```

This verifies that the copied GMM, dataset, pretrained backbone, GPU, and script all work together.

4. If probe passes, run 4 epoch formal training.

```powershell
python scripts\train.py --preset official-res101-like --data-root data --output-dir outputs\official_res101_like_4epoch --fisher-init outputs\official_res101_like\fisher_gmm.pt --pretrained --epochs 4 --num-workers 4
```

5. Evaluate the best checkpoint on VOC 2007 test.

```powershell
python scripts\evaluate.py --preset official-res101-like --data-root data --checkpoint outputs\official_res101_like_4epoch\best.pt --num-workers 4
```

Record:

- final `mAP`
- per-class AP
- training wall time
- GPU model
- peak VRAM if possible
- exact command used

## If Training Is Too Slow

Try these in order:

1. Increase `--num-workers` to 8 if CPU/disk can keep up.
2. Keep `batch-size=1` because random multi-scale images have different shapes.
3. Keep `grad-accum-steps=8` to follow the official-like effective batch strategy.
4. Avoid reducing `max-patches` for the formal run unless the goal changes to engineering demonstration only.

## Known Gaps Before Claiming Paper-Level Reproduction

This implementation is a practical PyTorch reproduction, not an exact Caffe clone. Known gaps:

- The official Caffe code has custom Fisher layers; this project reimplements them in PyTorch.
- Scheduler is epoch-ratio based, while official Caffe Res101 solver uses iteration milestone `stepvalue 8000`.
- Current evaluation uses single-scale square resizing through the dataset wrapper; the paper/official configs mention multi-scale test behavior.
- The PyTorch training loss is multi-label BCE for VOC20; official released Res101 file includes a different classifier head in its prototxt.
- No long official-like training result has been produced yet.

For the report, describe results honestly as:

```text
PyTorch reproduction of Deep FisherNet idea with official ResNet101-like hyperparameters.
```

Do not claim:

```text
Exact reproduction of the paper's final VOC 2007 number.
```

until the 4090 training and evaluation are complete and the remaining gaps are addressed or explicitly justified.

