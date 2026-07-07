# Deep FisherNet for Object Classification

PyTorch reproduction scaffold for **Deep FisherNet for Object Classification** on PASCAL VOC 2007.

## Setup

```powershell
pip install -r requirements.txt
```

## Smoke Test

```powershell
python scripts/smoke_test.py
```

## Train on VOC 2007

```powershell
python scripts/init_gmm.py --data-root data --download --pretrained
python scripts/train.py --data-root data --pretrained --fisher-init outputs/fishernet_voc2007/fisher_gmm.pt --epochs 10 --batch-size 1
```

For a quicker baseline without GMM initialization:

```powershell
python scripts/train.py --data-root data --download --pretrained --epochs 10 --batch-size 1
```

## Evaluate

```powershell
python scripts/evaluate.py --data-root data --checkpoint outputs/fishernet_voc2007/best.pt
```

The implementation report is in `reports/implementation_report.md`.
