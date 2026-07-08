from __future__ import annotations

from argparse import Namespace
from copy import deepcopy
from typing import Any


PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "backbone": "alexnet",
        "epochs": 1,
        "batch_size": 1,
        "optimizer": "adamw",
        "lr": 1e-4,
        "backbone_lr": 1e-2,
        "classifier_lr": 1e-3,
        "classifier_bias_lr": 1e-4,
        "fisher_lr": 1e-1,
        "fisher_bias_lr": 1e-4,
        "weight_decay": 1e-4,
        "grad_accum_steps": 1,
        "image_size": 224,
        "train_scales": None,
        "hflip_prob": 0.0,
        "patch_sizes": [96, 128, 192, 256],
        "patch_stride": 64,
        "max_patches": 32,
        "patch_dim": 64,
        "num_components": 8,
    },
    "alexnet-paper-like": {
        "backbone": "alexnet",
        "epochs": 10,
        "optimizer": "sgd",
        "lr": 1e-3,
        "backbone_lr": 1e-4,
        "classifier_lr": 1e-3,
        "classifier_bias_lr": 2e-3,
        "fisher_lr": 1e-1,
        "fisher_bias_lr": 2e-1,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "grad_accum_steps": 8,
        "lr_gamma": 0.1,
        "train_scales": [336, 504, 672, 840, 1008],
        "hflip_prob": 0.5,
        "patch_sizes": [64, 96, 128, 160, 192, 224, 256],
        "patch_stride": 32,
        "max_patches": 800,
        "patch_dim": 256,
        "num_components": 32,
    },
    "vgg16-paper-like": {
        # Exact hyperparameters from the NIPS 2016 paper (Section 4.1)
        "backbone": "vgg16",
        "epochs": 16,  # 40k iters × batch=2 / 5011 images ≈ 16 epochs
        "batch_size": 1,
        "optimizer": "sgd",
        "lr": 1e-3,       # backbone base_lr
        "backbone_lr": 1e-4,
        "classifier_lr": 1e-3,       # paper: 0.001 for score FC
        "classifier_bias_lr": 2e-3,  # 2× rule for bias
        "fisher_lr": 1e-1,           # paper: 0.1 for Fisher Layer
        "fisher_bias_lr": 2e-1,      # 2× rule for bias
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "grad_accum_steps": 2,       # batch=1 × accum=2 → effective batch=2 (matches paper)
        "lr_step_ratio": 0.6,        # LR drops at 60% of total epochs
        "lr_gamma": 0.1,
        "train_scales": [480, 576, 688, 864, 1200],  # paper's 5 scales
        "hflip_prob": 0.5,
        "patch_sizes": [64, 96, 128, 160, 192, 224, 256],  # paper: 7 scales
        "patch_stride": 32,
        "max_patches": 800,
        "patch_dim": 256,            # paper: 256-dim reduction
        "num_components": 32,        # paper: K=32
        "roi_output_size": 7,        # VGG16 fc6 expects 512×7×7
    },
    "stage1-vgg16-paper": {
        "backbone": "vgg16",
        "epochs": 57,       # 9k iters × batch=32 / 5011 images ≈ 57 epochs
        "batch_size": 32,
        "optimizer": "sgd",
        "backbone_lr": 1e-2,     # paper: lr=0.01 for most layers
        "classifier_lr": 1e-3,   # paper: lr=0.001 for last fc layer
        "classifier_bias_lr": 2e-3,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "lr_step_ratio": 0.6,
        "lr_gamma": 0.1,
        "image_size": 224,       # whole-image finetune uses fixed 224×224
    },
    "official-res101-like": {
        "backbone": "resnet101",
        "epochs": 10,
        "optimizer": "sgd",
        "lr": 1e-3,
        "backbone_lr": 1e-4,
        "classifier_lr": 1e-3,
        "classifier_bias_lr": 2e-3,
        "fisher_lr": 1e-1,
        "fisher_bias_lr": 2e-1,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "grad_accum_steps": 8,
        "lr_gamma": 0.1,
        "train_scales": [336, 504, 672, 840, 1008],
        "hflip_prob": 0.5,
        "patch_sizes": [64, 96, 128, 160, 192, 224, 256],
        "patch_stride": 32,
        "max_patches": 800,
        "patch_dim": 128,
        "num_components": 64,
        "roi_output_size": 1,
    },
}


def apply_preset(args: Namespace) -> Namespace:
    if args.preset is None:
        return args
    values = deepcopy(PRESETS[args.preset])
    explicit = set(getattr(args, "_explicit_args", set()))
    for key, value in values.items():
        cli_name = "--" + key.replace("_", "-")
        if cli_name not in explicit:
            setattr(args, key, value)
    return args
