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
        "backbone_lr": 1e-4,
        "classifier_lr": 1e-4,
        "classifier_bias_lr": 1e-4,
        "fisher_lr": 1e-4,
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
        "optimizer": "sgd",
        "lr": 1e-3,
        "backbone_lr": 1e-3,
        "classifier_lr": 1e-3,
        "classifier_bias_lr": 2e-3,
        "fisher_lr": 1e-2,
        "fisher_bias_lr": 2e-2,
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
    "official-res101-like": {
        "backbone": "resnet101",
        "optimizer": "sgd",
        "lr": 1e-3,
        "backbone_lr": 1e-3,
        "classifier_lr": 1e-3,
        "classifier_bias_lr": 2e-3,
        "fisher_lr": 1e-2,
        "fisher_bias_lr": 2e-2,
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
