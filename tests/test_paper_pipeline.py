from __future__ import annotations

import unittest

import torch
from torch import nn

from fishernet.configs import PRESETS
from fishernet.data.patches import make_dense_patch_boxes, transform_patch_boxes
from fishernet.data.voc import collate_voc_batch
from fishernet.models import load_stage1_weights
from fishernet.utils.metrics import mean_average_precision
from scripts.evaluate import normalize_fv_tensor
from scripts.train import append_caffe_param_groups, multilabel_bce


class PaperPatchTests(unittest.TestCase):
    def test_caffe_boxes_are_inclusive_and_scale_from_original(self) -> None:
        boxes = make_dense_patch_boxes(
            height=100,
            width=120,
            patch_sizes=(64,),
            stride=32,
            max_patches=None,
            coordinate_mode="caffe",
        )
        torch.testing.assert_close(boxes[0], torch.tensor([0.0, 0.0, 63.0, 63.0]))
        scaled = transform_patch_boxes(
            boxes,
            source_hw=(100, 120),
            target_hw=(200, 240),
            coordinate_mode="caffe",
        )
        self.assertEqual(boxes.shape[0], scaled.shape[0])
        torch.testing.assert_close(scaled[0], torch.tensor([0.0, 0.0, 126.0, 126.0]))

    def test_caffe_horizontal_flip_preserves_box_order_and_extent(self) -> None:
        box = torch.tensor([[10.0, 4.0, 29.0, 23.0]])
        flipped = transform_patch_boxes(
            box,
            source_hw=(80, 100),
            target_hw=(80, 100),
            horizontal_flip=True,
            coordinate_mode="caffe",
        )
        torch.testing.assert_close(flipped, torch.tensor([[70.0, 4.0, 89.0, 23.0]]))

    def test_variable_size_batch_is_padded_at_bottom_and_right(self) -> None:
        batch = [
            {
                "image": torch.ones(3, 4, 5),
                "labels": torch.zeros(20),
                "boxes": torch.zeros(1, 4),
                "image_id": "a.jpg",
                "image_hw": (4, 5),
            },
            {
                "image": torch.full((3, 2, 7), 2.0),
                "labels": torch.zeros(20),
                "boxes": torch.zeros(1, 4),
                "image_id": "b.jpg",
                "image_hw": (2, 7),
            },
        ]
        collated = collate_voc_batch(batch)
        self.assertEqual(tuple(collated["images"].shape), (2, 3, 4, 7))
        self.assertTrue(torch.equal(collated["images"][0, :, :, 5:], torch.zeros(3, 4, 2)))
        self.assertTrue(torch.equal(collated["images"][1, :, 2:, :], torch.zeros(3, 2, 7)))


class PaperMetricTests(unittest.TestCase):
    def test_voc2007_ap_ignores_difficult_examples(self) -> None:
        targets = torch.tensor([[1.0], [0.0], [-1.0]])
        scores = torch.tensor([[0.8], [0.2], [1.0]])
        mean_ap, aps = mean_average_precision(targets, scores, mode="voc2007")
        self.assertAlmostEqual(mean_ap, 1.0)
        self.assertAlmostEqual(float(aps[0]), 1.0)

    def test_caffe_multilabel_loss_sums_classes_then_averages_batch(self) -> None:
        logits = torch.zeros(2, 20, requires_grad=True)
        labels = torch.zeros_like(logits)
        element_loss = multilabel_bce(logits, labels, normalization="elements")
        batch_loss = multilabel_bce(logits, labels, normalization="batch")
        self.assertAlmostEqual(float(batch_loss / element_loss), 20.0, places=5)

        element_grad = torch.autograd.grad(element_loss, logits, retain_graph=True)[0]
        batch_grad = torch.autograd.grad(batch_loss, logits)[0]
        torch.testing.assert_close(batch_grad, element_grad * 20.0)

    def test_paper_multiscale_normalizes_each_view_before_mean(self) -> None:
        first = torch.tensor([[1.0, 0.0]])
        second = torch.tensor([[0.0, 16.0]])
        per_view_mean = torch.stack(
            [normalize_fv_tensor(first), normalize_fv_tensor(second)]
        ).mean(dim=0)
        post_mean_normalized = normalize_fv_tensor(torch.stack([first, second]).mean(dim=0))

        torch.testing.assert_close(per_view_mean, torch.tensor([[0.5, 0.5]]))
        self.assertFalse(torch.allclose(per_view_mean, post_mean_normalized))


class _TinyFisherModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Linear(2, 2))
        self.patch_mlp = nn.Sequential(
            nn.Linear(2, 3),
            nn.Identity(),
            nn.Identity(),
            nn.Linear(3, 4),
            nn.Identity(),
            nn.Identity(),
            nn.Linear(4, 5),
        )


class StageTransferTests(unittest.TestCase):
    def test_fc6_fc7_and_reduction_all_transfer(self) -> None:
        model = _TinyFisherModel()
        source = {
            "features.0.weight": torch.full_like(model.features[0].weight, 1.0),
            "features.0.bias": torch.full_like(model.features[0].bias, 2.0),
            "classifier.1.weight": torch.full_like(model.patch_mlp[0].weight, 3.0),
            "classifier.1.bias": torch.full_like(model.patch_mlp[0].bias, 4.0),
            "classifier.4.weight": torch.full_like(model.patch_mlp[3].weight, 5.0),
            "classifier.4.bias": torch.full_like(model.patch_mlp[3].bias, 6.0),
            "classifier.7.weight": torch.full_like(model.patch_mlp[6].weight, 7.0),
            "classifier.7.bias": torch.full_like(model.patch_mlp[6].bias, 8.0),
        }
        loaded = load_stage1_weights(model, source)
        self.assertEqual(len(loaded), 8)
        torch.testing.assert_close(model.patch_mlp[6].weight, source["classifier.7.weight"])


class CaffeOptimizerRuleTests(unittest.TestCase):
    def test_stable_vgg_preset_uses_caffe_accumulation_and_lr_decay(self) -> None:
        preset = PRESETS["vgg16-paper-stable"]
        self.assertEqual(preset["grad_accum_steps"], 8)
        self.assertEqual(preset["lr_step_iterations"], [1000, 2000])
        self.assertEqual(preset["val_every_epochs"], 1)
        self.assertEqual(preset["save_every_epochs"], 1)

    def test_bias_uses_separate_lr_and_zero_weight_decay(self) -> None:
        layer = nn.Linear(4, 3)
        groups: list[dict[str, object]] = []
        append_caffe_param_groups(
            groups,
            list(layer.named_parameters(prefix="layer")),
            weight_lr=1e-3,
            bias_lr=2e-3,
        )
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]["lr"], 1e-3)
        self.assertNotIn("weight_decay", groups[0])
        self.assertEqual(groups[1]["lr"], 2e-3)
        self.assertEqual(groups[1]["weight_decay"], 0.0)
        self.assertIs(groups[0]["params"][0], layer.weight)
        self.assertIs(groups[1]["params"][0], layer.bias)


if __name__ == "__main__":
    unittest.main()
