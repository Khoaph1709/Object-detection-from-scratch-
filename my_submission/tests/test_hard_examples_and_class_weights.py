from __future__ import annotations

import unittest
from argparse import Namespace
from collections import OrderedDict

import torch

from my_submission.scripts.mine_hard_examples import mine_hard_examples
from my_submission.scripts.tune_predictions import filter_predictions
from my_submission.train import get_quality_blend_weight
from my_submission.utils.losses import FCOSLoss


class HardExampleMiningTest(unittest.TestCase):
    def test_mines_chair_and_backpack_and_combines_image_weights(self) -> None:
        ground_truth = {
            "images": [{"id": "chair-image"}, {"id": "backpack-image"}, {"id": "empty-image"}],
            "annotations": [
                {"image_id": "chair-image", "class": "chair", "bbox": [10.0, 10.0, 50.0, 50.0]},
                {"image_id": "backpack-image", "class": "backpack", "bbox": [20.0, 20.0, 60.0, 60.0]},
            ],
        }
        predictions = [
            {
                "image_id": "chair-image",
                "boxes": [
                    {"class": "chair", "confidence": 0.9, "bbox": [100.0, 100.0, 120.0, 120.0]}
                ],
            },
            {"image_id": "backpack-image", "boxes": []},
            {
                "image_id": "empty-image",
                "boxes": [
                    {"class": "backpack", "confidence": 0.8, "bbox": [0.0, 0.0, 20.0, 20.0]}
                ],
            },
        ]

        result = mine_hard_examples(
            ground_truth,
            predictions,
            classes=["chair", "backpack"],
            class_score_thresholds={"chair": 0.2, "backpack": 0.2},
            class_topk={"chair": 0, "backpack": 0},
        )

        hard_negative_classes = {item["class"] for item in result["hard_negatives"]}
        hard_positive_classes = {item["class"] for item in result["hard_positives"]}
        self.assertIn("chair", hard_negative_classes)
        self.assertIn("backpack", hard_negative_classes)
        self.assertIn("backpack", hard_positive_classes)
        self.assertGreater(result["image_weights"]["empty-image"], 1.0)
        self.assertIn("chair", result["summary"]["classes"])
        self.assertIn("backpack", result["summary"]["classes"])


class PerClassFilteringTest(unittest.TestCase):
    def test_applies_class_specific_threshold_and_limit(self) -> None:
        predictions = [
            {
                "image_id": "image-1",
                "boxes": [
                    {"class": "chair", "confidence": 0.20, "bbox": [0, 0, 10, 10]},
                    {"class": "chair", "confidence": 0.10, "bbox": [10, 10, 20, 20]},
                    {"class": "backpack", "confidence": 0.30, "bbox": [20, 20, 30, 30]},
                    {"class": "backpack", "confidence": 0.25, "bbox": [30, 30, 40, 40]},
                ],
            }
        ]
        filtered = filter_predictions(
            predictions,
            threshold=0.05,
            limit=100,
            class_thresholds={"chair": 0.15},
            class_limits={"backpack": 1},
        )
        self.assertEqual([box["class"] for box in filtered[0]["boxes"]], ["backpack", "chair"])
        self.assertEqual(filtered[0]["boxes"][0]["confidence"], 0.30)


class BackpackLossWeightTest(unittest.TestCase):
    def _make_outputs_and_targets(self):
        features = OrderedDict(P2=torch.zeros(1, 8, 1, 1))
        outputs = {
            "features": features,
            "cls_logits": OrderedDict(P2=torch.zeros(1, 5, 1, 1)),
            "bbox_regression": OrderedDict(P2=torch.zeros(1, 4, 1, 1)),
            "centerness": OrderedDict(P2=torch.zeros(1, 1, 1, 1)),
        }
        targets = {
            "labels": torch.tensor([[4]], dtype=torch.long),
            "reg_targets": torch.ones(1, 1, 4),
            "centerness": torch.ones(1, 1),
            "locations": torch.tensor([[2.0, 2.0]]),
        }
        return outputs, targets

    def test_backpack_positive_weight_changes_classification_loss(self) -> None:
        outputs, targets = self._make_outputs_and_targets()
        base = FCOSLoss(backpack_positive_weight=1.0)(outputs, targets, {"P2": 4})
        weighted = FCOSLoss(backpack_positive_weight=2.0)(outputs, targets, {"P2": 4})
        self.assertTrue(torch.isfinite(weighted["loss"]))
        self.assertGreater(float(weighted["loss_cls"]), float(base["loss_cls"]))


class BoxLossAblationTest(unittest.TestCase):
    def _make_outputs_and_targets(self):
        features = OrderedDict(P2=torch.zeros(1, 8, 1, 1))
        outputs = {
            "features": features,
            "cls_logits": OrderedDict(P2=torch.zeros(1, 5, 1, 1)),
            "bbox_regression": OrderedDict(P2=torch.zeros(1, 4, 1, 1)),
            "centerness": OrderedDict(P2=torch.zeros(1, 1, 1, 1)),
        }
        targets = {
            "labels": torch.tensor([[2]], dtype=torch.long),
            "reg_targets": torch.tensor([[[1.0, 1.0, 3.0, 3.0]]]),
            "centerness": torch.tensor([[1.0]], dtype=torch.float32),
            "locations": torch.tensor([[2.0, 2.0]]),
        }
        return outputs, targets

    def test_diou_box_loss_is_finite_and_differs_from_giou(self) -> None:
        outputs, targets = self._make_outputs_and_targets()
        giou = FCOSLoss(box_loss_type="giou")(outputs, targets, {"P2": 4})
        diou = FCOSLoss(box_loss_type="diou")(outputs, targets, {"P2": 4})
        self.assertTrue(torch.isfinite(diou["loss_box"]))
        self.assertNotAlmostEqual(float(giou["loss_box"]), float(diou["loss_box"]), places=6)

    def test_invalid_box_loss_type_fails_fast(self) -> None:
        with self.assertRaises(ValueError):
            FCOSLoss(box_loss_type="ciou")


class QualityAwareLossTest(unittest.TestCase):
    def _make_outputs_and_targets(self):
        features = OrderedDict(P2=torch.zeros(1, 8, 1, 1))
        outputs = {
            "features": features,
            "cls_logits": OrderedDict(P2=torch.zeros(1, 5, 1, 1)),
            "bbox_regression": OrderedDict(P2=torch.zeros(1, 4, 1, 1)),
            "centerness": OrderedDict(P2=torch.zeros(1, 1, 1, 1)),
        }
        targets = {
            "labels": torch.tensor([[2]], dtype=torch.long),
            "reg_targets": torch.ones(1, 1, 4),
            "centerness": torch.tensor([[0.5]], dtype=torch.float32),
            "locations": torch.tensor([[2.0, 2.0]]),
        }
        return outputs, targets

    def test_quality_aware_loss_is_finite_and_reports_quality_target(self) -> None:
        outputs, targets = self._make_outputs_and_targets()
        criterion = FCOSLoss(quality_aware_cls=True, quality_target_floor=0.20)
        losses = criterion(outputs, targets, {"P2": 4})
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertIn("quality_target_mean", losses)
        quality = float(losses["quality_target_mean"])
        self.assertGreaterEqual(quality, 0.20)
        self.assertLessEqual(quality, 1.0)

    def test_quality_aware_mode_differs_from_binary_focal_mode(self) -> None:
        outputs, targets = self._make_outputs_and_targets()
        binary = FCOSLoss()(outputs, targets, {"P2": 4})
        quality = FCOSLoss(quality_aware_cls=True)(outputs, targets, {"P2": 4})
        self.assertTrue(torch.isfinite(quality["loss"]))
        self.assertNotEqual(float(binary["loss_cls"]), float(quality["loss_cls"]))

    def test_blended_loss_interpolates_binary_and_quality_losses(self) -> None:
        outputs, targets = self._make_outputs_and_targets()
        binary = FCOSLoss()(outputs, targets, {"P2": 4})
        quality = FCOSLoss(quality_aware_cls=True)(outputs, targets, {"P2": 4})
        blended = FCOSLoss(quality_aware_cls=True)(
            outputs,
            targets,
            {"P2": 4},
            quality_blend_weight=0.25,
        )
        expected = 0.75 * float(binary["loss_cls"]) + 0.25 * float(quality["loss_cls"])
        self.assertAlmostEqual(float(blended["quality_blend_weight"]), 0.25, places=6)
        self.assertAlmostEqual(float(blended["loss_cls"]), expected, places=5)

    def test_quality_blend_ramp_schedule(self) -> None:
        args = Namespace(
            quality_aware_cls=True,
            quality_blend_start=0.25,
            quality_blend_ramp_epochs=3,
        )
        self.assertAlmostEqual(get_quality_blend_weight(args, 1), 0.25)
        self.assertAlmostEqual(get_quality_blend_weight(args, 2), 0.50)
        self.assertAlmostEqual(get_quality_blend_weight(args, 3), 0.75)
        self.assertAlmostEqual(get_quality_blend_weight(args, 4), 1.00)


if __name__ == "__main__":
    unittest.main()
