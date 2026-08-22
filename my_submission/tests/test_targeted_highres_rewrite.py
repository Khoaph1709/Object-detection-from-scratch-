from collections import OrderedDict
from types import SimpleNamespace
import unittest

import torch

from my_submission.models.detector import build_detector
from my_submission.models.targeted_highres import SmallObjectResidualHead, TargetedHighResNeck
from my_submission.train import build_optimizer


class TargetedHighResRewriteTest(unittest.TestCase):
    def test_forward_keeps_fcos_levels_and_shapes(self) -> None:
        model = build_detector(
            num_classes=5,
            pretrained_backbone=False,
            backbone_name="convnext_tiny",
            fpn_type="bifpn",
            bifpn_layers=1,
            targeted_highres=True,
        )
        model.eval()
        with torch.no_grad():
            outputs = model(torch.zeros(1, 3, 128, 128))
        self.assertEqual(list(outputs["features"]), ["p2", "p3", "p4", "p5", "p6", "p7"])
        self.assertEqual(model.strides, {"p2": 4, "p3": 8, "p4": 16, "p5": 32, "p6": 64, "p7": 128})
        for level in outputs["features"]:
            self.assertEqual(outputs["cls_logits"][level].shape[-2:], outputs["features"][level].shape[-2:])
            self.assertEqual(outputs["bbox_regression"][level].shape[-2:], outputs["features"][level].shape[-2:])
            self.assertEqual(outputs["centerness"][level].shape[-2:], outputs["features"][level].shape[-2:])
            self.assertTrue(torch.isfinite(outputs["cls_logits"][level]).all())

    def test_new_branches_start_with_zero_residual_outputs(self) -> None:
        neck = TargetedHighResNeck(32)
        features = OrderedDict(
            {
                "p2": torch.randn(1, 32, 16, 16),
                "p3": torch.randn(1, 32, 8, 8),
            }
        )
        refined = neck(features)
        self.assertTrue(torch.allclose(refined["p2"], features["p2"], atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(refined["p3"], features["p3"], atol=1e-6, rtol=1e-6))

        head = SmallObjectResidualHead(32, num_classes=5, num_convs=2)
        residuals = head(features)
        for key in ("cls_logits", "bbox_regression", "centerness"):
            for value in residuals[key].values():
                self.assertTrue(torch.allclose(value, torch.zeros_like(value), atol=1e-6, rtol=1e-6))

    def test_targeted_modules_are_in_optimizer(self) -> None:
        model = build_detector(
            num_classes=5,
            pretrained_backbone=False,
            backbone_name="convnext_tiny",
            fpn_type="bifpn",
            bifpn_layers=1,
            targeted_highres=True,
        )
        args = SimpleNamespace(
            backbone_lr=1e-6,
            head_lr=8e-6,
            lr=8e-6,
            weight_decay=0.05,
        )
        optimizer = build_optimizer(model, args)
        optimized_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        for module in (model.highres_neck, model.small_object_head):
            self.assertTrue(all(id(parameter) in optimized_ids for parameter in module.parameters()))

    def test_baseline_state_dict_warm_starts_targeted_model_safely(self) -> None:
        baseline = build_detector(
            num_classes=5,
            pretrained_backbone=False,
            backbone_name="convnext_tiny",
            fpn_type="bifpn",
            bifpn_layers=1,
            targeted_highres=False,
        )
        targeted = build_detector(
            num_classes=5,
            pretrained_backbone=False,
            backbone_name="convnext_tiny",
            fpn_type="bifpn",
            bifpn_layers=1,
            targeted_highres=True,
        )
        missing, unexpected = targeted.load_state_dict(baseline.state_dict(), strict=False)
        self.assertEqual(unexpected, [])
        self.assertTrue(missing)
        self.assertTrue(
            all(key.startswith(("highres_neck.", "small_object_head.")) for key in missing)
        )

    def test_p1_and_targeted_highres_are_mutually_exclusive(self) -> None:
        with self.assertRaises(ValueError):
            build_detector(
                num_classes=5,
                pretrained_backbone=False,
                backbone_name="convnext_tiny",
                fpn_type="bifpn",
                bifpn_layers=1,
                use_p1=True,
                targeted_highres=True,
            )


if __name__ == "__main__":
    unittest.main()
