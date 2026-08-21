from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from my_submission.models.detector import build_detector
from my_submission.train import build_optimizer
from my_submission.models.highres import P1Refinement
from my_submission.scripts.ensemble_predictions import ensemble_prediction_files, weighted_boxes_fusion
from my_submission.utils.augmentations import DetectionTransform
from my_submission.utils.tiling import TiledImageDataset, generate_tile_specs


class TilingContractTest(unittest.TestCase):
    def test_tile_specs_cover_edges_without_out_of_bounds(self) -> None:
        specs = generate_tile_specs(1000, 1300, tile_size=640, overlap=0.20)
        self.assertGreater(len(specs), 1)
        self.assertTrue(any(spec.top + spec.height == 1000 for spec in specs))
        self.assertTrue(any(spec.left + spec.width == 1300 for spec in specs))
        for spec in specs:
            self.assertGreater(spec.height, 0)
            self.assertGreater(spec.width, 0)
            self.assertLessEqual(spec.top + spec.height, 1000)
            self.assertLessEqual(spec.left + spec.width, 1300)

    def test_tiled_dataset_preserves_original_coordinate_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (1300, 1000), color=(200, 200, 200)).save(root / "sample.jpg")
            transform = DetectionTransform(train=False, short_size=320, max_size=320)
            dataset = TiledImageDataset(root, transform, tile_size=640, tile_overlap=0.20)
            _, target = dataset[0]
            self.assertEqual(tuple(target["original_size"].tolist()), (1000, 1300))
            self.assertEqual(len(target["crop_offset"]), 2)
            self.assertEqual(len(target["crop_size"]), 2)
            self.assertEqual(len(target["tile_spec"]), 4)


class TileTrainTransformTest(unittest.TestCase):
    def test_tile_training_keeps_selected_object_and_records_offset(self) -> None:
        image = Image.new("RGB", (1000, 800), color=(100, 100, 100))
        target = {
            "boxes": torch.tensor([[390.0, 290.0, 410.0, 310.0]]),
            "labels": torch.tensor([4], dtype=torch.long),
            "image_id": "tiny",
        }
        transform = DetectionTransform(
            train=True,
            short_size=320,
            max_size=320,
            horizontal_flip_prob=0.0,
            tile_train_prob=1.0,
            tile_train_size=320,
            tile_train_area_threshold=0.05,
            tile_train_min_visible_fraction=0.70,
            small_object_crop_prob=1.0,
        )
        _, transformed = transform(image, target)
        self.assertEqual(transformed["labels"].tolist(), [4])
        self.assertTrue(torch.all(transformed["boxes"][:, 2:] > transformed["boxes"][:, :2]))
        self.assertNotEqual(tuple(transformed["crop_size"].tolist()), (800, 1000))


class P1ContractTest(unittest.TestCase):
    def test_detector_p1_forward_has_seven_levels_and_valid_head_shapes(self) -> None:
        model = build_detector(
            num_classes=5,
            pretrained_backbone=False,
            backbone_name="convnext_tiny",
            fpn_type="bifpn",
            bifpn_layers=1,
            use_p1=True,
        )
        model.eval()
        with torch.no_grad():
            outputs = model(torch.zeros(1, 3, 128, 128))
        self.assertEqual(list(outputs["features"]), ["p1", "p2", "p3", "p4", "p5", "p6", "p7"])
        self.assertEqual(model.strides["p1"], 2)
        for level in outputs["features"]:
            self.assertEqual(outputs["cls_logits"][level].shape[1], 5)
            self.assertEqual(outputs["bbox_regression"][level].shape[1], 4)
            self.assertEqual(outputs["centerness"][level].shape[1], 1)

    def test_p1_refinement_doubles_spatial_resolution(self) -> None:
        module = P1Refinement(32)
        output = module(torch.zeros(1, 32, 8, 10))
        self.assertEqual(tuple(output.shape), (1, 32, 16, 20))

    def test_p1_head_is_in_optimizer_with_differential_learning_rates(self) -> None:
        model = build_detector(
            num_classes=5,
            pretrained_backbone=False,
            backbone_name="convnext_tiny",
            fpn_type="bifpn",
            bifpn_layers=1,
            use_p1=True,
        )
        args = type("OptimizerArgs", (), {
            "backbone_lr": 5e-7,
            "head_lr": 5e-6,
            "lr": 5e-6,
            "weight_decay": 0.05,
        })()
        optimizer = build_optimizer(model, args)
        optimized_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        self.assertTrue(all(id(parameter) in optimized_ids for parameter in model.p1_head.parameters()))
        self.assertEqual(len(optimized_ids), len({id(parameter) for parameter in model.parameters()}))


class WBFTest(unittest.TestCase):
    def test_wbf_is_class_aware(self) -> None:
        fused = weighted_boxes_fusion(
            [
                {"class": "backpack", "confidence": 0.9, "bbox": [0, 0, 10, 10]},
                {"class": "backpack", "confidence": 0.8, "bbox": [1, 1, 11, 11]},
                {"class": "chair", "confidence": 0.95, "bbox": [0, 0, 10, 10]},
            ],
            iou_threshold=0.5,
        )
        self.assertEqual(len(fused), 2)
        self.assertEqual({item["class"] for item in fused}, {"backpack", "chair"})

    def test_ensemble_script_preserves_evaluator_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.json"
            second = root / "second.json"
            output = root / "out.json"
            payload = [{"image_id": "image.jpg", "boxes": [{"class": "backpack", "confidence": 0.8, "bbox": [0, 0, 10, 10]}]}]
            first.write_text(json.dumps(payload))
            second.write_text(json.dumps(payload))
            ensemble_prediction_files([first, second], output)
            result = json.loads(output.read_text())
            self.assertEqual(result[0]["image_id"], "image.jpg")
            self.assertEqual(set(result[0]["boxes"][0]), {"class", "confidence", "bbox"})


if __name__ == "__main__":
    unittest.main()
