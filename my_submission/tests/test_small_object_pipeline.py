from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch
from PIL import Image

from my_submission.train import build_train_sampler
from my_submission.utils.assigner import FCOSTargetAssigner
from my_submission.utils.augmentations import DetectionTransform
from my_submission.utils.dataset import DetectionDataset
from my_submission.utils.postprocess import scale_detections_to_original


class SmallObjectCropTest(unittest.TestCase):
    def test_crop_remaps_boxes_and_inverse_scaling_restores_original_coordinates(self) -> None:
        image = Image.new("RGB", (400, 400), color=(120, 120, 120))
        target = {
            "boxes": torch.tensor([[190.0, 190.0, 210.0, 210.0], [0.0, 0.0, 400.0, 400.0]]),
            "labels": torch.tensor([4, 0], dtype=torch.long),
            "image_id": "tiny",
        }
        transform = DetectionTransform(
            train=True,
            short_size=160,
            max_size=160,
            horizontal_flip_prob=0.0,
            small_object_crop_prob=1.0,
            small_object_area_threshold=0.05,
            small_object_crop_context=0.75,
            small_object_crop_min_size=100,
        )
        image_tensor, transformed = transform(image, target)

        self.assertEqual(tuple(image_tensor.shape), (3, 160, 160))
        self.assertEqual(tuple(transformed["original_size"].tolist()), (400, 400))
        self.assertEqual(tuple(transformed["crop_offset"].tolist()), (150, 150))
        self.assertTrue(torch.all(transformed["boxes"][:, :2] >= 0))
        self.assertTrue(torch.all(transformed["boxes"][:, 2:] <= 160))
        self.assertEqual(transformed["labels"].tolist(), [4])

        restored = scale_detections_to_original(
            [{"class": "backpack", "confidence": 0.8, "bbox": transformed["boxes"][0].tolist()}],
            resized_size=tuple(int(v) for v in transformed["resized_size"].tolist()),
            original_size=tuple(int(v) for v in transformed["original_size"].tolist()),
            crop_offset=tuple(int(v) for v in transformed["crop_offset"].tolist()),
            crop_size=tuple(int(v) for v in transformed["crop_size"].tolist()),
        )
        restored_box = torch.tensor(restored[0]["bbox"])
        expected = target["boxes"][0]
        self.assertTrue(torch.allclose(restored_box, expected, atol=1.5))

    def test_crop_disabled_preserves_zero_offset(self) -> None:
        image = Image.new("RGB", (100, 80), color=(0, 0, 0))
        target = {
            "boxes": torch.tensor([[10.0, 10.0, 30.0, 30.0]]),
            "labels": torch.tensor([0], dtype=torch.long),
            "image_id": "image",
        }
        transform = DetectionTransform(train=False, short_size=80, max_size=80)
        _, transformed = transform(image, target)
        self.assertEqual(tuple(transformed["crop_offset"].tolist()), (0, 0))


class SmallObjectAssignmentTest(unittest.TestCase):
    def test_range_overlap_expands_high_resolution_levels(self) -> None:
        strides = {"p2": 4, "p3": 8, "p4": 16, "p5": 32, "p6": 64, "p7": 128}
        assigner = FCOSTargetAssigner(strides, center_sampling_radius=2.0, range_overlap=8.0)
        self.assertEqual(assigner.regression_ranges["p2"], (0.0, 40.0))
        self.assertEqual(assigner.regression_ranges["p3"], (24.0, 72.0))
        self.assertEqual(assigner.regression_ranges["p4"], (56.0, 136.0))


class SmallObjectSamplerTest(unittest.TestCase):
    def test_sampler_prioritizes_images_containing_small_objects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotation_path = root / "annotations.json"
            annotation_path.write_text(
                json.dumps(
                    {
                        "classes": ["bottle", "cup", "chair", "laptop", "backpack"],
                        "images": [
                            {"id": "small.jpg", "file_name": "small.jpg", "width": 100, "height": 100},
                            {"id": "large.jpg", "file_name": "large.jpg", "width": 100, "height": 100},
                        ],
                        "annotations": [
                            {"image_id": "small.jpg", "class": "backpack", "bbox": [0, 0, 10, 10]},
                            {"image_id": "large.jpg", "class": "laptop", "bbox": [0, 0, 80, 80]},
                        ],
                    }
                )
            )
            dataset = DetectionDataset(annotation_path, root)
            args = Namespace(
                hard_negative_sampling=False,
                mined_sampler_weights="",
                small_object_sampling=True,
                small_object_sampling_area_threshold=0.05,
                small_object_image_weight=1.5,
                empty_image_weight=1.0,
                chair_positive_image_weight=1.0,
                backpack_positive_image_weight=1.0,
            )
            sampler = build_train_sampler(dataset, args)
            self.assertIsNotNone(sampler)
            self.assertGreater(float(sampler.weights[0]), float(sampler.weights[1]))
            self.assertAlmostEqual(float(sampler.weights[0]), 1.5, places=6)
            self.assertAlmostEqual(float(sampler.weights[1]), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
