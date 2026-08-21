from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from my_submission.scripts.mine_overconfident_samples import mine_overconfident_samples


class OverconfidenceMiningTest(unittest.TestCase):
    def test_mines_error_types_and_renders_top_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            image_dir.mkdir()
            for name in ["image-1.jpg", "image-2.jpg"]:
                Image.new("RGB", (160, 120), color=(80, 80, 80)).save(image_dir / name)

            ground_truth = {
                "classes": ["chair", "backpack"],
                "images": [
                    {"id": "image-1.jpg", "file_name": "image-1.jpg", "width": 160, "height": 120},
                    {"id": "image-2.jpg", "file_name": "image-2.jpg", "width": 160, "height": 120},
                ],
                "annotations": [
                    {"image_id": "image-1.jpg", "class": "chair", "bbox": [10, 10, 50, 50]},
                    {"image_id": "image-2.jpg", "class": "chair", "bbox": [10, 10, 50, 50]},
                ],
            }
            predictions = [
                {
                    "image_id": "image-1.jpg",
                    "boxes": [
                        {"class": "chair", "confidence": 0.95, "bbox": [90, 70, 140, 110]},
                        {"class": "backpack", "confidence": 0.90, "bbox": [20, 70, 70, 110]},
                    ],
                },
                {
                    "image_id": "image-2.jpg",
                    "boxes": [
                        {"class": "chair", "confidence": 0.95, "bbox": [10, 10, 50, 50]},
                        {"class": "chair", "confidence": 0.85, "bbox": [10, 10, 50, 50]},
                    ],
                },
            ]
            output_dir = root / "output"
            report = mine_overconfident_samples(
                ground_truth,
                predictions,
                image_dir=image_dir,
                output_dir=output_dir,
                score_threshold=0.30,
                match_iou_threshold=0.50,
                max_images=2,
                max_errors_per_image=8,
            )

            self.assertEqual(report["summary"]["num_overconfident_error_predictions"], 3)
            self.assertEqual(report["summary"]["reason_counts"]["localization_error"], 1)
            self.assertEqual(report["summary"]["reason_counts"]["background_false_positive"], 1)
            self.assertEqual(report["summary"]["reason_counts"]["duplicate_prediction"], 1)
            self.assertEqual(report["summary"]["num_rendered_images"], 2)
            self.assertTrue((output_dir / "overconfidence_report.json").exists())
            self.assertTrue((output_dir / "overconfidence_errors.csv").exists())
            visualizations = list((output_dir / "visualizations").glob("*.jpg"))
            self.assertEqual(len(visualizations), 2)

            saved = json.loads((output_dir / "summary.json").read_text())
            self.assertEqual(saved["num_selected_images"], 2)


if __name__ == "__main__":
    unittest.main()
