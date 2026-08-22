import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from my_submission.scripts.audit_annotations import audit, parse_args


class DataQualityAuditTest(unittest.TestCase):
    def test_audit_is_read_only_and_reports_geometry_overlap_and_review_groups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (100, 100), color=(128, 128, 128)).save(image_dir / "a.jpg")
            Image.new("RGB", (100, 100), color=(128, 128, 128)).save(image_dir / "b.jpg")
            annotation_path = root / "train.json"
            annotation_payload = {
                "classes": ["bottle", "cup", "chair", "laptop", "backpack"],
                "images": [
                    {"id": "a.jpg", "file_name": "a.jpg", "width": 100, "height": 100},
                    {"id": "b.jpg", "file_name": "b.jpg", "width": 100, "height": 100},
                ],
                "annotations": [
                    {"image_id": "a.jpg", "class": "chair", "bbox": [10, 10, 50, 50]},
                    {"image_id": "a.jpg", "class": "chair", "bbox": [10, 10, 50, 50]},
                    {"image_id": "a.jpg", "class": "backpack", "bbox": [10, 10, 50, 50]},
                    {"image_id": "a.jpg", "class": "bottle", "bbox": [-1, 5, 1, 5]},
                ],
            }
            annotation_path.write_text(json.dumps(annotation_payload), encoding="utf-8")
            predictions_path = root / "predictions.json"
            predictions_path.write_text(
                json.dumps(
                    [
                        {
                            "image_id": "a.jpg",
                            "boxes": [
                                {"class": "chair", "confidence": 0.95, "bbox": [60, 60, 90, 90]},
                                {"class": "backpack", "confidence": 0.90, "bbox": [55, 55, 90, 90]},
                            ],
                        },
                        {"image_id": "b.jpg", "boxes": []},
                    ]
                ),
                encoding="utf-8",
            )
            output_dir = root / "audit"
            args = type(
                "AuditArgs",
                (),
                {
                    "annotations": str(annotation_path),
                    "image_dir": str(image_dir),
                    "predictions": str(predictions_path),
                    "output_dir": str(output_dir),
                    "review_count": 50,
                    "overlap_review_count": 100,
                    "tiny_area_ratio": 0.05,
                    "duplicate_iou": 0.90,
                    "overlap_iou": 0.90,
                    "prediction_match_iou": 0.50,
                    "seed": 42,
                    "no_render": True,
                },
            )()
            before = annotation_path.read_bytes()
            summary = audit(args)
            self.assertEqual(annotation_path.read_bytes(), before)
            self.assertTrue(summary["class_order_matches"])
            self.assertEqual(summary["images"], 2)
            self.assertEqual(summary["annotations"], 4)
            self.assertGreaterEqual(summary["issue_counts"]["duplicate_box"], 1)
            self.assertGreaterEqual(summary["issue_counts"]["cross_class_overlap"], 1)
            self.assertGreaterEqual(summary["issue_counts"]["out_of_bounds"], 1)
            manifest = json.loads((output_dir / "review_manifest.json").read_text(encoding="utf-8"))
            self.assertIn("chair_false_positive_high", manifest["groups"])
            self.assertIn("backpack_false_positive_high", manifest["groups"])
            self.assertIn("b.jpg", manifest["groups"]["no_chair_or_backpack"])


if __name__ == "__main__":
    unittest.main()
