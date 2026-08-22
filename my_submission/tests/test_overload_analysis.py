import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from my_submission.scripts.analyze_overload_images import analyze


class OverloadAnalysisTest(unittest.TestCase):
    def test_classifies_duplicate_background_and_confusion_without_editing_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (100, 100), color=(100, 100, 100)).save(image_dir / "a.jpg")
            annotations = {
                "classes": ["bottle", "cup", "chair", "laptop", "backpack"],
                "images": [{"id": "a.jpg", "file_name": "a.jpg", "width": 100, "height": 100}],
                "annotations": [
                    {"image_id": "a.jpg", "class": "chair", "bbox": [10, 10, 40, 40]},
                ],
            }
            predictions = [
                {
                    "image_id": "a.jpg",
                    "boxes": [
                        {"class": "chair", "confidence": 0.95, "bbox": [10, 10, 40, 40]},
                        {"class": "chair", "confidence": 0.90, "bbox": [11, 11, 39, 39]},
                        {"class": "backpack", "confidence": 0.80, "bbox": [10, 10, 40, 40]},
                        {"class": "cup", "confidence": 0.70, "bbox": [60, 60, 90, 90]},
                    ],
                }
            ]
            annotation_path = root / "train.json"
            prediction_path = root / "predictions.json"
            annotation_path.write_text(json.dumps(annotations), encoding="utf-8")
            prediction_path.write_text(json.dumps(predictions), encoding="utf-8")
            output_dir = root / "analysis"
            args = type(
                "Args",
                (),
                {
                    "annotations": str(annotation_path),
                    "predictions": str(prediction_path),
                    "image_dir": str(image_dir),
                    "output_dir": str(output_dir),
                    "topk": 30,
                    "score_threshold": 0.0,
                    "match_iou": 0.50,
                    "duplicate_iou": 0.45,
                    "high_score": 0.20,
                },
            )()
            before_annotations = annotation_path.read_bytes()
            summary = analyze(args)
            self.assertEqual(annotation_path.read_bytes(), before_annotations)
            self.assertEqual(summary["predictions_analyzed"], 4)
            self.assertEqual(summary["true_positive_count"], 1)
            self.assertEqual(summary["duplicate_count"], 1)
            self.assertEqual(summary["class_confusion_count"], 1)
            self.assertEqual(summary["background_fp_count"], 1)
            self.assertEqual(summary["overload_images"], ["a.jpg"])
            self.assertTrue((output_dir / "overload_top30" / "a.jpg").exists())


if __name__ == "__main__":
    unittest.main()
