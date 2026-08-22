import unittest

from my_submission.scripts.sweep_inference_postprocess import postprocess_predictions


class SweepInferencePostprocessTest(unittest.TestCase):
    def test_class_aware_nms_removes_same_class_duplicates(self) -> None:
        predictions = [
            {
                "image_id": "image.jpg",
                "boxes": [
                    {"class": "chair", "confidence": 0.9, "bbox": [0, 0, 100, 100]},
                    {"class": "chair", "confidence": 0.8, "bbox": [5, 5, 95, 95]},
                    {"class": "backpack", "confidence": 0.7, "bbox": [5, 5, 95, 95]},
                    {"class": "chair", "confidence": 0.04, "bbox": [200, 200, 220, 220]},
                ],
            }
        ]
        result = postprocess_predictions(
            predictions,
            score_threshold=0.05,
            nms_threshold=0.5,
            max_detections_per_image=10,
        )
        boxes = result[0]["boxes"]
        self.assertEqual(len(boxes), 2)
        self.assertEqual([box["class"] for box in boxes], ["chair", "backpack"])

    def test_global_limit_is_applied_after_class_nms(self) -> None:
        predictions = [
            {
                "image_id": "image.jpg",
                "boxes": [
                    {"class": "chair", "confidence": 0.9, "bbox": [0, 0, 20, 20]},
                    {"class": "backpack", "confidence": 0.8, "bbox": [30, 30, 50, 50]},
                    {"class": "cup", "confidence": 0.7, "bbox": [60, 60, 80, 80]},
                ],
            }
        ]
        result = postprocess_predictions(
            predictions,
            score_threshold=0.0,
            nms_threshold=0.5,
            max_detections_per_image=2,
        )
        self.assertEqual(len(result[0]["boxes"]), 2)
        self.assertEqual([box["class"] for box in result[0]["boxes"]], ["chair", "backpack"])


if __name__ == "__main__":
    unittest.main()
