from __future__ import annotations

import unittest

from my_submission.utils.postprocess import merge_detections_consensus


class ConsensusMergeTest(unittest.TestCase):
    def test_fuses_matched_boxes_and_penalizes_single_view_boxes(self) -> None:
        primary = [
            {"class": "chair", "confidence": 0.8, "bbox": [10.0, 10.0, 50.0, 50.0]},
            {"class": "chair", "confidence": 0.5, "bbox": [120.0, 10.0, 160.0, 50.0]},
        ]
        secondary = [
            {"class": "chair", "confidence": 0.6, "bbox": [12.0, 10.0, 52.0, 50.0]},
        ]

        merged = merge_detections_consensus(
            primary,
            secondary,
            image_size=(100, 200),
            match_iou_threshold=0.6,
            nms_threshold=0.55,
            max_detections_per_image=10,
            single_view_score_factor=0.5,
            class_score_thresholds={"chair": 0.2},
        )

        self.assertEqual(len(merged), 2)
        self.assertAlmostEqual(merged[0]["confidence"], 0.7, places=5)
        self.assertAlmostEqual(merged[0]["bbox"][0], 10.857142, places=5)
        self.assertAlmostEqual(merged[1]["confidence"], 0.25, places=5)

    def test_filters_class_specific_single_view_noise(self) -> None:
        primary = [
            {"class": "chair", "confidence": 0.3, "bbox": [10.0, 10.0, 50.0, 50.0]},
            {"class": "dog", "confidence": 0.3, "bbox": [60.0, 10.0, 90.0, 50.0]},
        ]

        merged = merge_detections_consensus(
            primary,
            [],
            image_size=(100, 100),
            single_view_score_factor=0.5,
            class_score_thresholds={"chair": 0.2},
        )

        classes = [det["class"] for det in merged]
        self.assertEqual(classes, ["dog"])


if __name__ == "__main__":
    unittest.main()
