from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from my_submission.utils.dataset import DetectionDataset


class DatasetClassMappingTest(unittest.TestCase):
    def test_uses_classes_from_annotation_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            image_dir.mkdir(parents=True, exist_ok=True)
            image_path = image_dir / "sample.jpg"
            Image.new("RGB", (64, 64), color=(0, 0, 0)).save(image_path)

            annotation_path = root / "train.json"
            annotation_path.write_text(
                json.dumps(
                    {
                        "classes": ["bottle", "cup", "chair", "laptop", "backpack"],
                        "images": [
                            {
                                "id": "sample.jpg",
                                "file_name": "train/images/sample.jpg",
                                "width": 64,
                                "height": 64,
                            }
                        ],
                        "annotations": [
                            {"image_id": "sample.jpg", "class": "laptop", "bbox": [1, 2, 10, 20]}
                        ],
                    }
                ),
                encoding="utf-8",
            )

            dataset = DetectionDataset(annotation_path, image_dir, transform=None)
            _, target = dataset[0]
            self.assertEqual(target["labels"].tolist(), [3])


if __name__ == "__main__":
    unittest.main()
