import json
import tempfile
import unittest
from pathlib import Path

from my_submission.scripts.merge_train_val_annotations import merge_splits


class MergeTrainValTest(unittest.TestCase):
    def _write_split(self, path: Path, split: str, image_id: str, class_name: str) -> None:
        image = {
            "id": image_id,
            "file_name": f"{split}/images/{image_id}",
            "width": 10,
            "height": 10,
        }
        payload = {
            "classes": ["bottle", "cup", "chair", "laptop", "backpack"],
            "images": [image],
            "annotations": [
                {"image_id": image_id, "class": class_name, "bbox": [1, 1, 4, 4]}
            ],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_merges_splits_without_rewriting_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train.json"
            valid = root / "val.json"
            output = root / "public" / "annotations" / "train_val_merged.json"
            self._write_split(train, "train", "train-image.jpg", "bottle")
            self._write_split(valid, "val", "val-image.jpg", "backpack")
            result = merge_splits(train, valid, output)
            self.assertEqual(len(result["images"]), 2)
            self.assertEqual(len(result["annotations"]), 2)
            self.assertEqual(result["images"][0]["file_name"], "train/images/train-image.jpg")
            self.assertEqual(result["images"][1]["file_name"], "val/images/val-image.jpg")
            self.assertEqual(result["classes"][2], "chair")
            self.assertTrue(output.is_file())

    def test_rejects_duplicate_image_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train.json"
            valid = root / "val.json"
            output = root / "merged.json"
            self._write_split(train, "train", "same.jpg", "bottle")
            self._write_split(valid, "val", "same.jpg", "backpack")
            with self.assertRaisesRegex(ValueError, "Duplicate image ID"):
                merge_splits(train, valid, output)


if __name__ == "__main__":
    unittest.main()
