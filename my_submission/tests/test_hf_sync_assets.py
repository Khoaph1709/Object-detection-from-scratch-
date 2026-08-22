import tempfile
import unittest
from pathlib import Path

from my_submission.scripts.hf_upload_assets import resolve_dataset_upload
from my_submission.scripts.hf_sync_assets import (
    checkpoint_path,
    dataset_marker,
    download_checkpoint_atomic,
)


class HuggingFaceSyncTest(unittest.TestCase):
    def _make_dataset(self, root: Path) -> Path:
        public = root / "indoor5-v2-student" / "public"
        (public / "annotations").mkdir(parents=True)
        (public / "train" / "images").mkdir(parents=True)
        (public / "val" / "images").mkdir(parents=True)
        (public / "annotations" / "train.json").write_text("{}")
        (public / "annotations" / "val.json").write_text("{}")
        return public

    def test_upload_layout_is_canonical_from_dataset_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            public = self._make_dataset(root)
            folder, prefix = resolve_dataset_upload(public.parent)
            self.assertEqual(folder, public)
            self.assertEqual(prefix, "indoor5-v2-student/public")

    def test_upload_layout_is_canonical_from_public_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            public = self._make_dataset(root)
            folder, prefix = resolve_dataset_upload(public)
            self.assertEqual(folder, public)
            self.assertEqual(prefix, "indoor5-v2-student/public")

    def test_expected_project_paths(self) -> None:
        root = Path("/srv/project")
        self.assertEqual(
            dataset_marker(root),
            root / "indoor5-v2-student/public/annotations/train.json",
        )
        self.assertEqual(
            checkpoint_path(root, "baseline", "best.pth"),
            root / "checkpoints/baseline/best.pth",
        )

    def test_checkpoint_download_is_atomic_and_preserves_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = b"valid checkpoint payload"
            calls = []

            def fake_download(**kwargs):
                calls.append(kwargs)
                output = Path(kwargs["local_dir"]) / kwargs["filename"]
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(payload)
                return str(output)

            destination = download_checkpoint_atomic(
                fake_download,
                repo_id="user/private-model",
                revision="main",
                project_root=root,
                run_name="baseline",
                filename="best.pth",
                force=False,
                token=True,
            )
            self.assertEqual(destination, root / "checkpoints/baseline/best.pth")
            self.assertEqual(destination.read_bytes(), payload)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["filename"], "checkpoints/baseline/best.pth")
            self.assertFalse(any(p.name.startswith(".hf-checkpoint-") for p in root.iterdir()))

    def test_existing_checkpoint_is_not_redownloaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            destination = checkpoint_path(root, "baseline")
            destination.parent.mkdir(parents=True)
            destination.write_bytes(b"existing")
            calls = []

            def fake_download(**kwargs):
                calls.append(kwargs)
                raise AssertionError("existing checkpoint must not be downloaded")

            resolved = download_checkpoint_atomic(
                fake_download,
                repo_id="user/private-model",
                revision="main",
                project_root=root,
                run_name="baseline",
                filename="best.pth",
                force=False,
                token=True,
            )
            self.assertEqual(resolved, destination)
            self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
