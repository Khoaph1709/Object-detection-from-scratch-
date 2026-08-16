from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from my_submission.utils.checkpoint import resolve_checkpoint_path


class CheckpointDownloadTest(unittest.TestCase):
    def test_uses_existing_checkpoint_without_downloading(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            script_dir = Path(temp_dir)
            checkpoint_path = script_dir / "models" / "best.pth"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_bytes(b"local-checkpoint")

            resolved = resolve_checkpoint_path("models/best.pth", script_dir=script_dir)
            self.assertEqual(resolved, checkpoint_path)

    def test_downloads_checkpoint_from_url_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            script_dir = Path(temp_dir)
            source_path = script_dir / "remote_source.pth"
            payload = b"checkpoint-payload"
            source_path.write_bytes(payload)
            expected_sha256 = hashlib.sha256(payload).hexdigest()

            resolved = resolve_checkpoint_path(
                "models/best.pth",
                script_dir=script_dir,
                checkpoint_url=source_path.as_uri(),
                checkpoint_sha256=expected_sha256,
            )
            self.assertTrue(resolved.exists())
            self.assertEqual(resolved.read_bytes(), payload)

    def test_downloads_when_checkpoint_argument_is_url(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            script_dir = Path(temp_dir)
            source_path = script_dir / "weights_v1.pth"
            source_path.write_bytes(b"weights")

            resolved = resolve_checkpoint_path(source_path.as_uri(), script_dir=script_dir)
            self.assertTrue(resolved.exists())
            self.assertEqual(resolved.name, "weights_v1.pth")


if __name__ == "__main__":
    unittest.main()
