from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen


def is_remote_checkpoint(path: str) -> bool:
    scheme = urlparse(path).scheme.lower()
    return scheme in {"http", "https", "file"}


def resolve_checkpoint_path(
    checkpoint: str,
    *,
    script_dir: Path,
    checkpoint_url: str = "",
    checkpoint_sha256: str = "",
    download_dir: Path | None = None,
) -> Path:
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.exists():
        return checkpoint_path

    fallback = script_dir / checkpoint_path
    if fallback.exists():
        return fallback

    source_url = checkpoint_url.strip()
    if not source_url and is_remote_checkpoint(checkpoint):
        source_url = checkpoint
    if not source_url:
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint}'. Provide a local file or set --checkpoint_url "
            "to download the checkpoint."
        )

    if is_remote_checkpoint(checkpoint):
        destination = (download_dir or script_dir / "models" / "downloads") / _infer_filename(source_url)
    else:
        destination = fallback if not checkpoint_path.is_absolute() else checkpoint_path
    return download_checkpoint(source_url, destination, checkpoint_sha256=checkpoint_sha256)


def download_checkpoint(url: str, destination: Path, *, checkpoint_sha256: str = "") -> Path:
    normalized_sha256 = checkpoint_sha256.strip().lower()
    if destination.exists():
        if not normalized_sha256 or _sha256(destination) == normalized_sha256:
            return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    hasher = hashlib.sha256() if normalized_sha256 else None

    with urlopen(url) as response, temp_path.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            if hasher:
                hasher.update(chunk)

    if hasher and hasher.hexdigest() != normalized_sha256:
        temp_path.unlink(missing_ok=True)
        raise ValueError(f"Downloaded checkpoint SHA256 mismatch for {destination}")

    temp_path.replace(destination)
    return destination


def _infer_filename(url: str) -> str:
    filename = Path(urlparse(url).path).name
    return filename or "model.pth"


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()
