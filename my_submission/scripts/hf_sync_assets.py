#!/usr/bin/env python3
"""Sync project data/checkpoints from Hugging Face Hub into the local project.

The script never deletes local data. Authentication is read from the local
Hugging Face login/configuration or HF_TOKEN; tokens are not command arguments
and are never printed.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path


def dataset_marker(project_root: Path) -> Path:
    return project_root / "indoor5-v2-student" / "public" / "annotations" / "train.json"


def checkpoint_path(project_root: Path, run_name: str, filename: str = "best.pth") -> Path:
    return project_root / "checkpoints" / run_name / filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dataset-repo", required=True, help="HF dataset repo, e.g. user/indoor5-v2-student")
    parser.add_argument("--model-repo", required=True, help="HF model repo, e.g. user/indoor5-targeted-model")
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--model-run-name", required=True)
    parser.add_argument("--download-last", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def download_checkpoint_atomic(
    hf_hub_download,
    *,
    repo_id: str,
    revision: str,
    project_root: Path,
    run_name: str,
    filename: str,
    force: bool,
    token: bool,
) -> Path:
    destination = checkpoint_path(project_root, run_name, filename)
    if destination.is_file() and not force:
        print(f"Model already present: {destination}")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    # Download into a sibling temporary directory and replace only after the
    # Hub client reports success, so a partial file cannot be used for resume.
    with tempfile.TemporaryDirectory(prefix=".hf-checkpoint-", dir=project_root) as temp_dir:
        temp_root = Path(temp_dir)
        print(f"Downloading {filename} from {repo_id}@{revision} ...")
        downloaded = Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type="model",
                revision=revision,
                filename=f"checkpoints/{run_name}/{filename}",
                local_dir=temp_root,
                token=token,
                force_download=force,
            )
        )
        if not downloaded.is_file() or downloaded.stat().st_size == 0:
            raise RuntimeError(f"Hub download returned an invalid checkpoint: {downloaded}")
        downloaded.replace(destination)
    print(f"Model ready: {destination}")
    return destination


def main() -> None:
    args = parse_args()
    try:
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required. Install it with: "
            "python3 -m pip install -U 'huggingface_hub[hf_xet]'"
        ) from exc

    project_root = args.project_root.resolve()
    project_root.mkdir(parents=True, exist_ok=True)
    token = True  # Read `hf auth login` or HF_TOKEN; do not expose the token.

    marker = dataset_marker(project_root)
    if args.force or not marker.exists():
        print(f"Downloading dataset repo {args.dataset_repo}@{args.dataset_revision} ...")
        snapshot_download(
            repo_id=args.dataset_repo,
            repo_type="dataset",
            revision=args.dataset_revision,
            local_dir=project_root,
            allow_patterns=["indoor5-v2-student/**"],
            token=token,
            max_workers=16,
        )
    else:
        print(f"Dataset already present: {marker}")

    model_path = download_checkpoint_atomic(
        hf_hub_download,
        repo_id=args.model_repo,
        revision=args.model_revision,
        project_root=project_root,
        run_name=args.model_run_name,
        filename="best.pth",
        force=args.force,
        token=token,
    )
    if args.download_last:
        download_checkpoint_atomic(
            hf_hub_download,
            repo_id=args.model_repo,
            revision=args.model_revision,
            project_root=project_root,
            run_name=args.model_run_name,
            filename="last.pth",
            force=args.force,
            token=token,
        )

    # Authenticated metadata checks catch a typo in a private repo ID even if
    # a local artifact happened to exist already.
    api = HfApi(token=token)
    dataset_info = api.repo_info(args.dataset_repo, repo_type="dataset", revision=args.dataset_revision)
    model_info = api.repo_info(args.model_repo, repo_type="model", revision=args.model_revision)
    result = {
        "dataset_repo": args.dataset_repo,
        "dataset_revision": getattr(dataset_info, "sha", args.dataset_revision),
        "model_repo": args.model_repo,
        "model_revision": getattr(model_info, "sha", args.model_revision),
        "dataset_marker": str(marker),
        "model_path": str(model_path),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
