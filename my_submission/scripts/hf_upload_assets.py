#!/usr/bin/env python3
"""Upload the project dataset and/or a checkpoint run to Hugging Face Hub."""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dataset-repo", help="HF dataset repo, e.g. user/indoor5-v2-student")
    parser.add_argument("--model-repo", help="HF model repo, e.g. user/indoor5-targeted-model")
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--private", action="store_true", default=True)
    parser.add_argument("--upload-last", action="store_true")
    return parser.parse_args()


def resolve_dataset_upload(dataset_dir: Path) -> tuple[Path, str]:
    """Return folder and Hub prefix that produce indoor5-v2-student/public/."""
    dataset_dir = dataset_dir.resolve()
    if dataset_dir.name == "public":
        public_dir = dataset_dir
        path_in_repo = "indoor5-v2-student/public"
    elif (dataset_dir / "public").is_dir():
        public_dir = dataset_dir / "public"
        path_in_repo = "indoor5-v2-student/public"
    else:
        raise ValueError(
            f"Expected {dataset_dir} to be indoor5-v2-student/ or its public/ directory."
        )

    required = [
        public_dir / "annotations" / "train.json",
        public_dir / "annotations" / "val.json",
        public_dir / "train" / "images",
        public_dir / "val" / "images",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ValueError("Dataset is missing required paths:\n" + "\n".join(missing))
    return public_dir, path_in_repo


def main() -> None:
    args = parse_args()
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required. Install it with: "
            "python3 -m pip install -U 'huggingface_hub[hf_xet]'"
        ) from exc

    project_root = args.project_root.resolve()
    api = HfApi(token=True)  # Reads `hf auth login`/HF_TOKEN; never print the token.

    if args.dataset_repo:
        dataset_dir = (args.dataset_dir or (project_root / "indoor5-v2-student")).resolve()
        if not dataset_dir.is_dir():
            raise SystemExit(f"Dataset directory not found: {dataset_dir}")
        try:
            public_dir, path_in_repo = resolve_dataset_upload(dataset_dir)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        api.create_repo(args.dataset_repo, repo_type="dataset", private=True, exist_ok=True)
        print(f"Uploading dataset public/ directory {public_dir} -> {args.dataset_repo}:{path_in_repo} ...")
        api.upload_folder(
            repo_id=args.dataset_repo,
            repo_type="dataset",
            folder_path=str(public_dir),
            path_in_repo=path_in_repo,
            ignore_patterns=["**/.cache/**", "**/audit*/**", "**/predictions*/**"],
            commit_message="Upload indoor5-v2-student/public dataset",
        )
        print(f"Dataset uploaded: https://huggingface.co/datasets/{args.dataset_repo}")

    if args.model_repo:
        if not args.run_name:
            raise SystemExit("--run-name is required when --model-repo is provided")
        run_dir = project_root / "checkpoints" / args.run_name
        if not run_dir.is_dir():
            raise SystemExit(f"Checkpoint run directory not found: {run_dir}")
        api.create_repo(args.model_repo, repo_type="model", private=True, exist_ok=True)
        filenames = ["best.pth"]
        if args.upload_last:
            filenames.append("last.pth")
        for filename in filenames:
            local_path = run_dir / filename
            if not local_path.is_file():
                if filename == "last.pth":
                    print(f"WARNING: optional checkpoint not found; skipping: {local_path}")
                    continue
                raise SystemExit(f"Required checkpoint not found: {local_path}")
            repo_path = f"checkpoints/{args.run_name}/{filename}"
            print(f"Uploading {local_path} -> {args.model_repo}:{repo_path} ...")
            api.upload_file(
                repo_id=args.model_repo,
                repo_type="model",
                path_or_fileobj=str(local_path),
                path_in_repo=repo_path,
                commit_message=f"Upload {args.run_name}/{filename}",
            )
        print(f"Model uploaded: https://huggingface.co/{args.model_repo}")


if __name__ == "__main__":
    main()
