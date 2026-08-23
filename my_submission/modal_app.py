from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import modal


APP_NAME = "xla-fcos-training"
VOLUME_NAME = "xla-fcos-volume"
VOLUME_MOUNT = Path("/data")
REMOTE_PROJECT = Path("/root/project")
REMOTE_SUBMISSION = REMOTE_PROJECT / "my_submission"
DEFAULT_RUN_NAME = "run_hem_l40s"
DEFAULT_TENSORBOARD_RUN_NAME = "run_hem_l40s"
TENSORBOARD_ACTIVE_RUN_FILE = VOLUME_MOUNT / "checkpoints" / ".tensorboard_active_run.txt"
DEFAULT_CONFIG = str(REMOTE_SUBMISSION / "configs" / "train_hem_l40s.json")
DEFAULT_PREDICT_CONFIG = str(REMOTE_SUBMISSION / "configs" / "predict_hem_l40s.json")

LOCAL_ROOT = Path(__file__).resolve().parents[1]


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libglib2.0-0", "libgl1")
    .pip_install(
        "torch",
        "torchvision",
        "timm",
        "Pillow",
        "numpy",
        "tqdm",
        "tensorboard",
    )
    .add_local_dir(LOCAL_ROOT / "my_submission", remote_path=str(REMOTE_SUBMISSION))
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


TRAIN_RETRIES = modal.Retries(initial_delay=0.0, max_retries=10)


@app.function(
    volumes={str(VOLUME_MOUNT): volume},
    cpu=2.0,
    memory=8192,
    timeout=60 * 60,
)
def merge_train_val_remote(
    output_name: str = "train_val_merged.json",
) -> dict:
    """Merge only the labeled train/val annotations inside the Modal Volume."""
    data_root = VOLUME_MOUNT / "indoor5-v2-student" / "public"
    train_path = data_root / "annotations" / "train.json"
    valid_path = data_root / "annotations" / "val.json"
    output_path = data_root / "annotations" / output_name
    require_path(train_path)
    require_path(valid_path)
    command = [
        sys.executable,
        str(REMOTE_SUBMISSION / "scripts" / "merge_train_val_annotations.py"),
        "--train",
        str(train_path),
        "--valid",
        str(valid_path),
        "--output",
        str(output_path),
        "--image-root",
        str(data_root),
    ]
    print("Running merge command:")
    print(" ".join(shlex.quote(part) for part in command))
    subprocess.run(command, cwd=str(REMOTE_PROJECT), check=True)
    volume.commit()
    return {
        "output": str(output_path),
        "train": str(train_path),
        "valid": str(valid_path),
    }


@app.function(
    volumes={str(VOLUME_MOUNT): volume},
    cpu=8.0,
    memory=32768,
    timeout=24 * 60 * 60,
    startup_timeout=20 * 60,
    retries=TRAIN_RETRIES,
    single_use_containers=True,
)
def train_remote(
    config_path: str = DEFAULT_CONFIG,
    run_name: str = DEFAULT_RUN_NAME,
    epochs: int = 0,
    batch_size: int = 0,
    val_interval: int = 0,
    score_threshold: float = -1.0,
    pre_nms_topk: int = 0,
    amp: bool = False,
    no_pretrained_backbone: bool = False,
    resume_model_only: bool = False,
    resume_checkpoint_name: str = "",
    resume_checkpoint_path: str = "",
    train_annotation_name: str = "train.json",
    val_annotation_name: str = "val.json",
    train_image_dir_name: str = "train/images",
    val_image_dir_name: str = "val/images",
) -> dict:
    checkpoint_dir = VOLUME_MOUNT / "checkpoints" / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    data_root = VOLUME_MOUNT / "indoor5-v2-student" / "public"
    train_annotation = data_root / "annotations" / train_annotation_name
    val_annotation = data_root / "annotations" / val_annotation_name
    train_image_dir = data_root / train_image_dir_name
    val_image_dir = data_root / val_image_dir_name
    require_path(train_annotation)
    require_path(val_annotation)
    require_path(train_image_dir)
    require_path(val_image_dir)

    command = [
        sys.executable,
        str(REMOTE_SUBMISSION / "train.py"),
        "--config",
        config_path,
        "--train_data",
        str(train_annotation),
        "--val_data",
        str(val_annotation),
        "--image_dir",
        str(train_image_dir),
        "--val_image_dir",
        str(val_image_dir),
        "--checkpoint_dir",
        str(checkpoint_dir),
        "--device",
        "cuda",
    ]
    if epochs > 0:
        command += ["--epochs", str(epochs)]
    if batch_size > 0:
        command += ["--batch_size", str(batch_size)]
    if val_interval > 0:
        command += ["--val_interval", str(val_interval)]
    if score_threshold >= 0:
        command += ["--score_threshold", str(score_threshold)]
    if pre_nms_topk > 0:
        command += ["--pre_nms_topk", str(pre_nms_topk)]
    if amp:
        command.append("--amp")
    if no_pretrained_backbone:
        command.append("--no_pretrained_backbone")
    existing_last = checkpoint_dir / "last.pth"
    resume_existing_last = existing_last.exists() and bool(resume_checkpoint_path or resume_checkpoint_name)
    if resume_existing_last:
        # A retry must continue from the latest durable checkpoint, not restart
        # from the original warm-start model. Explicit --resume takes priority
        # inside train.py, so omit it when a run checkpoint already exists.
        command.append("--auto_resume")
        print(f"Found existing checkpoint {existing_last}; retry will auto-resume from it.")
    else:
        if resume_model_only:
            command.append("--resume_model_only")
        if resume_checkpoint_path:
            command += ["--resume", resume_checkpoint_path]
        elif resume_checkpoint_name:
            command += ["--resume", str(checkpoint_dir / resume_checkpoint_name)]

    print("Running training command:")
    print(" ".join(shlex.quote(part) for part in command))
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    subprocess.run(command, cwd=str(REMOTE_PROJECT), check=True, env=env)
    volume.commit()
    return summarize_run(checkpoint_dir)


@app.function(volumes={str(VOLUME_MOUNT): volume}, timeout=60 * 60)
def pack_outputs_remote(run_name: str = DEFAULT_RUN_NAME) -> str:
    import zipfile

    checkpoint_dir = VOLUME_MOUNT / "checkpoints" / run_name
    require_path(checkpoint_dir)
    export_dir = VOLUME_MOUNT / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    zip_path = export_dir / f"{run_name}_outputs.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in checkpoint_dir.rglob("*"):
            if path.is_file():
                archive.write(path, arcname=str(path.relative_to(checkpoint_dir)))
    volume.commit()
    print(f"Packed outputs to {zip_path}")
    return str(zip_path)


@app.function(timeout=10 * 60)
def gpu_status_remote() -> dict:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True)
    return {"nvidia_smi": output.strip()}


@app.function(
    volumes={str(VOLUME_MOUNT): volume},
    cpu=8.0,
    memory=32768,
    timeout=6 * 60 * 60,
    startup_timeout=20 * 60,
)
def predict_val_remote(
    run_name: str = DEFAULT_RUN_NAME,
    config_path: str = DEFAULT_PREDICT_CONFIG,
    checkpoint_name: str = "best.pth",
    output_name: str = "val_predictions_tta.json",
) -> dict:
    checkpoint_dir = VOLUME_MOUNT / "checkpoints" / run_name
    checkpoint_path = checkpoint_dir / checkpoint_name
    output_path = checkpoint_dir / output_name
    score_path = output_path.with_suffix(".score.json")
    data_root = VOLUME_MOUNT / "indoor5-v2-student" / "public"
    require_path(checkpoint_path)
    require_path(data_root / "annotations" / "val.json")
    require_path(data_root / "val" / "images")

    predict_command = [
        sys.executable,
        str(REMOTE_SUBMISSION / "predict.py"),
        "--config",
        config_path,
        "--image_dir",
        str(data_root / "val" / "images"),
        "--output",
        str(output_path),
        "--checkpoint",
        str(checkpoint_path),
        "--device",
        "cuda",
    ]
    print("Running validation prediction command:")
    print(" ".join(shlex.quote(part) for part in predict_command))
    subprocess.run(predict_command, cwd=str(REMOTE_PROJECT), check=True)

    evaluator = data_root / "tools" / "evaluate_predictions.py"
    eval_command = [
        sys.executable,
        str(evaluator),
        "--ground_truth",
        str(data_root / "annotations" / "val.json"),
        "--predictions",
        str(output_path),
        "--output",
        str(score_path),
    ]
    subprocess.run(eval_command, check=True)
    score = json.loads(score_path.read_text(encoding="utf-8"))
    volume.commit()
    return {
        "checkpoint": str(checkpoint_path),
        "predictions": str(output_path),
        "score_file": str(score_path),
        "score": score,
    }


@app.function(
    volumes={str(VOLUME_MOUNT): volume},
    cpu=8.0,
    memory=32768,
    timeout=6 * 60 * 60,
    startup_timeout=20 * 60,
)
def predict_remote(
    run_name: str = DEFAULT_RUN_NAME,
    config_path: str = DEFAULT_PREDICT_CONFIG,
    image_dir: str = "/data/indoor5-v2-student/public/val/images",
    checkpoint_name: str = "best.pth",
    output_name: str = "predictions.json",
) -> dict:
    checkpoint_dir = VOLUME_MOUNT / "checkpoints" / run_name
    checkpoint_path = checkpoint_dir / checkpoint_name
    output_path = checkpoint_dir / output_name
    require_path(checkpoint_path)
    require_path(Path(image_dir))

    command = [
        sys.executable,
        str(REMOTE_SUBMISSION / "predict.py"),
        "--config",
        config_path,
        "--image_dir",
        image_dir,
        "--output",
        str(output_path),
        "--checkpoint",
        str(checkpoint_path),
        "--device",
        "cuda",
    ]
    print("Running prediction command:")
    print(" ".join(shlex.quote(part) for part in command))
    subprocess.run(command, cwd=str(REMOTE_PROJECT), check=True)
    volume.commit()
    return {
        "checkpoint": str(checkpoint_path),
        "image_dir": image_dir,
        "predictions": str(output_path),
    }



@app.function(volumes={str(VOLUME_MOUNT): volume}, timeout=2 * 60)
def set_active_tensorboard_run_remote(run_name: str) -> str:
    checkpoints_dir = VOLUME_MOUNT / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_ACTIVE_RUN_FILE.write_text(run_name, encoding="utf-8")
    volume.commit()
    return run_name


@app.function(
    volumes={str(VOLUME_MOUNT): volume},
    timeout=12 * 60 * 60,
    scaledown_window=10 * 60,
)
@modal.web_server(6006, startup_timeout=60)
def tensorboard() -> None:
    volume.reload()
    active_run = read_active_tensorboard_run()
    logdir = VOLUME_MOUNT / "checkpoints" / active_run / "tensorboard"
    logdir.mkdir(parents=True, exist_ok=True)
    print(f"TensorBoard active run: {active_run}")
    print(f"Log directory: {logdir}")
    subprocess.Popen(
        [
            "tensorboard",
            "--logdir",
            str(logdir),
            "--host",
            "0.0.0.0",
            "--port",
            "6006",
        ]
    )


@app.local_entrypoint()
def main(
    action: str = "train",
    gpu: str = "L40S",
    run_name: str = DEFAULT_RUN_NAME,
    config_path: str = DEFAULT_CONFIG,
    local_data_dir: str = "indoor5-v2-student",
    local_checkpoint_dir: str = "my_submission/models/baseline_fcos_smoke_2ep",
    local_image_dir: str = "",
    remote_image_dir: str = "/data/indoor5-v2-student/public/val/images",
    checkpoint_name: str = "best.pth",
    output_name: str = "predictions.json",
    epochs: int = 0,
    batch_size: int = 0,
    val_interval: int = 0,
    score_threshold: float = -1.0,
    pre_nms_topk: int = 0,
    amp: bool = False,
    no_pretrained_backbone: bool = False,
    resume_model_only: bool = False,
    resume_checkpoint_name: str = "",
    resume_checkpoint_path: str = "",
    train_annotation_name: str = "train.json",
    val_annotation_name: str = "val.json",
    train_image_dir_name: str = "train/images",
    val_image_dir_name: str = "val/images",
) -> None:
    if action == "upload":
        upload_dataset(local_data_dir)
        return
    if action == "upload_checkpoint":
        upload_checkpoint(local_checkpoint_dir, run_name)
        return
    if action == "merge_train_val":
        result = merge_train_val_remote.remote(output_name="train_val_merged.json")
        print(json.dumps(result, indent=2))
        return
    if action == "upload_images":
        upload_images(local_image_dir, remote_image_dir)
        return
    if action == "train":
        active_run = set_active_tensorboard_run_remote.remote(run_name)
        print(f"Set TensorBoard active run to: {active_run}")
        train_call = train_remote.with_options(gpu=gpu).spawn(
            config_path=config_path,
            run_name=run_name,
            epochs=epochs,
            batch_size=batch_size,
            val_interval=val_interval,
            score_threshold=score_threshold,
            pre_nms_topk=pre_nms_topk,
            amp=amp,
            no_pretrained_backbone=no_pretrained_backbone,
            resume_model_only=resume_model_only,
            resume_checkpoint_name=resume_checkpoint_name,
            resume_checkpoint_path=resume_checkpoint_path,
            train_annotation_name=train_annotation_name,
            val_annotation_name=val_annotation_name,
            train_image_dir_name=train_image_dir_name,
            val_image_dir_name=val_image_dir_name,
        )
        result = train_call.get()
        print(json.dumps(result, indent=2))
        print_download_commands(run_name)
        return
    if action == "pack":
        active_run = set_active_tensorboard_run_remote.remote(run_name)
        print(f"Set TensorBoard active run to: {active_run}")
        zip_path = pack_outputs_remote.remote(run_name=run_name)
        print(f"Remote zip: {zip_path}")
        print(f"Download with: modal volume get {VOLUME_NAME} /exports/{run_name}_outputs.zip .")
        return
    if action == "gpu_status":
        result = gpu_status_remote.with_options(gpu=gpu).remote()
        print(json.dumps(result, indent=2))
        return
    if action == "predict_val":
        active_run = set_active_tensorboard_run_remote.remote(run_name)
        print(f"Set TensorBoard active run to: {active_run}")
        result = predict_val_remote.with_options(gpu=gpu).remote(
            run_name=run_name,
            config_path=config_path if config_path != DEFAULT_CONFIG else DEFAULT_PREDICT_CONFIG,
            checkpoint_name=checkpoint_name,
            output_name=output_name if output_name != "predictions.json" else "val_predictions_tta.json",
        )
        print(json.dumps(result, indent=2))
        return
    if action == "predict":
        active_run = set_active_tensorboard_run_remote.remote(run_name)
        print(f"Set TensorBoard active run to: {active_run}")
        result = predict_remote.with_options(gpu=gpu).remote(
            run_name=run_name,
            config_path=config_path if config_path != DEFAULT_CONFIG else DEFAULT_PREDICT_CONFIG,
            image_dir=remote_image_dir,
            checkpoint_name=checkpoint_name,
            output_name=output_name,
        )
        print(json.dumps(result, indent=2))
        print(f"Download with: modal volume get {VOLUME_NAME} /checkpoints/{run_name}/{output_name} .")
        return
    if action == "commands":
        print_download_commands(run_name)
        print("TensorBoard: modal serve my_submission/modal_app.py")
        return
    raise ValueError(f"Unknown action: {action}")


def upload_dataset(local_data_dir: str) -> None:
    local_path = Path(local_data_dir).resolve()
    public_path = local_path if local_path.name == "public" else local_path / "public"
    if not public_path.is_dir():
        raise FileNotFoundError(
            f"Expected dataset public/ directory not found: {public_path}. "
            "Pass either indoor5-v2-student/ or indoor5-v2-student/public/."
        )
    required = [
        public_path / "annotations" / "train.json",
        public_path / "annotations" / "val.json",
        public_path / "train" / "images",
        public_path / "val" / "images",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Dataset is missing required paths:\n" + "\n".join(missing))
    print(f"Uploading local {public_path} to Modal volume {VOLUME_NAME}:/indoor5-v2-student")
    with volume.batch_upload() as batch:
        # put_directory preserves the local directory basename. Uploading
        # local public/ to its parent creates /indoor5-v2-student/public/.
        batch.put_directory(str(public_path), "/indoor5-v2-student")
    print("Upload complete.")
    print(f"Check with: modal volume ls {VOLUME_NAME} /indoor5-v2-student/public/annotations")


def upload_checkpoint(local_checkpoint_dir: str, run_name: str) -> None:
    local_path = Path(local_checkpoint_dir)
    if not local_path.exists():
        raise FileNotFoundError(f"Checkpoint folder not found: {local_path}")
    if not (local_path / "last.pth").exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {local_path / 'last.pth'}")
    remote_path = f"/checkpoints/{run_name}"
    print(f"Uploading {local_path} to Modal volume {VOLUME_NAME}:{remote_path}")
    with volume.batch_upload() as batch:
        batch.put_directory(str(local_path), remote_path)
    print("Checkpoint upload complete.")
    print(f"Check with: modal volume ls {VOLUME_NAME} {remote_path}")


def upload_images(local_image_dir: str, remote_image_dir: str) -> None:
    if not local_image_dir:
        raise ValueError("Provide --local-image-dir for action=upload_images.")
    local_path = Path(local_image_dir)
    if not local_path.exists():
        raise FileNotFoundError(f"Image folder not found: {local_path}")
    if not remote_image_dir.startswith("/data/"):
        raise ValueError("remote_image_dir must be inside /data on the Modal volume.")
    volume_path = remote_image_dir.removeprefix("/data")
    print(f"Uploading {local_path} to Modal volume {VOLUME_NAME}:{volume_path}")
    with volume.batch_upload() as batch:
        batch.put_directory(str(local_path), volume_path)
    print("Image upload complete.")
    print(f"Check with: modal volume ls {VOLUME_NAME} {volume_path}")


def summarize_run(checkpoint_dir: Path) -> dict:
    summary = {
        "checkpoint_dir": str(checkpoint_dir),
        "last_checkpoint": str(checkpoint_dir / "last.pth"),
        "best_checkpoint": str(checkpoint_dir / "best.pth"),
        "train_log": str(checkpoint_dir / "train_log.csv"),
        "val_history": str(checkpoint_dir / "val_history.jsonl"),
        "tensorboard": str(checkpoint_dir / "tensorboard"),
    }
    score_path = checkpoint_dir / "val_predictions.score.json"
    if score_path.exists():
        summary["latest_score"] = json.loads(score_path.read_text(encoding="utf-8"))
    return summary


def print_download_commands(run_name: str) -> None:
    print("Useful download commands:")
    print(f"modal volume get {VOLUME_NAME} /checkpoints/{run_name}/best.pth ./modal_outputs/{run_name}/best.pth")
    print(f"modal volume get {VOLUME_NAME} /checkpoints/{run_name}/last.pth ./modal_outputs/{run_name}/last.pth")
    print(f"modal volume get {VOLUME_NAME} /checkpoints/{run_name}/train_log.csv ./modal_outputs/{run_name}/train_log.csv")
    print(f"modal volume get {VOLUME_NAME} /checkpoints/{run_name}/val_history.jsonl ./modal_outputs/{run_name}/val_history.jsonl")
    print(f"modal volume get {VOLUME_NAME} /checkpoints/{run_name}/val_predictions.score.json ./modal_outputs/{run_name}/val_predictions.score.json")
    print(f"modal volume get {VOLUME_NAME} /checkpoints/{run_name}/val_predictions_tta.score.json ./modal_outputs/{run_name}/val_predictions_tta.score.json")
    print(f"modal volume get {VOLUME_NAME} /checkpoints/{run_name}/predictions.json ./modal_outputs/{run_name}/predictions.json")
    print(f"Pack everything first: modal run my_submission/modal_app.py --action pack --run-name {run_name}")


def require_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required path not found: {path}")


def read_active_tensorboard_run() -> str:
    if TENSORBOARD_ACTIVE_RUN_FILE.exists():
        run_name = TENSORBOARD_ACTIVE_RUN_FILE.read_text(encoding="utf-8").strip()
        if run_name:
            return run_name
    return DEFAULT_TENSORBOARD_RUN_NAME
