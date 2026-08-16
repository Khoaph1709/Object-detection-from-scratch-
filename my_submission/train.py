from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from my_submission.models.detector import build_detector
from my_submission.utils.assigner import FCOSTargetAssigner
from my_submission.utils.augmentations import DetectionTransform
from my_submission.utils.classes import set_active_classes
from my_submission.utils.config import add_config_argument, apply_config_defaults
from my_submission.utils.dataset import DetectionDataset, detection_collate_fn
from my_submission.utils.locations import generate_locations
from my_submission.utils.losses import FCOSLoss
from my_submission.utils.postprocess import decode_detections, scale_detections_to_original


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the custom object detector.")
    add_config_argument(parser)
    parser.add_argument("--train_data", default="")
    parser.add_argument("--val_data", default="")
    parser.add_argument("--image_dir", default="")
    parser.add_argument("--val_image_dir", default="")
    parser.add_argument("--checkpoint_dir", default="")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--backbone_lr", type=float, default=0.0)
    parser.add_argument("--head_lr", type=float, default=0.0)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--scheduler", choices=["cosine", "none"], default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--short_size", type=int, default=512)
    parser.add_argument("--max_size", type=int, default=768)
    parser.add_argument("--score_threshold", type=float, default=0.05)
    parser.add_argument("--score_cls_power", type=float, default=0.5)
    parser.add_argument("--score_centerness_power", type=float, default=0.5)
    parser.add_argument("--nms_threshold", type=float, default=0.55)
    parser.add_argument("--pre_nms_topk", type=int, default=1000)
    parser.add_argument("--center_sampling_radius", type=float, default=1.5)
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--chair_positive_weight", type=float, default=1.0)
    parser.add_argument("--chair_negative_weight", type=float, default=1.0)
    parser.add_argument("--hard_negative_sampling", action="store_true")
    parser.add_argument("--empty_image_weight", type=float, default=1.0)
    parser.add_argument("--chair_confuser_weight", type=float, default=1.0)
    parser.add_argument("--chair_positive_image_weight", type=float, default=1.0)
    parser.add_argument("--mined_sampler_weights", default="")
    parser.add_argument("--freeze_backbone_epochs", type=int, default=0)
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--no_tensorboard", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Enable CUDA automatic mixed precision for faster training.")
    parser.add_argument("--no_amp", action="store_true", help="Deprecated compatibility flag; AMP is disabled unless --amp is set.")
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--overfit_images", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_pretrained_backbone", action="store_true")
    parser.add_argument("--resume", default="", help="Path to a checkpoint to resume from.")
    parser.add_argument(
        "--resume_model_only",
        action="store_true",
        help="Load only model weights from --resume and start a fresh optimizer/scheduler.",
    )
    parser.add_argument(
        "--auto_resume",
        action="store_true",
        help="Resume from checkpoint_dir/last.pth when it exists.",
    )
    return apply_config_defaults(parser)


def main() -> None:
    args = parse_args()
    require_args(args, ["train_data", "val_data", "image_dir", "val_image_dir", "checkpoint_dir"])
    torch.manual_seed(42)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")
    train_transform = DetectionTransform(
        train=True,
        short_size=(args.short_size, max(320, args.short_size - 64), args.short_size + 64),
        max_size=args.max_size,
        horizontal_flip_prob=0.5,
        color_jitter=0.2,
    )
    val_transform = DetectionTransform(
        train=False,
        short_size=args.short_size,
        max_size=args.max_size,
        horizontal_flip_prob=0.0,
        color_jitter=0.0,
    )

    train_dataset = DetectionDataset(args.train_data, args.image_dir, transform=train_transform)
    val_dataset = DetectionDataset(args.val_data, args.val_image_dir, transform=val_transform)
    if train_dataset.classes != val_dataset.classes:
        raise ValueError(
            "Train/val class lists must match exactly. "
            f"train={train_dataset.classes}, val={val_dataset.classes}"
        )
    set_active_classes(train_dataset.classes)
    num_classes = len(train_dataset.classes)
    if args.overfit_images > 0:
        train_dataset = Subset(train_dataset, list(range(min(args.overfit_images, len(train_dataset)))))
        val_dataset = Subset(val_dataset, list(range(min(args.overfit_images, len(val_dataset)))))

    train_sampler = build_train_sampler(train_dataset, args)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=detection_collate_fn,
    )

    model = build_detector(num_classes=num_classes, pretrained_backbone=not args.no_pretrained_backbone).to(device)
    assigner = FCOSTargetAssigner(model.strides, center_sampling_radius=args.center_sampling_radius)
    criterion = FCOSLoss(
        num_classes=num_classes,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        chair_positive_weight=args.chair_positive_weight,
        chair_negative_weight=args.chair_negative_weight,
    )
    optimizer = build_optimizer(model, args)
    total_steps = args.max_steps if args.max_steps > 0 else args.epochs * len(train_loader)
    scheduler = build_scheduler(optimizer, args, total_steps)
    amp_enabled = device.type == "cuda" and args.amp and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    train_log_path = checkpoint_dir / "train_log.csv"
    val_history_path = checkpoint_dir / "val_history.jsonl"
    ensure_train_log_header(train_log_path)
    tb_writer = create_tensorboard_writer(checkpoint_dir, disabled=args.no_tensorboard)

    best_map = -1.0
    global_step = 0
    start_epoch = 1
    resume_path = resolve_resume_path(args, checkpoint_dir)
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device)
        checkpoint_classes = checkpoint.get("class_names")
        if checkpoint_classes and checkpoint_classes != train_dataset.classes:
            raise ValueError(
                "Resume checkpoint class names do not match current dataset classes. "
                f"checkpoint={checkpoint_classes}, dataset={train_dataset.classes}"
            )
        model.load_state_dict(checkpoint["model"])
        if not args.resume_model_only and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if not args.resume_model_only and "scheduler" in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if not args.resume_model_only and "scaler" in checkpoint and device.type == "cuda":
            scaler.load_state_dict(checkpoint["scaler"])
        best_map = float(checkpoint.get("best_map", -1.0))
        if args.resume_model_only:
            print(
                f"Loaded model weights from {resume_path}; starting fresh optimizer/scheduler "
                f"with previous best mAP@0.5={best_map:.6f}"
            )
        else:
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            global_step = int(checkpoint.get("global_step", 0))
            print(
                f"Resumed from {resume_path} at epoch {start_epoch} "
                f"with best mAP@0.5={best_map:.6f}"
            )
        if not args.resume_model_only and start_epoch > args.epochs:
            print(
                f"Checkpoint already reached epoch {start_epoch - 1}; "
                f"increase --epochs above {start_epoch - 1} to continue training."
            )
            return

    for epoch in range(start_epoch, args.epochs + 1):
        set_backbone_trainable(model, trainable=epoch > args.freeze_backbone_epochs)
        model.train()
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch in progress:
            images = batch["images"].to(device)
            targets = move_targets_to_device(batch["targets"], device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = model(images)
                locations = generate_locations(outputs["features"], model.strides)
                assigned = assigner(locations, targets)
                losses = criterion(outputs, assigned, model.strides)

            if not torch.isfinite(losses["loss"]):
                global_step += 1
                gpu_stats = get_gpu_memory_stats(device)
                loss_row = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "lr": optimizer.param_groups[0]["lr"],
                    "loss": float("nan"),
                    "loss_cls": float(losses["loss_cls"].detach().cpu())
                    if torch.isfinite(losses["loss_cls"])
                    else float("nan"),
                    "loss_box": float(losses["loss_box"].detach().cpu())
                    if torch.isfinite(losses["loss_box"])
                    else float("nan"),
                    "loss_centerness": float(losses["loss_centerness"].detach().cpu())
                    if torch.isfinite(losses["loss_centerness"])
                    else float("nan"),
                    "num_positive": int(losses["num_positive"].cpu()),
                    **gpu_stats,
                }
                append_train_log(train_log_path, loss_row)
                log_train_tensorboard(tb_writer, loss_row)
                print(f"WARNING: non-finite loss at step {global_step}; skipped optimizer update.")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scale_before_step = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_step_applied = scaler.get_scale() >= scale_before_step
            if scheduler is not None and optimizer_step_applied:
                scheduler.step()

            global_step += 1
            gpu_stats = get_gpu_memory_stats(device)
            loss_row = {
                "epoch": epoch,
                "global_step": global_step,
                "lr": optimizer.param_groups[0]["lr"],
                "loss": float(losses["loss"].detach().cpu()),
                "loss_cls": float(losses["loss_cls"].detach().cpu()),
                "loss_box": float(losses["loss_box"].detach().cpu()),
                "loss_centerness": float(losses["loss_centerness"].detach().cpu()),
                "num_positive": int(losses["num_positive"].cpu()),
                **gpu_stats,
            }
            progress.set_postfix(
                loss=f"{loss_row['loss']:.3f}",
                cls=f"{loss_row['loss_cls']:.3f}",
                box=f"{loss_row['loss_box']:.3f}",
                ctr=f"{loss_row['loss_centerness']:.3f}",
                pos=loss_row["num_positive"],
                mem=f"{loss_row['gpu_reserved_mb']:.0f}MB",
            )
            if global_step == 1 or global_step % args.log_interval == 0:
                append_train_log(train_log_path, loss_row)
                log_train_tensorboard(tb_writer, loss_row)
            if args.max_steps and global_step >= args.max_steps:
                break

        save_checkpoint(
            checkpoint_dir / "last.pth",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_map,
            global_step,
            args,
            class_names=train_dataset.classes,
        )

        if epoch % args.val_interval == 0:
            predictions_path = checkpoint_dir / "val_predictions.json"
            write_predictions(
                model,
                val_dataset,
                device,
                predictions_path,
                score_threshold=args.score_threshold,
                nms_threshold=args.nms_threshold,
                pre_nms_topk=args.pre_nms_topk,
                score_cls_power=args.score_cls_power,
                score_centerness_power=args.score_centerness_power,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            score = run_public_evaluator(
                Path(args.val_data),
                predictions_path,
                allow_missing_images=args.overfit_images > 0,
            )
            current_map = float(score.get("mAP@0.5", 0.0))
            print(json.dumps(score, indent=2))
            append_val_history(
                val_history_path,
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "mAP@0.5": current_map,
                    "micro_precision": score.get("micro_precision", 0.0),
                    "micro_recall": score.get("micro_recall", 0.0),
                    "num_predictions": score.get("num_predictions", 0),
                    "per_class": score.get("per_class", {}),
                },
            )
            log_val_tensorboard(tb_writer, epoch, global_step, score)
            print(f"Logged training metrics to {train_log_path}")
            print(f"Logged validation history to {val_history_path}")
            if tb_writer is not None:
                print(f"Logged TensorBoard events to {checkpoint_dir / 'tensorboard'}")
            if current_map > best_map:
                best_map = current_map
                save_checkpoint(
                    checkpoint_dir / "best.pth",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    best_map,
                    global_step,
                    args,
                    class_names=train_dataset.classes,
                )
                print(f"Saved new best checkpoint with mAP@0.5={best_map:.6f}")
            save_checkpoint(
                checkpoint_dir / "last.pth",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_map,
                global_step,
                args,
                class_names=train_dataset.classes,
            )

        if args.max_steps and global_step >= args.max_steps:
            break

    if not (checkpoint_dir / "best.pth").exists():
        save_checkpoint(
            checkpoint_dir / "best.pth",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_map,
            global_step,
            args,
            class_names=train_dataset.classes,
        )
    if tb_writer is not None:
        tb_writer.close()


def move_targets_to_device(targets: list[dict], device: torch.device) -> list[dict]:
    moved = []
    for target in targets:
        moved.append(
            {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in target.items()
            }
        )
    return moved


def require_args(args: argparse.Namespace, names: list[str]) -> None:
    missing = [name for name in names if not getattr(args, name)]
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join('--' + name for name in missing)}")


def build_train_sampler(dataset, args: argparse.Namespace):
    if not args.hard_negative_sampling and not args.mined_sampler_weights:
        return None

    base_dataset = dataset
    indices = None
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        indices = list(dataset.indices)

    if not isinstance(base_dataset, DetectionDataset):
        return None

    mined_weights = read_mined_sampler_weights(args.mined_sampler_weights)
    source_indices = indices if indices is not None else list(range(len(base_dataset)))
    weights = []
    for source_index in source_indices:
        image_info = base_dataset.images[source_index]
        if image_info["id"] in mined_weights:
            weights.append(mined_weights[image_info["id"]])
            continue

        annotations = base_dataset.annotations_by_image.get(image_info["id"], [])
        labels = {annotation["class"] for annotation in annotations}
        if not annotations:
            weight = args.empty_image_weight
        elif "chair" in labels:
            weight = args.chair_positive_image_weight
        elif "person" in labels:
            weight = args.chair_confuser_weight
        else:
            weight = 1.0
        weights.append(max(float(weight), 1e-6))

    if mined_weights:
        print(f"Using mined sampler weights from {args.mined_sampler_weights}")
    else:
        print(
            "Using hard-negative sampler: "
            f"empty={args.empty_image_weight}, "
            f"person_no_chair={args.chair_confuser_weight}, "
            f"chair_positive={args.chair_positive_image_weight}"
        )
    return WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def read_mined_sampler_weights(path: str) -> dict[str, float]:
    if not path:
        return {}
    weight_path = Path(path)
    if not weight_path.exists():
        raise FileNotFoundError(f"Mined sampler weights not found: {weight_path}")
    data = json.loads(weight_path.read_text(encoding="utf-8"))
    if "image_weights" in data:
        data = data["image_weights"]
    return {str(image_id): max(float(weight), 1e-6) for image_id, weight in data.items()}


def build_optimizer(model, args: argparse.Namespace):
    backbone_lr = args.backbone_lr if args.backbone_lr > 0 else args.lr
    head_lr = args.head_lr if args.head_lr > 0 else args.lr
    if backbone_lr == args.lr and head_lr == args.lr:
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    return torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": backbone_lr},
            {"params": model.fpn.parameters(), "lr": head_lr},
            {"params": model.head.parameters(), "lr": head_lr},
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


def set_backbone_trainable(model, trainable: bool) -> None:
    for parameter in model.backbone.parameters():
        parameter.requires_grad = trainable


def get_gpu_memory_stats(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {
            "gpu_allocated_mb": 0.0,
            "gpu_reserved_mb": 0.0,
            "gpu_max_allocated_mb": 0.0,
        }
    return {
        "gpu_allocated_mb": torch.cuda.memory_allocated(device) / (1024**2),
        "gpu_reserved_mb": torch.cuda.memory_reserved(device) / (1024**2),
        "gpu_max_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
    }


def build_scheduler(optimizer, args: argparse.Namespace, total_steps: int):
    if args.scheduler == "none":
        return None

    warmup_steps = min(args.warmup_steps, max(total_steps - 1, 0))
    min_lr_ratio = args.min_lr / args.lr if args.lr > 0 else 0.0

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-3)

        decay_steps = max(total_steps - warmup_steps, 1)
        decay_step = min(max(step - warmup_steps, 0), decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * decay_step / decay_steps))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def ensure_train_log_header(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "global_step",
                "lr",
                "loss",
                "loss_cls",
                "loss_box",
                "loss_centerness",
                "num_positive",
                "gpu_allocated_mb",
                "gpu_reserved_mb",
                "gpu_max_allocated_mb",
            ],
        )
        writer.writeheader()


def append_train_log(path: Path, row: dict) -> None:
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        writer.writerow(row)


def append_val_history(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def create_tensorboard_writer(checkpoint_dir: Path, disabled: bool = False):
    if disabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("TensorBoard is not installed. CSV/JSONL logging remains enabled.")
        return None
    log_dir = checkpoint_dir / "tensorboard"
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))


def log_train_tensorboard(writer, row: dict) -> None:
    if writer is None:
        return
    step = row["global_step"]
    writer.add_scalar("train/loss_total", row["loss"], step)
    writer.add_scalar("train/loss_cls", row["loss_cls"], step)
    writer.add_scalar("train/loss_box", row["loss_box"], step)
    writer.add_scalar("train/loss_centerness", row["loss_centerness"], step)
    writer.add_scalar("train/num_positive", row["num_positive"], step)
    writer.add_scalar("train/lr", row["lr"], step)
    writer.add_scalar("gpu/memory_allocated_mb", row.get("gpu_allocated_mb", 0.0), step)
    writer.add_scalar("gpu/memory_reserved_mb", row.get("gpu_reserved_mb", 0.0), step)
    writer.add_scalar("gpu/memory_max_allocated_mb", row.get("gpu_max_allocated_mb", 0.0), step)
    writer.flush()


def log_val_tensorboard(writer, epoch: int, global_step: int, score: dict) -> None:
    if writer is None:
        return
    writer.add_scalar("val/mAP@0.5", score.get("mAP@0.5", 0.0), global_step)
    writer.add_scalar("val/micro_precision", score.get("micro_precision", 0.0), global_step)
    writer.add_scalar("val/micro_recall", score.get("micro_recall", 0.0), global_step)
    writer.add_scalar("val/num_predictions", score.get("num_predictions", 0), global_step)
    for class_name, metrics in score.get("per_class", {}).items():
        writer.add_scalar(f"val_ap/{class_name}", metrics.get("ap", 0.0), global_step)
        writer.add_scalar(f"val_recall/{class_name}", metrics.get("recall", 0.0), global_step)
        writer.add_scalar(f"val_precision/{class_name}", metrics.get("precision", 0.0), global_step)
    writer.add_scalar("epoch", epoch, global_step)
    writer.flush()


def resolve_resume_path(args: argparse.Namespace, checkpoint_dir: Path) -> Path | None:
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        return resume_path

    if args.auto_resume:
        resume_path = checkpoint_dir / "last.pth"
        if resume_path.exists():
            return resume_path
        print(f"--auto_resume enabled, but no checkpoint found at {resume_path}. Starting fresh.")
    return None


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_map: float,
    global_step: int,
    args: argparse.Namespace,
    class_names: list[str],
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_map": best_map,
            "global_step": global_step,
            "args": vars(args),
            "class_names": class_names,
        },
        path,
    )


@torch.no_grad()
def write_predictions(
    model,
    dataset,
    device: torch.device,
    output_path: Path,
    score_threshold: float,
    nms_threshold: float,
    pre_nms_topk: int,
    score_cls_power: float = 0.5,
    score_centerness_power: float = 0.5,
    batch_size: int = 4,
    num_workers: int = 0,
) -> None:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
    )
    model.eval()
    results = []
    for batch in tqdm(loader, desc="predict-val"):
        images = batch["images"].to(device)
        outputs = model(images)
        image_sizes = [tuple(int(v) for v in target["resized_size"].tolist()) for target in batch["targets"]]
        detections = decode_detections(
            outputs,
            image_sizes=image_sizes,
            strides=model.strides,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            pre_nms_topk=pre_nms_topk,
            score_cls_power=score_cls_power,
            score_centerness_power=score_centerness_power,
        )
        for target, det in zip(batch["targets"], detections):
            scaled = scale_detections_to_original(
                det,
                resized_size=tuple(int(v) for v in target["resized_size"].tolist()),
                original_size=tuple(int(v) for v in target["original_size"].tolist()),
            )
            results.append({"image_id": target["image_id"], "boxes": scaled})

    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def run_public_evaluator(ground_truth: Path, predictions: Path, allow_missing_images: bool = False) -> dict:
    public_root = ground_truth.parents[1]
    evaluator = public_root / "tools" / "evaluate_predictions.py"
    if not evaluator.exists():
        return {"mAP@0.5": 0.0, "warning": f"Evaluator not found: {evaluator}"}

    output_path = predictions.with_suffix(".score.json")
    command = [
        sys.executable,
        str(evaluator),
        "--ground_truth",
        str(ground_truth),
        "--predictions",
        str(predictions),
        "--output",
        str(output_path),
    ]
    if allow_missing_images:
        command.append("--allow_missing_images")

    completed = subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(output_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
