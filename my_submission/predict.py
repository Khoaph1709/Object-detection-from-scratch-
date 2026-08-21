from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from my_submission.models.detector import build_detector
from my_submission.utils.augmentations import DetectionTransform
from my_submission.utils.classes import set_active_classes
from my_submission.utils.checkpoint import resolve_checkpoint_path
from my_submission.utils.config import add_config_argument, apply_config_defaults
from my_submission.utils.dataset import detection_collate_fn
from my_submission.utils.tiling import TiledImageDataset, group_tile_predictions
from my_submission.utils.postprocess import (
    decode_detections,
    flip_detections_horizontal,
    merge_detections,
    merge_detections_consensus,
    scale_detections_to_original,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with the custom object detector.")
    add_config_argument(parser)
    parser.add_argument("--image_dir", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--tile_inference", action="store_true")
    parser.add_argument("--tile_size", type=int, default=640)
    parser.add_argument("--tile_overlap", type=float, default=0.20)
    parser.add_argument("--checkpoint", default="models/best.pth")
    parser.add_argument("--checkpoint_url", default="")
    parser.add_argument("--checkpoint_sha256", default="")
    parser.add_argument("--short_size", type=int, default=512)
    parser.add_argument("--max_size", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--score_threshold", type=float, default=0.05)
    parser.add_argument("--score_cls_power", type=float, default=0.5)
    parser.add_argument("--score_centerness_power", type=float, default=0.5)
    parser.add_argument("--nms_threshold", type=float, default=0.55)
    parser.add_argument("--pre_nms_topk", type=int, default=1000)
    parser.add_argument("--max_detections_per_image", type=int, default=100)
    parser.add_argument("--tta_flip", action="store_true", help="Merge original and horizontal-flip predictions.")
    parser.add_argument(
        "--tta_merge_strategy",
        choices=["nms", "consensus"],
        default="nms",
        help="How to merge original and horizontal-flip predictions.",
    )
    parser.add_argument("--tta_match_iou", type=float, default=0.6)
    parser.add_argument("--tta_single_view_score_factor", type=float, default=0.6)
    parser.add_argument("--tta_matched_score_factor", type=float, default=1.0)
    parser.add_argument("--chair_score_threshold", type=float, default=-1.0)
    parser.add_argument("--chair_topk", type=int, default=0)
    parser.add_argument(
        "--class_score_thresholds",
        default="",
        help="JSON object or comma-separated class=threshold values for per-class filtering.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return apply_config_defaults(parser)


def main() -> None:
    args = parse_args()
    require_args(args, ["image_dir", "output"])
    device = torch.device(args.device)
    checkpoint_path = resolve_checkpoint_path(
        args.checkpoint,
        script_dir=Path(__file__).resolve().parent,
        checkpoint_url=args.checkpoint_url,
        checkpoint_sha256=args.checkpoint_sha256,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint.get("class_names")
    if class_names:
        set_active_classes(class_names)
    num_classes = len(class_names) if class_names else 5
    checkpoint_args = checkpoint.get("args", {}) or {}
    model = build_detector(
        num_classes=num_classes,
        pretrained_backbone=False,
        backbone_name=checkpoint_args.get("backbone_name", "convnext_tiny"),
        fpn_type=checkpoint_args.get("fpn_type", "fpn"),
        bifpn_layers=int(checkpoint_args.get("bifpn_layers", 1)),
        use_p1=bool(checkpoint_args.get("use_p1", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    transform = DetectionTransform(
        train=False,
        short_size=args.short_size,
        max_size=args.max_size,
        horizontal_flip_prob=0.0,
        color_jitter=0.0,
    )
    class_score_thresholds = parse_class_score_thresholds(args.class_score_thresholds)
    if args.chair_score_threshold >= 0:
        class_score_thresholds["chair"] = args.chair_score_threshold
    if args.tile_inference:
        dataset = TiledImageDataset(
            args.image_dir,
            transform=transform,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
        )
    else:
        dataset = ImageFolderDataset(args.image_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=detection_collate_fn,
    )

    raw_predictions = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="predict"):
            images = batch["images"].to(device)
            outputs = model(images)
            image_sizes = [tuple(int(v) for v in target["resized_size"].tolist()) for target in batch["targets"]]
            detections = decode_detections(
                outputs,
                image_sizes=image_sizes,
                strides=model.strides,
                score_threshold=args.score_threshold,
                nms_threshold=args.nms_threshold,
                max_detections_per_image=args.max_detections_per_image,
                pre_nms_topk=args.pre_nms_topk,
                score_cls_power=args.score_cls_power,
                score_centerness_power=args.score_centerness_power,
                class_score_thresholds=class_score_thresholds,
            )
            if args.tta_flip:
                flipped_images = flip_valid_image_regions(images, image_sizes)
                flipped_outputs = model(flipped_images)
                flipped_detections = decode_detections(
                    flipped_outputs,
                    image_sizes=image_sizes,
                    strides=model.strides,
                    score_threshold=args.score_threshold,
                    nms_threshold=args.nms_threshold,
                    max_detections_per_image=args.max_detections_per_image,
                    pre_nms_topk=args.pre_nms_topk,
                    score_cls_power=args.score_cls_power,
                    score_centerness_power=args.score_centerness_power,
                    class_score_thresholds=class_score_thresholds,
                )
                if args.tta_merge_strategy == "consensus":
                    class_thresholds = dict(class_score_thresholds)

                    class_topk = {}
                    if args.chair_topk > 0:
                        class_topk["chair"] = args.chair_topk
                    detections = [
                        merge_detections_consensus(
                            primary_detections=det,
                            secondary_detections=flip_detections_horizontal(flip_det, image_size[1]),
                            image_size=image_size,
                            match_iou_threshold=args.tta_match_iou,
                            nms_threshold=args.nms_threshold,
                            max_detections_per_image=args.max_detections_per_image,
                            single_view_score_factor=args.tta_single_view_score_factor,
                            matched_score_factor=args.tta_matched_score_factor,
                            class_score_thresholds=class_thresholds,
                            class_topk=class_topk,
                        )
                        for det, flip_det, image_size in zip(detections, flipped_detections, image_sizes)
                    ]
                else:
                    detections = [
                        merge_detections(
                            det + flip_detections_horizontal(flip_det, image_size[1]),
                            image_size=image_size,
                            nms_threshold=args.nms_threshold,
                            max_detections_per_image=args.max_detections_per_image,
                        )
                        for det, flip_det, image_size in zip(detections, flipped_detections, image_sizes)
                    ]
            for target, det in zip(batch["targets"], detections):
                scaled = scale_detections_to_original(
                    det,
                    resized_size=tuple(int(v) for v in target["resized_size"].tolist()),
                    original_size=tuple(int(v) for v in target["original_size"].tolist()),
                    crop_offset=tuple(
                        int(v) for v in target.get("crop_offset", torch.zeros(2, dtype=torch.int64)).tolist()
                    ),
                    crop_size=tuple(
                        int(v)
                        for v in target.get("crop_size", target["original_size"]).tolist()
                    ),
                )
                raw_predictions.append(
                    {
                        "image_id": target["image_id"],
                        "boxes": scaled,
                        "original_size": [
                            int(v) for v in target["original_size"].tolist()
                        ],
                    }
                )

    if args.tile_inference:
        grouped = group_tile_predictions(raw_predictions)
        sizes = {}
        for item in raw_predictions:
            sizes.setdefault(item["image_id"], tuple(item["original_size"]))
        predictions = [
            {
                "image_id": image_id,
                "boxes": merge_detections(
                    boxes,
                    image_size=sizes[image_id],
                    nms_threshold=args.nms_threshold,
                    max_detections_per_image=args.max_detections_per_image,
                ),
            }
            for image_id, boxes in grouped.items()
        ]
    else:
        predictions = [
            {"image_id": item["image_id"], "boxes": item["boxes"]}
            for item in raw_predictions
        ]

    Path(args.output).write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")


class ImageFolderDataset(Dataset):
    def __init__(self, image_dir: str | Path, transform: DetectionTransform) -> None:
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.paths = sorted(
            path for path in self.image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        image = Image.open(path).convert("RGB")
        width, height = image.size
        target = {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "image_id": path.name,
            "original_size": torch.tensor([height, width], dtype=torch.int64),
        }
        return self.transform(image, target)


def parse_class_score_thresholds(raw: object) -> dict[str, float]:
    if isinstance(raw, dict):
        return {str(key): float(value) for key, value in raw.items()}
    if not isinstance(raw, str) or not raw.strip():
        return {}
    if raw.lstrip().startswith("{"):
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("--class_score_thresholds JSON must be an object")
        return {str(key): float(value) for key, value in data.items()}
    result = {}
    for item in raw.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"Expected class=threshold, got: {item}")
        class_name, value = item.split("=", 1)
        result[class_name.strip()] = float(value)
    return result


def require_args(args: argparse.Namespace, names: list[str]) -> None:
    missing = [name for name in names if not getattr(args, name)]
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join('--' + name for name in missing)}")


def flip_valid_image_regions(images: torch.Tensor, image_sizes: list[tuple[int, int]]) -> torch.Tensor:
    flipped = images.clone()
    for index, (_, width) in enumerate(image_sizes):
        flipped[index, :, :, :width] = torch.flip(images[index, :, :, :width], dims=(-1,))
    return flipped


if __name__ == "__main__":
    main()
