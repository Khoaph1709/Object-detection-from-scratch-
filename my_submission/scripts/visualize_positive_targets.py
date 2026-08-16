from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from my_submission.models.detector import build_detector
from my_submission.utils.assigner import FCOSTargetAssigner
from my_submission.utils.augmentations import DetectionTransform
from my_submission.utils.dataset import DetectionDataset, detection_collate_fn
from my_submission.utils.locations import generate_locations
from my_submission.utils.visualization import draw_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize FCOS positive target locations.")
    parser.add_argument("--annotation", default="indoor5-v2-student/public/annotations/train.json")
    parser.add_argument("--image_root", default="indoor5-v2-student/public/train/images")
    parser.add_argument("--output", default="my_submission/artifacts/positive_targets/positive_targets.jpg")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--short_size", type=int, default=512)
    parser.add_argument("--max_size", type=int, default=768)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transform = DetectionTransform(
        train=False,
        short_size=args.short_size,
        max_size=args.max_size,
        horizontal_flip_prob=0.0,
        color_jitter=0.0,
    )
    dataset = DetectionDataset(args.annotation, args.image_root, transform=transform)
    image, target = dataset[args.index]
    batch = detection_collate_fn([(image, target)])

    model = build_detector(pretrained_backbone=False)
    model.eval()
    with torch.no_grad():
        outputs = model(batch["images"])
        locations_by_level = generate_locations(outputs["features"], model.strides)
        assigned = FCOSTargetAssigner(model.strides)(locations_by_level, batch["targets"])

    pil_image = draw_target(image, target, normalized=True)
    draw = ImageDraw.Draw(pil_image)
    labels = assigned["labels"][0]
    locations = assigned["locations"]
    positives = locations[labels >= 0]
    for x, y in positives.tolist():
        r = 2
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(37, 99, 235))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pil_image.save(output)
    print(f"Saved positive target visualization to {output}")
    print(f"num_positive={int((labels >= 0).sum())}")


if __name__ == "__main__":
    main()

