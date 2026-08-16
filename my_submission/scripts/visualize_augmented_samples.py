from __future__ import annotations

import argparse
import random
from pathlib import Path

from torch.utils.data import DataLoader

from my_submission.utils.augmentations import DetectionTransform
from my_submission.utils.dataset import DetectionDataset, detection_collate_fn
from my_submission.utils.visualization import draw_target, save_image_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize augmented detection samples.")
    parser.add_argument("--annotation", default="indoor5-v2-student/public/annotations/train.json")
    parser.add_argument("--image_root", default="indoor5-v2-student/public/train/images")
    parser.add_argument("--output", default="my_submission/artifacts/augmented_samples/train_aug_grid.jpg")
    parser.add_argument("--num_images", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    transform = DetectionTransform(
        train=True,
        short_size=(448, 512, 576, 640),
        max_size=896,
        horizontal_flip_prob=0.5,
        color_jitter=0.2,
    )
    dataset = DetectionDataset(args.annotation, args.image_root, transform=transform)

    indices = list(range(len(dataset)))
    random.shuffle(indices)
    subset = [dataset[index] for index in indices[: args.num_images]]
    batch = detection_collate_fn(subset)

    images = [
        draw_target(image, target, normalized=True)
        for image, target in zip(batch["images"], batch["targets"])
    ]
    save_image_grid(images, Path(args.output), columns=5)
    print(f"Saved {len(images)} augmented samples to {args.output}")
    print(f"Batch images: {tuple(batch['images'].shape)}")
    print(f"Batch masks: {tuple(batch['masks'].shape)}")
    print("Boxes per image:", [int(target["boxes"].shape[0]) for target in batch["targets"]])


if __name__ == "__main__":
    main()

