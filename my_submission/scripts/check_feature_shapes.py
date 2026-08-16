from __future__ import annotations

import argparse

from my_submission.models.fpn import print_fpn_feature_shapes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print ConvNeXt-Tiny C2-C5 and FPN P2-P7 shapes.")
    parser.add_argument("--image_size", type=int, default=640)
    parser.add_argument("--batch_size", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_fpn_feature_shapes(image_size=args.image_size, batch_size=args.batch_size)


if __name__ == "__main__":
    main()

