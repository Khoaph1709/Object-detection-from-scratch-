from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="", help="Path to a JSON config file. CLI arguments override config values.")


def apply_config_defaults(parser: argparse.ArgumentParser, argv: list[str] | None = None) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[1:]

    config_parser = argparse.ArgumentParser(add_help=False)
    add_config_argument(config_parser)
    config_args, _ = config_parser.parse_known_args(argv)
    if config_args.config:
        config = read_config(config_args.config)
        parser.set_defaults(**config)

    return parser.parse_args(argv)


def read_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a JSON object: {config_path}")
    return config
