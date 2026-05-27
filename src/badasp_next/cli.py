from __future__ import annotations

import argparse
from pathlib import Path

from .config import default_pipeline_config
from .pipeline import build_stage_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="badasp-next")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Repository root used to resolve data and results paths.",
    )
    parser.add_argument(
        "--show-stages",
        action="store_true",
        help="Print the replacement pipeline stage manifest and exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = default_pipeline_config(args.project_root)
    if args.show_stages:
        for stage in build_stage_manifest(config):
            print(f"{stage['name']}: {stage['results_dir']}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())