#!/usr/bin/env python3
"""Aggregate XLeRobot train1 recording shards into one LeRobot dataset.

Typical usage from the repository root:

    uv run python aggregate_train1.py

On macOS with this workspace layout, the defaults resolve to:

    input:  /Users/tony/robot_dev/traindata/train1
    output: /Users/tony/robot_dev/traindata/xlerobot_right_arm_train1

On the AGX layout, pass explicit paths if needed:

    uv run python aggregate_train1.py \
      --input-dir /data/cdzhitu/robot_dev/traindata/train1 \
      --output-dir /data/cdzhitu/robot_dev/traindata/xlerobot_right_arm_train1
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))


DEFAULT_REPO_ID = "cdzhitu/xlerobot_right_arm_train1"


def default_input_dir() -> Path:
    local_train1 = REPO_ROOT.parent / "traindata" / "train1"
    if local_train1.exists():
        return local_train1

    agx_train1 = Path("/data/cdzhitu/robot_dev/traindata/train1")
    if agx_train1.exists():
        return agx_train1

    return local_train1


def default_output_dir(input_dir: Path) -> Path:
    return input_dir.parent / "xlerobot_right_arm_train1"


def read_info(dataset_dir: Path) -> dict:
    info_path = dataset_dir / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def discover_dataset_roots(input_dir: Path, pattern: str, include_empty: bool) -> list[Path]:
    roots: list[Path] = []
    skipped_empty: list[Path] = []
    skipped_missing_info: list[Path] = []

    for dataset_dir in sorted(input_dir.glob(pattern)):
        if not dataset_dir.is_dir():
            continue

        info_path = dataset_dir / "meta" / "info.json"
        if not info_path.exists():
            skipped_missing_info.append(dataset_dir)
            continue

        info = read_info(dataset_dir)
        total_episodes = int(info.get("total_episodes") or 0)
        if total_episodes <= 0 and not include_empty:
            skipped_empty.append(dataset_dir)
            continue

        roots.append(dataset_dir)

    if skipped_missing_info:
        print("Skipped directories without meta/info.json:")
        for path in skipped_missing_info:
            print(f"  - {path}")

    if skipped_empty:
        print("Skipped empty datasets:")
        for path in skipped_empty:
            print(f"  - {path.name}")

    return roots


def summarize_roots(roots: list[Path]) -> tuple[int, int]:
    total_episodes = 0
    total_frames = 0

    print("Datasets to aggregate:")
    for root in roots:
        info = read_info(root)
        episodes = int(info.get("total_episodes") or 0)
        frames = int(info.get("total_frames") or 0)
        robot_type = info.get("robot_type")
        fps = info.get("fps")
        total_episodes += episodes
        total_frames += frames
        print(f"  - {root.name}: episodes={episodes}, frames={frames}, robot_type={robot_type}, fps={fps}")

    print(f"Total: datasets={len(roots)}, episodes={total_episodes}, frames={total_frames}")
    return total_episodes, total_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate train1 XLeRobot recording directories into one LeRobot dataset."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input_dir(),
        help="Directory containing xlerobot_right_arm_record_* dataset folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for the aggregated dataset. Defaults to INPUT_DIR/../xlerobot_right_arm_train1.",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"repo_id written into the aggregated dataset metadata. Default: {DEFAULT_REPO_ID}",
    )
    parser.add_argument(
        "--pattern",
        default="xlerobot_right_arm_record_*",
        help="Glob pattern used to select dataset folders under input-dir.",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Include datasets with total_episodes <= 0. By default they are skipped.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output-dir before aggregation if it already exists.",
    )
    parser.add_argument(
        "--concatenate-videos",
        action="store_true",
        help="Pack source videos into larger output shards. Default keeps source files separate.",
    )
    parser.add_argument(
        "--concatenate-data",
        action="store_true",
        help="Pack source parquet files into larger output shards. Default keeps source files separate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve() if args.output_dir else default_output_dir(input_dir).resolve()
    )

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    roots = discover_dataset_roots(input_dir, args.pattern, args.include_empty)
    if not roots:
        raise RuntimeError(f"No datasets matched {args.pattern!r} under {input_dir}")

    summarize_roots(roots)
    print(f"Output: {output_dir}")
    print(f"Aggregated repo_id: {args.repo_id}")

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}\n"
                "Use --overwrite to delete it before aggregation, or choose --output-dir."
            )
        shutil.rmtree(output_dir)

    from lerobot.datasets.aggregate import aggregate_datasets

    aggregate_datasets(
        repo_ids=[root.name for root in roots],
        roots=roots,
        aggr_repo_id=args.repo_id,
        aggr_root=output_dir,
        concatenate_videos=args.concatenate_videos,
        concatenate_data=args.concatenate_data,
    )

    print("Aggregation complete.")
    print(f"Use for training with: --dataset.repo_id={args.repo_id} --dataset.root={output_dir}")


if __name__ == "__main__":
    main()
