#!/usr/bin/env python3
"""Plot LeRobot training curves from a local training log.

Example:

    uv run python plot_training_curves.py

The parser avoids relying on compact log step labels such as "2K", because
multiple log lines can share the same rounded label. Instead it uses log order
and --log-freq to reconstruct x-axis steps.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


DEFAULT_LOG_DIR = Path("/Users/tony/robot_dev/train_logs")
DEFAULT_OUT = Path("outputs/train/mac_xlerobot_right_arm_smolvla_2k/training_curves.png")


def default_log_path() -> Path:
    logs = sorted(DEFAULT_LOG_DIR.glob("*.log"), key=lambda path: path.stat().st_mtime, reverse=True)
    if logs:
        return logs[0]
    return DEFAULT_LOG_DIR / "xlerobot_right_arm_train1_smolvla.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot loss, grad norm, LR, and throughput from a train log.")
    default_log = default_log_path()
    parser.add_argument(
        "--log",
        type=Path,
        default=default_log,
        help=f"Training log path. Default: newest .log under {DEFAULT_LOG_DIR} ({default_log}).",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"Output PNG path. Default: {DEFAULT_OUT}")
    parser.add_argument(
        "--log-freq",
        type=int,
        default=50,
        help="Training log frequency used by the run. Default: 50.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Moving-average window for loss and grad norm. Default: 5.",
    )
    return parser.parse_args()


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values

    averaged = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        chunk = values[start : idx + 1]
        averaged.append(sum(chunk) / len(chunk))
    return averaged


def parse_log(log_path: Path, log_freq: int) -> list[dict[str, float]]:
    text = log_path.read_text(errors="ignore")
    pattern = re.compile(
        r"epch:(?P<epch>[0-9.]+).*?"
        r"loss:(?P<loss>[0-9.]+).*?"
        r"grdn:(?P<grdn>[0-9.]+).*?"
        r"lr:(?P<lr>[0-9.e-]+).*?"
        r"updt_s:(?P<updt>[0-9.]+).*?"
        r"data_s:(?P<data>[0-9.]+).*?"
        r"smp/s:(?P<sps>[0-9.]+)"
    )

    rows = []
    for idx, match in enumerate(pattern.finditer(text), start=1):
        rows.append(
            {
                "step": float(idx * log_freq),
                "epch": float(match["epch"]),
                "loss": float(match["loss"]),
                "grdn": float(match["grdn"]),
                "lr": float(match["lr"]),
                "updt": float(match["updt"]),
                "data": float(match["data"]),
                "sps": float(match["sps"]),
            }
        )
    return rows


def plot(rows: list[dict[str, float]], output_path: Path, smooth_window: int) -> None:
    import matplotlib.pyplot as plt

    steps = [row["step"] for row in rows]
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)

    plots = [
        ("loss", "Loss", True),
        ("grdn", "Grad norm", True),
        ("lr", "Learning rate", False),
        ("epch", "Epoch", False),
        ("sps", "Samples/sec", False),
        ("data", "Data loading sec", False),
    ]

    for axis, (key, title, smooth) in zip(axes.ravel(), plots):
        values = [row[key] for row in rows]
        axis.plot(steps, values, marker="o", alpha=0.35, label="raw")
        if smooth and len(values) >= 3:
            axis.plot(steps, moving_average(values, smooth_window), linewidth=2, label="moving avg")
            axis.legend()
        axis.set_title(title)
        axis.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("step")
    axes[-1, 1].set_xlabel("step")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)


def main() -> None:
    args = parse_args()
    log_path = args.log.expanduser().resolve()
    output_path = args.out.expanduser()

    if not log_path.exists():
        raise FileNotFoundError(f"Log file does not exist: {log_path}")

    rows = parse_log(log_path, args.log_freq)
    if not rows:
        raise RuntimeError(f"No training metric lines found in: {log_path}")

    plot(rows, output_path, args.smooth_window)
    print(f"Parsed {len(rows)} metric points from {log_path}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
