#!/usr/bin/env python
"""Run YOLO detection/tracking on the calibrated XLerobot cameras.

This script is intentionally independent from xlerobot-autonomy. It uses the
verified local camera mapping directly and is meant as the first vision test
before connecting perception output to robot motion.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class CameraSpec:
    device: str
    rotate_180: bool = True


CAMERAS: dict[str, CameraSpec] = {
    "center": CameraSpec("/dev/xlerobot_cam_top"),
    "right": CameraSpec("/dev/xlerobot_cam_right"),
    "left": CameraSpec("/dev/xlerobot_cam_left"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", choices=sorted(CAMERAS), default="center")
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics model path/name.")
    parser.add_argument(
        "--target",
        default=None,
        help="Object class to prefer, for example 'cup', 'bottle', 'person'.",
    )
    parser.add_argument(
        "--world",
        action="store_true",
        help="Use YOLO-World open-vocabulary prompts. Useful for custom objects.",
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Use Ultralytics tracking instead of frame-by-frame prediction.",
    )
    parser.add_argument("--tracker", default="botsort.yaml", help="Tracker config for --track.")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument("--frames", type=int, default=0, help="Stop after N frames. 0 means run until q/Ctrl-C.")
    parser.add_argument("--save-dir", default="outputs/xlerobot_yolo")
    parser.add_argument("--save-every", type=int, default=30, help="Save one annotated frame every N frames.")
    parser.add_argument("--no-display", action="store_true", help="Do not open an OpenCV window.")
    return parser.parse_args()


def open_camera(spec: CameraSpec, args: argparse.Namespace) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(spec.device, cv2.CAP_V4L2)
    if args.fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {spec.device}")

    for _ in range(5):
        cap.read()
        time.sleep(0.03)
    return cap


def load_model(args: argparse.Namespace):
    try:
        from ultralytics import YOLO, YOLOWorld
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: ultralytics. Install it in the xlerobot conda env, "
            "for example: pip install ultralytics"
        ) from exc

    model_cls = YOLOWorld if args.world else YOLO
    model = model_cls(args.model)
    if args.world and args.target:
        model.set_classes([args.target])
    return model


def target_class_indices(model, target: str | None) -> list[int] | None:
    if not target:
        return None

    names = getattr(model, "names", None)
    if not isinstance(names, dict):
        return None

    matches = [idx for idx, name in names.items() if str(name).lower() == target.lower()]
    if not matches:
        print(
            f"Target '{target}' is not in this model's class list. "
            "Use --world for open-vocabulary detection or train a custom model."
        )
        return []
    return matches


def best_detection(result, target: str | None):
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None

    names = getattr(result, "names", {})
    candidates = []
    for box in boxes:
        cls_id = int(box.cls[0])
        label = str(names.get(cls_id, cls_id))
        if target and label.lower() != target.lower():
            continue
        conf = float(box.conf[0])
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        candidates.append((conf, label, x1, y1, x2, y2, getattr(box, "id", None)))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])


def print_detection(frame_index: int, frame_shape, detection) -> None:
    if detection is None:
        if frame_index % 15 == 0:
            print(f"frame={frame_index} detected=False")
        return

    conf, label, x1, y1, x2, y2, track_id_tensor = detection
    height, width = frame_shape[:2]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    dx = (cx - width / 2.0) / (width / 2.0)
    dy = (cy - height / 2.0) / (height / 2.0)
    area_ratio = ((x2 - x1) * (y2 - y1)) / float(width * height)

    track_id = None
    if track_id_tensor is not None:
        try:
            track_id = int(track_id_tensor[0])
        except TypeError:
            track_id = int(track_id_tensor)

    track_part = f" track_id={track_id}" if track_id is not None else ""
    print(
        f"frame={frame_index} detected=True label={label} conf={conf:.2f}{track_part} "
        f"bbox=({int(x1)},{int(y1)},{int(x2 - x1)},{int(y2 - y1)}) "
        f"center_error=({dx:.3f},{dy:.3f}) area={area_ratio:.3f}"
    )


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args)
    class_indices = None if args.world else target_class_indices(model, args.target)
    if class_indices == []:
        return

    spec = CAMERAS[args.camera]
    cap = open_camera(spec, args)
    print(
        f"camera={args.camera} device={spec.device} model={args.model} "
        f"target={args.target or 'any'} track={args.track}"
    )

    frame_index = 0
    try:
        while args.frames <= 0 or frame_index < args.frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"Failed to read frame from {spec.device}")
            if spec.rotate_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            if args.track:
                results = model.track(
                    frame,
                    persist=True,
                    tracker=args.tracker,
                    conf=args.conf,
                    iou=args.iou,
                    classes=class_indices,
                    verbose=False,
                )
            else:
                results = model.predict(
                    frame,
                    conf=args.conf,
                    iou=args.iou,
                    classes=class_indices,
                    verbose=False,
                )

            result = results[0]
            detection = best_detection(result, args.target)
            print_detection(frame_index, frame.shape, detection)

            annotated = result.plot()
            if args.save_every > 0 and frame_index % args.save_every == 0:
                cv2.imwrite(str(save_dir / f"{args.camera}_{frame_index:06d}.jpg"), annotated)

            if not args.no_display:
                cv2.imshow("XLerobot YOLO", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1
    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
