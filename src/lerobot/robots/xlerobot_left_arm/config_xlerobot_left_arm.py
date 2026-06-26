from dataclasses import dataclass, field

from lerobot.cameras.configs import CameraConfig, Cv2Rotation
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

from ..config import RobotConfig


def xlerobot_left_arm_cameras_config() -> dict[str, CameraConfig]:
    return {
        "center": OpenCVCameraConfig(
            index_or_path="/dev/xlerobot_cam_top",
            fps=30,
            width=640,
            height=480,
            rotation=Cv2Rotation.ROTATE_180,
            fourcc="MJPG",
        ),
        "right": OpenCVCameraConfig(
            index_or_path="/dev/xlerobot_cam_right",
            fps=30,
            width=640,
            height=480,
            rotation=Cv2Rotation.ROTATE_180,
            fourcc="MJPG",
        ),
        "left": OpenCVCameraConfig(
            index_or_path="/dev/xlerobot_cam_left",
            fps=30,
            width=640,
            height=480,
            rotation=Cv2Rotation.ROTATE_180,
            fourcc="MJPG",
        ),
    }


@RobotConfig.register_subclass("xlerobot_left_arm")
@dataclass
class XLerobotLeftArmConfig(RobotConfig):
    port1: str = "/dev/xlerobot_arm_left"
    port2: str = "/dev/xlerobot_arm_right"
    disable_torque_on_disconnect: bool = True

    max_relative_target: float | dict[str, float] | None = None

    cameras: dict[str, CameraConfig] = field(default_factory=xlerobot_left_arm_cameras_config)

    use_degrees: bool = False
    reuse_full_xlerobot_calibration: bool = True
