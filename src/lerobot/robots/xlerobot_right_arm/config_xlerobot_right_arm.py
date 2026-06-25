from dataclasses import dataclass, field

from lerobot.cameras.configs import CameraConfig, Cv2Rotation
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

from ..config import RobotConfig


def xlerobot_right_arm_cameras_config() -> dict[str, CameraConfig]:
    return {
        "center": OpenCVCameraConfig(
            index_or_path="/dev/video2",
            fps=15,
            width=640,
            height=480,
            rotation=Cv2Rotation.ROTATE_180,
            fourcc="MJPG",
        ),
        "right_wrist": OpenCVCameraConfig(
            index_or_path="/dev/video0",
            fps=15,
            width=640,
            height=480,
            rotation=Cv2Rotation.ROTATE_180,
            fourcc="MJPG",
        ),
    }


@RobotConfig.register_subclass("xlerobot_right_arm")
@dataclass
class XLerobotRightArmConfig(RobotConfig):
    # Kept for command-line symmetry with the full XLeRobot config. The right-arm
    # wrapper only opens port2, where the right arm and base motors live.
    port1: str = "/dev/ttyACM1"
    port2: str = "/dev/ttyACM0"
    disable_torque_on_disconnect: bool = True

    max_relative_target: float | dict[str, float] | None = None

    cameras: dict[str, CameraConfig] = field(default_factory=xlerobot_right_arm_cameras_config)

    use_degrees: bool = False
    reuse_full_xlerobot_calibration: bool = True
