from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("xlerobot_left_arm_leader")
@dataclass
class XLerobotLeftArmLeaderConfig(TeleoperatorConfig):
    port: str = "/dev/xlerobot_leader_left"
    use_degrees: bool = False
    align_to_robot_on_start: bool = True
    reuse_full_xlerobot_calibration: bool = False
    xlerobot_calibration_id: str = "xlerobot"
