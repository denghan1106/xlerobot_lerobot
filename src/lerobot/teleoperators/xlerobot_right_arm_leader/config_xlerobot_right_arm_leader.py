from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("xlerobot_right_arm_leader")
@dataclass
class XLerobotRightArmLeaderConfig(TeleoperatorConfig):
    port: str
    use_degrees: bool = True
