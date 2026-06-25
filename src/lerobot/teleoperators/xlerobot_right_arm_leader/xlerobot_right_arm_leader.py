from typing import Any

from lerobot.teleoperators.so_leader.so_leader import SOLeader

from .config_xlerobot_right_arm_leader import XLerobotRightArmLeaderConfig


_JOINT_MAP = {
    "shoulder_pan.pos": "right_arm_shoulder_pan.pos",
    "shoulder_lift.pos": "right_arm_shoulder_lift.pos",
    "elbow_flex.pos": "right_arm_elbow_flex.pos",
    "wrist_flex.pos": "right_arm_wrist_flex.pos",
    "wrist_roll.pos": "right_arm_wrist_roll.pos",
    "gripper.pos": "right_arm_gripper.pos",
}
_INVERSE_JOINT_MAP = {value: key for key, value in _JOINT_MAP.items()}


class XLerobotRightArmLeader(SOLeader):
    config_class = XLerobotRightArmLeaderConfig
    # Reuse the existing SO leader calibration directory for the same physical leader arm.
    name = "so_leader"

    @property
    def action_features(self) -> dict[str, type]:
        return {mapped_name: float for mapped_name in _JOINT_MAP.values()}

    @property
    def feedback_features(self) -> dict[str, type]:
        return self.action_features

    def get_action(self) -> dict[str, float]:
        action = super().get_action()
        return {_JOINT_MAP[key]: value for key, value in action.items()}

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        unprefixed_feedback = {
            _INVERSE_JOINT_MAP[key]: value for key, value in feedback.items() if key in _INVERSE_JOINT_MAP
        }
        if unprefixed_feedback:
            super().send_feedback(unprefixed_feedback)
