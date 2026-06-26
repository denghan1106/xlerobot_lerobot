from typing import Any

from lerobot.teleoperators.so_leader.so_leader import SOLeader
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS

from .config_xlerobot_left_arm_leader import XLerobotLeftArmLeaderConfig


_JOINT_MAP = {
    "shoulder_pan.pos": "left_arm_shoulder_pan.pos",
    "shoulder_lift.pos": "left_arm_shoulder_lift.pos",
    "elbow_flex.pos": "left_arm_elbow_flex.pos",
    "wrist_flex.pos": "left_arm_wrist_flex.pos",
    "wrist_roll.pos": "left_arm_wrist_roll.pos",
    "gripper.pos": "left_arm_gripper.pos",
}
_INVERSE_JOINT_MAP = {value: key for key, value in _JOINT_MAP.items()}
_LEADER_DEBUG_MAP = _JOINT_MAP


class XLerobotLeftArmLeader(SOLeader):
    config_class = XLerobotLeftArmLeaderConfig
    name = "so_leader"

    def __init__(self, config: XLerobotLeftArmLeaderConfig):
        super().__init__(config)
        self._load_full_xlerobot_calibration_if_needed("right_arm")
        self._alignment_offsets: dict[str, float] | None = None
        self._last_leader_action: dict[str, float] = {}
        self._last_mapped_action: dict[str, float] = {}

    def _load_full_xlerobot_calibration_if_needed(self, arm_prefix: str) -> None:
        if not self.config.reuse_full_xlerobot_calibration:
            return

        full_calibration_path = (
            HF_LEROBOT_CALIBRATION / ROBOTS / "xlerobot" / f"{self.config.xlerobot_calibration_id}.json"
        )
        if not full_calibration_path.is_file():
            return

        self._load_calibration(full_calibration_path)
        self.calibration = {
            "shoulder_pan": self.calibration[f"{arm_prefix}_shoulder_pan"],
            "shoulder_lift": self.calibration[f"{arm_prefix}_shoulder_lift"],
            "elbow_flex": self.calibration[f"{arm_prefix}_elbow_flex"],
            "wrist_flex": self.calibration[f"{arm_prefix}_wrist_flex"],
            "wrist_roll": self.calibration[f"{arm_prefix}_wrist_roll"],
            "gripper": self.calibration[f"{arm_prefix}_gripper"],
        }
        self.bus.calibration = self.calibration

    @property
    def action_features(self) -> dict[str, type]:
        return {mapped_name: float for mapped_name in _JOINT_MAP.values()}

    @property
    def feedback_features(self) -> dict[str, type]:
        return self.action_features

    def set_robot_reference(self, observation: dict[str, Any]) -> None:
        if not self.config.align_to_robot_on_start or self._alignment_offsets is not None:
            return

        leader_action = self._read_action()
        offsets = {}
        for leader_key, robot_key in _JOINT_MAP.items():
            if leader_key in leader_action and robot_key in observation:
                offsets[robot_key] = observation[robot_key] - leader_action[leader_key]
        self._alignment_offsets = offsets

    def _read_action(self) -> dict[str, float]:
        raw_action = self.bus.sync_read("Present_Position", normalize=False)
        action = {}
        for motor, raw_value in raw_action.items():
            calibration = self.bus.calibration[motor]
            period = self.bus.model_resolution_table[self.bus.motors[motor].model]
            min_ = calibration.range_min
            max_ = calibration.range_max
            value = raw_value

            if max_ > period - 1:
                while value < min_:
                    value += period
            elif min_ < 0:
                while value > max_:
                    value -= period

            value = min(max_, max(min_, value))
            if motor == "gripper":
                norm = ((value - min_) / (max_ - min_)) * 100
            elif self.config.use_degrees:
                mid = (min_ + max_) / 2
                norm = (value - mid) * 360 / (period - 1)
            else:
                norm = (((value - min_) / (max_ - min_)) * 200) - 100
            action[f"{motor}.pos"] = norm
        return action

    def get_action(self) -> dict[str, float]:
        action = self._read_action()
        self._last_leader_action = {
            _LEADER_DEBUG_MAP[key]: value for key, value in action.items() if key in _LEADER_DEBUG_MAP
        }
        mapped_action = {_JOINT_MAP[key]: value for key, value in action.items()}
        if self._alignment_offsets:
            mapped_action = {
                key: value + self._alignment_offsets.get(key, 0.0) for key, value in mapped_action.items()
            }
        self._last_mapped_action = mapped_action
        return mapped_action

    def get_debug_action(self) -> dict[str, dict[str, float]]:
        return {
            "leader": self._last_leader_action,
            "mapped": self._last_mapped_action,
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        unprefixed_feedback = {
            _INVERSE_JOINT_MAP[key]: value for key, value in feedback.items() if key in _INVERSE_JOINT_MAP
        }
        if unprefixed_feedback:
            super().send_feedback(unprefixed_feedback)
