import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_xlerobot_left_arm import XLerobotLeftArmConfig

logger = logging.getLogger(__name__)


class XLerobotLeftArm(Robot):
    config_class = XLerobotLeftArmConfig
    name = "xlerobot_left_arm"

    def __init__(self, config: XLerobotLeftArmConfig):
        super().__init__(config)
        self.config = config
        self._load_full_xlerobot_calibration_if_needed()

        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = FeetechMotorsBus(
            port=self.config.port1,
            motors={
                "left_arm_shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "left_arm_shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "left_arm_elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "left_arm_wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "left_arm_wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "left_arm_gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self._bus_calibration(),
        )
        self.left_arm_motors = list(self.bus.motors)
        self.cameras = make_cameras_from_configs(config.cameras)

    def _load_full_xlerobot_calibration_if_needed(self) -> None:
        if self.calibration or not self.config.reuse_full_xlerobot_calibration or self.id is None:
            return
        full_calibration_path = self.calibration_dir.parent / "xlerobot" / f"{self.id}.json"
        if full_calibration_path.is_file():
            self._load_calibration(full_calibration_path)
            logger.info("Loaded full XLeRobot calibration from %s", full_calibration_path)

    def _bus_calibration(self) -> dict[str, MotorCalibration]:
        return {
            name: calibration
            for name, calibration in self.calibration.items()
            if name.startswith("left_arm")
        }

    @property
    def _state_ft(self) -> dict[str, type]:
        return dict.fromkeys((f"{motor}.pos" for motor in self.left_arm_motors), float)

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._state_ft

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        try:
            self.bus.connect()
            if self.calibration:
                self.bus.calibration = self._bus_calibration()
                if calibrate:
                    user_input = input(
                        f"Press ENTER to use provided calibration file associated with the id {self.id}, "
                        "or type 'CALIBRATE' and press ENTER to run calibration: "
                    )
                    if user_input.strip().lower() == "calibrate":
                        self.calibrate()
                    else:
                        self.bus.write_calibration(self.bus.calibration)
                else:
                    self.bus.write_calibration(self.bus.calibration)
            elif calibrate:
                self.calibrate()

            for cam in self.cameras.values():
                cam.connect()

            self.configure()
            logger.info("%s connected.", self)
        except Exception:
            self._disconnect_partial()
            raise

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        logger.info("\nRunning left-arm calibration of %s", self)
        self.bus.disable_torque(self.left_arm_motors)
        for name in self.left_arm_motors:
            self.bus.write("Operating_Mode", name, OperatingMode.POSITION.value)

        input("Move left arm motors to the middle of their range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings(self.left_arm_motors)

        full_turn_motors = [motor for motor in self.left_arm_motors if motor.endswith("wrist_roll")]
        unknown_range_motors = [motor for motor in self.left_arm_motors if motor not in full_turn_motors]
        print(
            f"Move all left arm joints except '{full_turn_motors}' sequentially through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)
        for name in full_turn_motors:
            range_mins[name] = 0
            range_maxes[name] = 4095

        calibration = {}
        for name, motor in self.bus.motors.items():
            calibration[name] = MotorCalibration(
                id=motor.id,
                drive_mode=0,
                homing_offset=homing_offsets[name],
                range_min=range_mins[name],
                range_max=range_maxes[name],
            )

        self.bus.write_calibration(calibration)
        self.calibration = calibration
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    def configure(self) -> None:
        self.bus.disable_torque()
        self.bus.configure_motors()

        for name in self.left_arm_motors:
            self.bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
            self.bus.write("P_Coefficient", name, 16)
            self.bus.write("I_Coefficient", name, 0)
            self.bus.write("D_Coefficient", name, 32)

        self.bus.enable_torque()

    def setup_motors(self) -> None:
        for motor in reversed(self.left_arm_motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        left_arm_pos = self.bus.sync_read("Present_Position", self.left_arm_motors)
        left_arm_state = {f"{k}.pos": v for k, v in left_arm_pos.items()}
        logger.debug("%s read state: %.1fms", self, (time.perf_counter() - start) * 1e3)

        return {**left_arm_state, **self.get_camera_observation()}

    def get_camera_observation(self) -> dict[str, Any]:
        obs_dict = {}
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            logger.debug("%s read %s: %.1fms", self, cam_key, (time.perf_counter() - start) * 1e3)
        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        left_arm_pos = {
            key: value
            for key, value in action.items()
            if key.startswith("left_arm_") and key.endswith(".pos")
        }
        if self.config.max_relative_target is not None and left_arm_pos:
            present_pos = self.bus.sync_read("Present_Position", self.left_arm_motors)
            goal_present_pos = {
                key: (goal_pos, present_pos[key.removesuffix(".pos")])
                for key, goal_pos in left_arm_pos.items()
            }
            left_arm_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        left_arm_pos_raw = {key.removesuffix(".pos"): value for key, value in left_arm_pos.items()}
        if left_arm_pos_raw:
            self.bus.sync_write("Goal_Position", left_arm_pos_raw)

        return left_arm_pos

    def disconnect(self) -> None:
        if not self.bus.is_connected and not any(cam.is_connected for cam in self.cameras.values()):
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self._disconnect_partial()
        logger.info("%s disconnected.", self)

    def _disconnect_partial(self) -> None:
        if self.bus.is_connected:
            try:
                self.bus.port_handler.clearPort()
                self.bus.port_handler.is_using = False
            except Exception as exc:
                logger.warning("Could not clear left-arm bus during disconnect: %s", exc)
            try:
                self.bus.disconnect(self.config.disable_torque_on_disconnect)
            except Exception as exc:
                logger.warning("Could not fully disconnect left-arm bus: %s", exc)
                self.bus.port_handler.is_using = False
                self.bus.port_handler.closePort()

        for cam in self.cameras.values():
            if cam.is_connected:
                cam.disconnect()
