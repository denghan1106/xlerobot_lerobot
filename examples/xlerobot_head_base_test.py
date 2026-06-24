#!/usr/bin/env python3
"""Small, isolated XLeRobot head/base hardware test.

This script never instantiates the full XLeRobot. Head mode only addresses
motor IDs 7 and 8 on the left bus. Base mode only addresses motor IDs 7, 8,
and 9 on the right bus.
"""

import argparse
import time

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode


LEFT_PORT = "/dev/tty.usbmodem5AE60816781"
RIGHT_PORT = "/dev/tty.usbmodem5AE60821991"

HEAD_STEP_TICKS = 50
HEAD_MAX_OFFSET_TICKS = 100
BASE_SPEED_TICKS = 500
BASE_PULSE_SECONDS = 0.5
BASE_FEEDBACK_DELAY_SECONDS = 0.1


def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def make_head_bus(port: str) -> FeetechMotorsBus:
    return FeetechMotorsBus(
        port=port,
        motors={
            "head_motor_1": Motor(7, "sts3215", MotorNormMode.DEGREES),
            "head_motor_2": Motor(8, "sts3215", MotorNormMode.DEGREES),
        },
    )


def make_base_bus(port: str) -> FeetechMotorsBus:
    return FeetechMotorsBus(
        port=port,
        motors={
            "base_left_wheel": Motor(7, "sts3215", MotorNormMode.RANGE_M100_100),
            "base_back_wheel": Motor(8, "sts3215", MotorNormMode.RANGE_M100_100),
            "base_right_wheel": Motor(9, "sts3215", MotorNormMode.RANGE_M100_100),
        },
    )


def test_head(port: str) -> None:
    bus = make_head_bus(port)
    motors = list(bus.motors)
    initial_positions: dict[str, int] = {}

    try:
        bus.connect()
        bus.disable_torque(motors)
        for motor in motors:
            bus.write("Operating_Mode", motor, OperatingMode.POSITION.value, normalize=False)

        initial_positions = {
            motor: int(value)
            for motor, value in bus.sync_read("Present_Position", motors, normalize=False).items()
        }
        position_limits = {
            motor: (
                int(bus.read("Min_Position_Limit", motor, normalize=False)),
                int(bus.read("Max_Position_Limit", motor, normalize=False)),
            )
            for motor in motors
        }
        targets = initial_positions.copy()

        # Match each goal to the present position before torque is enabled.
        bus.sync_write("Goal_Position", targets, normalize=False)
        bus.enable_torque(motors)

        print(f"Connected head bus on {port}")
        print(f"Initial raw positions: {initial_positions}")
        print("Commands (press Enter after each command):")
        print("  1+ / 1- : head_motor_1 (ID 7) +/- 50 ticks")
        print("  2+ / 2- : head_motor_2 (ID 8) +/- 50 ticks")
        print("  r        : return both motors to their startup positions")
        print("  q        : return to startup positions and quit")
        print("Each motor is restricted to +/- 100 ticks from its startup position.")

        commands = {
            "1+": ("head_motor_1", HEAD_STEP_TICKS),
            "1-": ("head_motor_1", -HEAD_STEP_TICKS),
            "2+": ("head_motor_2", HEAD_STEP_TICKS),
            "2-": ("head_motor_2", -HEAD_STEP_TICKS),
        }

        while True:
            command = input("head> ").strip().lower()
            if command == "q":
                break
            if command == "r":
                targets = initial_positions.copy()
                bus.sync_write("Goal_Position", targets, normalize=False)
                time.sleep(0.3)
                print(f"Targets: {targets}")
                continue
            if command not in commands:
                print("Unknown command. Use 1+, 1-, 2+, 2-, r, or q.")
                continue

            motor, delta = commands[command]
            motor_min, motor_max = position_limits[motor]
            safe_min = max(motor_min, initial_positions[motor] - HEAD_MAX_OFFSET_TICKS)
            safe_max = min(motor_max, initial_positions[motor] + HEAD_MAX_OFFSET_TICKS)
            targets[motor] = clamp(targets[motor] + delta, safe_min, safe_max)
            bus.write("Goal_Position", motor, targets[motor], normalize=False)
            time.sleep(0.3)
            actual = int(bus.read("Present_Position", motor, normalize=False))
            print(f"{motor}: target={targets[motor]} ticks, actual={actual} ticks")
    finally:
        if bus.is_connected:
            if initial_positions:
                try:
                    bus.sync_write("Goal_Position", initial_positions, normalize=False)
                    time.sleep(0.3)
                except Exception as exc:
                    print(f"Warning: could not return head to startup positions: {exc}")
            bus.disconnect(disable_torque=True)


def test_base(port: str) -> None:
    confirmation = input("Lift all three wheels off the ground, then type LIFTED: ").strip()
    if confirmation != "LIFTED":
        print("Cancelled without connecting to the base bus.")
        return

    bus = make_base_bus(port)
    motors = list(bus.motors)
    zero_velocity = dict.fromkeys(motors, 0)

    try:
        bus.connect()
        bus.disable_torque(motors)
        bus.configure_motors()
        for motor in motors:
            bus.write("Operating_Mode", motor, OperatingMode.VELOCITY.value, normalize=False)
        bus.sync_write("Goal_Velocity", zero_velocity, normalize=False)
        bus.enable_torque(motors)

        print(f"Connected base bus on {port}")
        diagnostics = {
            motor: {
                "mode": bus.read("Operating_Mode", motor, normalize=False),
                "torque": bus.read("Torque_Enable", motor, normalize=False),
                "voltage": bus.read("Present_Voltage", motor, normalize=False),
            }
            for motor in motors
        }
        print(f"Motor diagnostics: {diagnostics}")
        print("Each command runs one wheel at raw speed 500 for 0.5 seconds, then sends zero.")
        print("Commands: l+ l- (left/ID7), b+ b- (back/ID8), r+ r- (right/ID9), q")

        commands = {
            "l+": ("base_left_wheel", BASE_SPEED_TICKS),
            "l-": ("base_left_wheel", -BASE_SPEED_TICKS),
            "b+": ("base_back_wheel", BASE_SPEED_TICKS),
            "b-": ("base_back_wheel", -BASE_SPEED_TICKS),
            "r+": ("base_right_wheel", BASE_SPEED_TICKS),
            "r-": ("base_right_wheel", -BASE_SPEED_TICKS),
        }

        while True:
            command = input("base> ").strip().lower()
            if command == "q":
                break
            if command not in commands:
                print("Unknown command. Use l+/l-, b+/b-, r+/r-, or q.")
                continue

            motor, velocity = commands[command]
            pulse = zero_velocity.copy()
            pulse[motor] = velocity
            actual_during_pulse = 0
            position_before = int(bus.read("Present_Position", motor, normalize=False))
            try:
                bus.sync_write("Goal_Velocity", pulse, normalize=False)
                time.sleep(BASE_FEEDBACK_DELAY_SECONDS)
                actual_during_pulse = bus.read("Present_Velocity", motor, normalize=False)
                time.sleep(BASE_PULSE_SECONDS - BASE_FEEDBACK_DELAY_SECONDS)
            finally:
                bus.sync_write("Goal_Velocity", zero_velocity, normalize=False)
            position_after = int(bus.read("Present_Position", motor, normalize=False))
            position_delta = (position_after - position_before + 2048) % 4096 - 2048
            present_current = bus.read("Present_Current", motor, normalize=False)
            status = bus.read("Status", motor, normalize=False)
            print(
                f"{motor}: commanded={velocity}, velocity={actual_during_pulse}, "
                f"position_delta={position_delta}, current={present_current}, status={status}, stopped"
            )
    finally:
        if bus.is_connected:
            try:
                bus.sync_write("Goal_Velocity", zero_velocity, normalize=False)
            finally:
                bus.disconnect(disable_torque=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("device", choices=("head", "base"))
    parser.add_argument("--port", help="Override the default USB port for the selected device")
    args = parser.parse_args()

    if args.device == "head":
        test_head(args.port or LEFT_PORT)
    else:
        test_base(args.port or RIGHT_PORT)


if __name__ == "__main__":
    main()
