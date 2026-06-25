#!/usr/bin/env python3
"""Calibrate the complete three-wheel XLeRobot without starting teleoperation."""

from lerobot.robots.xlerobot import XLerobot, XLerobotConfig


ROBOT_ID = "xlerobot"
LEFT_HEAD_PORT = "/dev/ttyACM1"
RIGHT_BASE_PORT = "/dev/ttyACM0"


def main() -> None:
    print("Complete XLeRobot calibration")
    print(f"  port1 (left arm + head): {LEFT_HEAD_PORT}")
    print(f"  port2 (right arm + base): {RIGHT_BASE_PORT}")
    print("This recalibrates both arms and the two head motors.")
    print("The three base wheels must remain lifted off the ground.")
    confirmation = input("Type CALIBRATE to continue: ").strip()
    if confirmation != "CALIBRATE":
        print("Cancelled without connecting to the robot.")
        return

    config = XLerobotConfig(
        id=ROBOT_ID,
        port1=LEFT_HEAD_PORT,
        port2=RIGHT_BASE_PORT,
    )
    robot = XLerobot(config)
    zero_base_velocity = dict.fromkeys(robot.base_motors, 0)

    try:
        robot.bus1.connect()
        robot.bus2.connect()

        # Enter calibration with every actuator torque-disabled and no pending
        # wheel velocity command. This avoids starting the full robot control path.
        robot.bus1.disable_torque()
        robot.bus2.disable_torque()
        robot.bus2.sync_write("Goal_Velocity", zero_base_velocity, normalize=False)

        robot.calibrate()
    finally:
        if robot.bus2.is_connected:
            try:
                robot.bus2.sync_write("Goal_Velocity", zero_base_velocity, normalize=False)
            finally:
                robot.bus2.disconnect(disable_torque=True)
        if robot.bus1.is_connected:
            robot.bus1.disconnect(disable_torque=True)


if __name__ == "__main__":
    main()
