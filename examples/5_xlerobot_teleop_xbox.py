# To Run on the host
'''
PYTHONPATH=src python -m lerobot.robots.xlerobot.xlerobot_host --robot.id=my_xlerobot
'''

# To Run the teleop:
'''
PYTHONPATH=src python examples/5_xlerobot_teleop_xbox.py
'''

import time
import numpy as np
import math
import pygame

from lerobot.robots.xlerobot import XLerobotConfig, XLerobot
# from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from lerobot.model.SO101Robot import SO101Kinematics

STICK_DEADZONE = 0.5
TRIGGER_DEADZONE = 0.5

# Keymaps (semantic action: controller mapping) - Intuitive human control
LEFT_KEYMAP = {
    # Left stick controls left arm XY (when not pressed)
    'x+': 'left_stick_up', 'x-': 'left_stick_down',
    'y+': 'left_stick_right', 'y-': 'left_stick_left',
    # Left stick pressed controls left arm shoulder_pan
    'shoulder_pan+': 'left_stick_pressed_right', 'shoulder_pan-': 'left_stick_pressed_left',
    # LB pressed controls left arm pitch and wrist_roll
    'pitch+': 'lb_up', 'pitch-': 'lb_down',
    'wrist_roll+': 'lb_right', 'wrist_roll-': 'lb_left',
    # Trigger closes the gripper; LB + trigger opens it.
    'gripper-': 'left_trigger',
    'gripper+': 'left_trigger_lb',
    # Head motors
    "head_motor_1+": 'x', "head_motor_1-": 'b',
    "head_motor_2+": 'a', "head_motor_2-": 'y',
}
RIGHT_KEYMAP = {
    # Right stick controls right arm XY (when not pressed)
    'x+': 'right_stick_up', 'x-': 'right_stick_down',
    'y+': 'right_stick_right', 'y-': 'right_stick_left',
    # Right stick pressed controls right arm shoulder_pan
    'shoulder_pan+': 'right_stick_pressed_right', 'shoulder_pan-': 'right_stick_pressed_left',
    # RB pressed controls right arm pitch and wrist_roll
    'pitch+': 'rb_up', 'pitch-': 'rb_down',
    'wrist_roll+': 'rb_right', 'wrist_roll-': 'rb_left',
    # Trigger closes the gripper; RB + trigger opens it.
    'gripper-': 'right_trigger',
    'gripper+': 'right_trigger_rb',
}

# Base control keymap - Only forward/backward and rotate left/right
BASE_KEYMAP = {
    'forward': 'dpad_up', 'backward': 'dpad_down',
    'rotate_left': 'dpad_left', 'rotate_right': 'dpad_right',
}

# Global reset key for all components
RESET_KEY = 'back'
QUIT_KEY = 'start'

LEFT_JOINT_MAP = {
    "shoulder_pan": "left_arm_shoulder_pan",
    "shoulder_lift": "left_arm_shoulder_lift",
    "elbow_flex": "left_arm_elbow_flex",
    "wrist_flex": "left_arm_wrist_flex",
    "wrist_roll": "left_arm_wrist_roll",
    "gripper": "left_arm_gripper",
}
RIGHT_JOINT_MAP = {
    "shoulder_pan": "right_arm_shoulder_pan",
    "shoulder_lift": "right_arm_shoulder_lift",
    "elbow_flex": "right_arm_elbow_flex",
    "wrist_flex": "right_arm_wrist_flex",
    "wrist_roll": "right_arm_wrist_roll",
    "gripper": "right_arm_gripper",
}

HEAD_MOTOR_MAP = {
    "head_motor_1": "head_motor_1",
    "head_motor_2": "head_motor_2",
}

class SimpleHeadControl:
    def __init__(self, initial_obs, kp=0.81):
        self.kp = kp
        self.degree_step = 1
        # Initialize head motor positions
        self.target_positions = {
            "head_motor_1": initial_obs.get("head_motor_1.pos", 0.0),
            "head_motor_2": initial_obs.get("head_motor_2.pos", 0.0),
        }
        self.zero_pos = {"head_motor_1": 0.0, "head_motor_2": 0.0}

    def move_to_zero_position(self, robot):
        self.target_positions = self.zero_pos.copy()
        action = self.p_control_action(robot)
        robot.send_action(action)

    def handle_keys(self, key_state):
        if key_state.get('head_motor_1+'):
            self.target_positions["head_motor_1"] += self.degree_step
            print(f"[HEAD] head_motor_1: {self.target_positions['head_motor_1']}")
        if key_state.get('head_motor_1-'):
            self.target_positions["head_motor_1"] -= self.degree_step
            print(f"[HEAD] head_motor_1: {self.target_positions['head_motor_1']}")
        if key_state.get('head_motor_2+'):
            self.target_positions["head_motor_2"] += self.degree_step
            print(f"[HEAD] head_motor_2: {self.target_positions['head_motor_2']}")
        if key_state.get('head_motor_2-'):
            self.target_positions["head_motor_2"] -= self.degree_step
            print(f"[HEAD] head_motor_2: {self.target_positions['head_motor_2']}")

    def p_control_action(self, robot, obs=None):
        if obs is None:
            obs = robot.get_observation()
        action = {}
        for motor in self.target_positions:
            current = obs.get(f"{HEAD_MOTOR_MAP[motor]}.pos", 0.0)
            error = self.target_positions[motor] - current
            control = self.kp * error
            action[f"{HEAD_MOTOR_MAP[motor]}.pos"] = current + control
        return action

class SimpleTeleopArm:
    def __init__(self, kinematics, joint_map, initial_obs, prefix="left", kp=0.81):
        self.kinematics = kinematics
        self.joint_map = joint_map
        self.prefix = prefix  # To distinguish left and right arm
        self.kp = kp
        # Initial joint positions
        self.joint_positions = {
            "shoulder_pan": initial_obs[f"{prefix}_arm_shoulder_pan.pos"],
            "shoulder_lift": initial_obs[f"{prefix}_arm_shoulder_lift.pos"],
            "elbow_flex": initial_obs[f"{prefix}_arm_elbow_flex.pos"],
            "wrist_flex": initial_obs[f"{prefix}_arm_wrist_flex.pos"],
            "wrist_roll": initial_obs[f"{prefix}_arm_wrist_roll.pos"],
            "gripper": initial_obs[f"{prefix}_arm_gripper.pos"],
        }
        # Set initial x/y to fixed values
        self.current_x = 0.1629
        self.current_y = 0.1131
        self.pitch = 0.0
        # Set the degree step and xy step
        self.degree_step = 3
        self.xy_step = 0.0081
        # Set target positions to zero for P control
        self.target_positions = {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        }
        self.zero_pos = {
            'shoulder_pan': 0.0,
            'shoulder_lift': 0.0,
            'elbow_flex': 0.0,
            'wrist_flex': 0.0,
            'wrist_roll': 0.0,
            'gripper': 0.0
        }

    def move_to_zero_position(self, robot):
        print(f"[{self.prefix}] Moving to Zero Position: {self.zero_pos} ......")
        self.target_positions = self.zero_pos.copy()  # Use copy to avoid reference issues
        
        # Reset kinematic variables to their initial state
        self.current_x = 0.1629
        self.current_y = 0.1131
        self.pitch = 0.0
        
        # Don't let handle_keys recalculate wrist_flex - set it explicitly
        self.target_positions["wrist_flex"] = 0.0
        
        action = self.p_control_action(robot)
        robot.send_action(action)

    def handle_keys(self, key_state):
        # Joint increments
        if key_state.get('shoulder_pan+'):
            self.target_positions["shoulder_pan"] += self.degree_step
            print(f"[{self.prefix}] shoulder_pan: {self.target_positions['shoulder_pan']}")
        if key_state.get('shoulder_pan-'):
            self.target_positions["shoulder_pan"] -= self.degree_step
            print(f"[{self.prefix}] shoulder_pan: {self.target_positions['shoulder_pan']}")
        if key_state.get('wrist_roll+'):
            self.target_positions["wrist_roll"] += self.degree_step
            print(f"[{self.prefix}] wrist_roll: {self.target_positions['wrist_roll']}")
        if key_state.get('wrist_roll-'):
            self.target_positions["wrist_roll"] -= self.degree_step
            print(f"[{self.prefix}] wrist_roll: {self.target_positions['wrist_roll']}")
        
        # Gripper control keeps the last target instead of auto-opening on release.
        if key_state.get('gripper+'):
            self.target_positions["gripper"] += self.degree_step
            self.target_positions["gripper"] = min(self.target_positions["gripper"], 100.0)
            print(f"[{self.prefix}] gripper: {self.target_positions['gripper']}")
        if key_state.get('gripper-'):
            self.target_positions["gripper"] -= self.degree_step
            self.target_positions["gripper"] = max(self.target_positions["gripper"], 0.0)
            print(f"[{self.prefix}] gripper: {self.target_positions['gripper']}")
        
        if key_state.get('pitch+'):
            self.pitch += self.degree_step
            print(f"[{self.prefix}] pitch: {self.pitch}")
        if key_state.get('pitch-'):
            self.pitch -= self.degree_step
            print(f"[{self.prefix}] pitch: {self.pitch}")

        # XY plane (IK)
        moved = False
        if key_state.get('x+'):
            self.current_x += self.xy_step
            moved = True
            print(f"[{self.prefix}] x+: {self.current_x:.4f}, y: {self.current_y:.4f}")
        if key_state.get('x-'):
            self.current_x -= self.xy_step
            moved = True
            print(f"[{self.prefix}] x-: {self.current_x:.4f}, y: {self.current_y:.4f}")
        if key_state.get('y+'):
            self.current_y += self.xy_step
            moved = True
            print(f"[{self.prefix}] x: {self.current_x:.4f}, y+: {self.current_y:.4f}")
        if key_state.get('y-'):
            self.current_y -= self.xy_step
            moved = True
            print(f"[{self.prefix}] x: {self.current_x:.4f}, y-: {self.current_y:.4f}")
        if moved:
            joint2, joint3 = self.kinematics.inverse_kinematics(self.current_x, self.current_y)
            self.target_positions["shoulder_lift"] = joint2
            self.target_positions["elbow_flex"] = joint3
            print(f"[{self.prefix}] shoulder_lift: {joint2}, elbow_flex: {joint3}")

        # Wrist flex is always coupled to pitch and the other two
        self.target_positions["wrist_flex"] = (
            -self.target_positions["shoulder_lift"]
            -self.target_positions["elbow_flex"]
            + self.pitch
        )
        # print(f"[{self.prefix}] wrist_flex: {self.target_positions['wrist_flex']}")

    def p_control_action(self, robot, obs=None):
        if obs is None:
            obs = robot.get_observation()
        current = {j: obs[f"{self.prefix}_arm_{j}.pos"] for j in self.joint_map}
        action = {}
        for j in self.target_positions:
            error = self.target_positions[j] - current[j]
            control = self.kp * error
            action[f"{self.joint_map[j]}.pos"] = current[j] + control
        return action
    

def get_axis(joystick, index, default=0.0):
    return joystick.get_axis(index) if joystick.get_numaxes() > index else default


def get_button(joystick, index):
    return bool(joystick.get_button(index)) if joystick.get_numbuttons() > index else False


def get_hat(joystick):
    return joystick.get_hat(0) if joystick.get_numhats() > 0 else (0, 0)


def trigger_pressed(value):
    return value > TRIGGER_DEADZONE


def print_controller_state(joystick):
    pygame.event.pump()
    axes = [round(joystick.get_axis(i), 3) for i in range(joystick.get_numaxes())]
    buttons = [joystick.get_button(i) for i in range(joystick.get_numbuttons())]
    hats = [joystick.get_hat(i) for i in range(joystick.get_numhats())]
    print(f"[MAIN] Controller axes at rest: {axes}")
    print(f"[MAIN] Controller buttons: {len(buttons)}, hats: {hats}")


# --- XBOX Controller Mapping ---
def get_xbox_key_state(joystick, keymap):
    """
    Map XBOX controller state to semantic action booleans using the provided keymap.
    """
    # Get stick pressed states
    left_x = get_axis(joystick, 0)
    left_y = get_axis(joystick, 1)
    right_x = get_axis(joystick, 2)
    right_y = get_axis(joystick, 3)
    left_trigger = get_axis(joystick, 4)
    right_trigger = get_axis(joystick, 5)
    hats = get_hat(joystick)

    left_stick_pressed = get_button(joystick, 9)
    right_stick_pressed = get_button(joystick, 10)
    lb_pressed = get_button(joystick, 4)
    rb_pressed = get_button(joystick, 5)

    # Map controller state to semantic actions
    state = {}
    for action, control in keymap.items():
        if control == 'left_trigger':
            state[action] = trigger_pressed(left_trigger) and not lb_pressed
        elif control == 'left_trigger_lb':
            state[action] = trigger_pressed(left_trigger) and lb_pressed
        elif control == 'right_trigger':
            state[action] = trigger_pressed(right_trigger) and not rb_pressed
        elif control == 'right_trigger_rb':
            state[action] = trigger_pressed(right_trigger) and rb_pressed
        elif control == 'a':
            state[action] = get_button(joystick, 0)
        elif control == 'b':
            state[action] = get_button(joystick, 1)
        elif control == 'x':
            state[action] = get_button(joystick, 2)
        elif control == 'y':
            state[action] = get_button(joystick, 3)
        elif control == 'back':
            state[action] = get_button(joystick, 6)
        elif control == 'start':
            state[action] = get_button(joystick, 7)
        elif control == 'dpad_up':
            state[action] = hats[1] == 1
        elif control == 'dpad_down':
            state[action] = hats[1] == -1
        elif control == 'dpad_left':
            state[action] = hats[0] == -1
        elif control == 'dpad_right':
            state[action] = hats[0] == 1
        # Left stick controls (when not pressed)
        elif control == 'left_stick_up':
            state[action] = (not left_stick_pressed) and (not lb_pressed) and (left_y < -STICK_DEADZONE)
        elif control == 'left_stick_down':
            state[action] = (not left_stick_pressed) and (not lb_pressed) and (left_y > STICK_DEADZONE)
        elif control == 'left_stick_left':
            state[action] = (not left_stick_pressed) and (not lb_pressed) and (left_x < -STICK_DEADZONE)
        elif control == 'left_stick_right':
            state[action] = (not left_stick_pressed) and (not lb_pressed) and (left_x > STICK_DEADZONE)
        # Right stick controls (when not pressed) - Fixed axis mapping
        elif control == 'right_stick_up':
            state[action] = (not right_stick_pressed) and (not rb_pressed) and (right_y < -STICK_DEADZONE)
        elif control == 'right_stick_down':
            state[action] = (not right_stick_pressed) and (not rb_pressed) and (right_y > STICK_DEADZONE)
        elif control == 'right_stick_left':
            state[action] = (not right_stick_pressed) and (not rb_pressed) and (right_x < -STICK_DEADZONE)
        elif control == 'right_stick_right':
            state[action] = (not right_stick_pressed) and (not rb_pressed) and (right_x > STICK_DEADZONE)
        # Left stick pressed controls
        elif control == 'left_stick_pressed_right':
            state[action] = left_stick_pressed and (not lb_pressed) and (left_x > STICK_DEADZONE)
        elif control == 'left_stick_pressed_left':
            state[action] = left_stick_pressed and (not lb_pressed) and (left_x < -STICK_DEADZONE)
        # Right stick pressed controls - Fixed axis mapping
        elif control == 'right_stick_pressed_right':
            state[action] = right_stick_pressed and (not rb_pressed) and (right_x > STICK_DEADZONE)
        elif control == 'right_stick_pressed_left':
            state[action] = right_stick_pressed and (not rb_pressed) and (right_x < -STICK_DEADZONE)
        # LB pressed controls (only when stick is moved)
        elif control == 'lb_up':
            state[action] = lb_pressed and (left_y < -STICK_DEADZONE)
        elif control == 'lb_down':
            state[action] = lb_pressed and (left_y > STICK_DEADZONE)
        elif control == 'lb_right':
            state[action] = lb_pressed and (left_x > STICK_DEADZONE)
        elif control == 'lb_left':
            state[action] = lb_pressed and (left_x < -STICK_DEADZONE)
        # RB pressed controls (only when stick is moved)
        elif control == 'rb_up':
            state[action] = rb_pressed and (right_y < -STICK_DEADZONE)
        elif control == 'rb_down':
            state[action] = rb_pressed and (right_y > STICK_DEADZONE)
        elif control == 'rb_right':
            state[action] = rb_pressed and (right_x > STICK_DEADZONE)
        elif control == 'rb_left':
            state[action] = rb_pressed and (right_x < -STICK_DEADZONE)
        else:
            state[action] = False
    return state

def get_base_action(joystick, robot):
    """
    Get base action from XBOX controller input - simplified to only forward/backward and rotate.
    """
    hats = get_hat(joystick)
    
    # Get pressed keys for base control
    pressed_keys = set()
    
    # Map controller inputs through the robot's configured keyboard base controls.
    if hats[1] == 1:   # D-pad up
        pressed_keys.add(robot.teleop_keys["forward"])
    if hats[1] == -1:  # D-pad down
        pressed_keys.add(robot.teleop_keys["backward"])
    if hats[0] == -1:  # D-pad left
        pressed_keys.add(robot.teleop_keys["rotate_left"])
    if hats[0] == 1:   # D-pad right
        pressed_keys.add(robot.teleop_keys["rotate_right"])
    
    # Convert to numpy array and get base action
    keyboard_keys = np.array(list(pressed_keys))
    base_action = robot._from_keyboard_to_base_action(keyboard_keys) or {}
    
    return base_action


def main():
    FPS = 50

    init_rerun(session_name="xlerobot_teleop_xbox")

    # Init XBOX controller
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("No XBOX controller detected!")
        pygame.quit()
        return
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"[MAIN] Using controller: {joystick.get_name()}")
    print_controller_state(joystick)

    robot_config = XLerobotConfig(
        id="xlerobot",
        port1="/dev/ttyACM1",
        port2="/dev/ttyACM0",
    )
    robot = XLerobot(robot_config)
    try:
        robot.connect()
        print(f"[MAIN] Successfully connected to robot")
    except Exception as e:
        print(f"[MAIN] Failed to connect to robot: {e}")
        print(robot_config)
        print(robot)
        joystick.quit()
        pygame.quit()
        return

    # Init the arm and head instances
    obs = robot.get_observation()
    kin_left = SO101Kinematics()
    kin_right = SO101Kinematics()
    left_arm = SimpleTeleopArm(kin_left, LEFT_JOINT_MAP, obs, prefix="left")
    right_arm = SimpleTeleopArm(kin_right, RIGHT_JOINT_MAP, obs, prefix="right")
    head_control = SimpleHeadControl(obs)

    # Move both arms and head to zero position at start
    left_arm.move_to_zero_position(robot)
    right_arm.move_to_zero_position(robot)
    precise_sleep(0.2)

    try:
        while True:
            loop_start_t = time.perf_counter()
            pygame.event.pump()
            left_key_state = get_xbox_key_state(joystick, LEFT_KEYMAP)
            right_key_state = get_xbox_key_state(joystick, RIGHT_KEYMAP)
            
            if get_button(joystick, 7):  # Start
                print("[MAIN] Start button pressed; stopping and disconnecting...")
                break

            global_reset = get_button(joystick, 6)  # Back
            
            # Handle global reset for all components
            if global_reset:
                print("[MAIN] Global reset triggered!")
                left_arm.move_to_zero_position(robot)
                right_arm.move_to_zero_position(robot)
                head_control.move_to_zero_position(robot)
                continue

            # Handle both arms separately and simultaneously
            left_arm.handle_keys(left_key_state)
            right_arm.handle_keys(right_key_state)
            head_control.handle_keys(left_key_state)  # Head controlled by left arm keymap

            obs = robot.get_observation()
            left_action = left_arm.p_control_action(robot, obs)
            right_action = right_arm.p_control_action(robot, obs)
            head_action = head_control.p_control_action(robot, obs)

            base_action = get_base_action(joystick, robot)

            # Merge all actions
            action = {**left_action, **right_action, **head_action, **base_action}
            robot.send_action(action)

            log_rerun_data(obs, action)
            precise_sleep(max(1.0 / FPS - (time.perf_counter() - loop_start_t), 0.0))
    finally:
        if robot.bus1.is_connected or robot.bus2.is_connected:
            robot.disconnect()
        joystick.quit()
        pygame.quit()
        print("Teleoperation ended.")

if __name__ == "__main__":
    main()
