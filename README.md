# XLeRobot LeRobot Fork

This repository is a working fork of the open-source LeRobot/XLeRobot codebase. It keeps the upstream LeRobot training, recording, policy, dataset, and hardware abstractions, but adds project-specific support for my XLeRobot hardware layout and teleoperation workflow.

The purpose of this README is not to introduce all of LeRobot. It documents what was changed in this fork and how to use the modified entry points.

## Main Changes From Upstream

### 1. Added single-arm follower robot types

Two new robot entry points were added so each XLeRobot arm can be controlled independently:

```bash
--robot.type=xlerobot_right_arm
--robot.type=xlerobot_left_arm
```

They expose only the 6 arm joints as policy/teleop action features:

```text
*_arm_shoulder_pan
*_arm_shoulder_lift
*_arm_elbow_flex
*_arm_wrist_flex
*_arm_wrist_roll
*_arm_gripper
```

The right-arm wrapper still opens the right bus that also contains base motors, but action control is limited to the right arm. Base motors are not used as arm actions.

### 2. Added dedicated leader teleoperators

Two new teleoperator entry points were added:

```bash
--teleop.type=xlerobot_right_arm_leader
--teleop.type=xlerobot_left_arm_leader
```

They map a 6-DOF SO-style leader arm to the corresponding XLeRobot follower arm:

```text
shoulder_pan  -> left/right_arm_shoulder_pan
shoulder_lift -> left/right_arm_shoulder_lift
elbow_flex    -> left/right_arm_elbow_flex
wrist_flex    -> left/right_arm_wrist_flex
wrist_roll    -> left/right_arm_wrist_roll
gripper       -> left/right_arm_gripper
```

The default leader ports are:

```text
xlerobot_left_arm_leader  -> /dev/xlerobot_leader_left
xlerobot_right_arm_leader -> /dev/xlerobot_leader_right
```

Dedicated leaders use their own calibration files by default and do not reuse the full XLeRobot robot calibration.

### 3. Stable device names

The project now expects stable udev names instead of raw `/dev/ttyACM*` and `/dev/video*` paths:

```text
/dev/xlerobot_arm_left
/dev/xlerobot_arm_right
/dev/xlerobot_leader_left
/dev/xlerobot_leader_right
/dev/xlerobot_cam_left
/dev/xlerobot_cam_right
/dev/xlerobot_cam_top
```

Camera names shown in Rerun are standardized as:

```text
left
right
center
```

All cameras were changed to 640x480 MJPG at 30 FPS.

### 4. Calibration behavior

When a calibration file exists, the code no longer silently skips calibration. It prompts:

```text
Press ENTER to use provided calibration file associated with the id ..., or type 'CALIBRATE' and press ENTER to run calibration:
```

Only the exact word `CALIBRATE` triggers recalibration. Pressing Enter uses the existing calibration.

Leader calibration files are stored under:

```text
~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/
```

The full XLeRobot robot calibration is stored separately under:

```text
~/.cache/huggingface/lerobot/calibration/robots/xlerobot/
```

Do not delete the full robot calibration unless intentionally recalibrating the whole robot.

### 5. Leader/follower startup alignment

The leader teleoperators align their initial command to the current follower observation at startup. This allows the leader and follower to start from slightly different physical poses without immediately commanding a large jump.

### 6. Display data improvements

`--display_data=true` now prints leader, follower current state, and follower goal separately:

```text
left_arm_leader
left_arm_follower_now
left_arm_follower_goal
```

or:

```text
right_arm_leader
right_arm_follower_now
right_arm_follower_goal
```

This makes it easier to tell whether an issue is in leader reading, mapping, follower command, or physical motion.

### 7. Wrap-aware leader reading

Dedicated leader arms can have a joint calibration range that crosses the Feetech 0/4095 encoder boundary. The custom leader wrappers read raw motor positions and normalize them with boundary-aware logic to avoid sudden jumps in `shoulder_pan`.

### 8. PID and torque-related adjustments

XLeRobot arm PID settings were adjusted to match the official SO-style follower settings more closely:

```text
P_Coefficient = 16
I_Coefficient = 0
D_Coefficient = 32
```

For the right-arm-only follower, torque enable is applied only to right-arm motors, not base motors.

## Environment

Use Python 3.12. This repository uses `uv` for the project environment.

From the project root:

```bash
cd /data/cdzhitu/robot_dev/xlerobot_lerobot
uv run python -V
```

Expected:

```text
Python 3.12.x
```

Run project commands with `uv run`, for example:

```bash
uv run lerobot-teleoperate
uv run lerobot-record
uv run lerobot-train
```

## Teleoperation

### Right leader controls right follower

```bash
uv run lerobot-teleoperate \
  --robot.type=xlerobot_right_arm \
  --robot.id=xlerobot \
  --teleop.type=xlerobot_right_arm_leader \
  --teleop.id=xlerobot_right_leader \
  --display_data=true
```

### Left leader controls left follower

```bash
uv run lerobot-teleoperate \
  --robot.type=xlerobot_left_arm \
  --robot.id=xlerobot \
  --teleop.type=xlerobot_left_arm_leader \
  --teleop.id=xlerobot_left_leader \
  --display_data=true
```

Use different `teleop.id` values for left and right leaders. Recommended:

```text
xlerobot_left_leader
xlerobot_right_leader
```

Do not mix old ids such as `right_arm_leader` unless intentionally creating a separate calibration file.

## Data Collection

After teleoperation is stable, record demonstrations with `lerobot-record`. Use the same robot and teleop types as teleoperation.

Example for the right arm:

```bash
uv run lerobot-record \
  --robot.type=xlerobot_right_arm \
  --robot.id=xlerobot \
  --teleop.type=xlerobot_right_arm_leader \
  --teleop.id=xlerobot_right_leader \
  --dataset.repo_id=cdzhitu/xlerobot_right_arm_demo \
  --dataset.num_episodes=50 \
  --dataset.episode_time_s=20 \
  --dataset.reset_time_s=10 \
  --display_data=true
```

Adjust `repo_id`, episode count, and episode duration for the actual task.

## SmolVLA Base Policy

The base SmolVLA policy can be loaded with:

```bash
uv run python -c 'from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy; policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base"); print("loaded smolvla_base ok"); print(type(policy))'
```

This downloads and caches the official base weights locally through Hugging Face.

## Training

After local demonstrations are recorded, fine-tune from SmolVLA:

```bash
uv run lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=cdzhitu/xlerobot_right_arm_demo \
  --output_dir=outputs/train/xlerobot_right_arm_smolvla
```

Training command details may need adjustment depending on dataset features, GPU memory, and the current LeRobot training config.

## GitHub Upload Commands

Check current changes:

```bash
git status
```

Stage all project changes:

```bash
git add README.md examples src
```

Commit:

```bash
git commit -m "Add XLeRobot single-arm teleoperation support"
```

If this repository already has your GitHub remote:

```bash
git remote -v
git push origin main
```

If this is a new GitHub repository, create an empty repo on GitHub first, then run:

```bash
git remote add origin git@github.com:<your-username>/<your-repo>.git
git branch -M main
git push -u origin main
```

If using HTTPS instead of SSH:

```bash
git remote add origin https://github.com/<your-username>/<your-repo>.git
git branch -M main
git push -u origin main
```

Before pushing, do not add local calibration caches, datasets, model weights, or `.venv` directories.
