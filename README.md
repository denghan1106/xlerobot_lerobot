# XLeRobot LeRobot Fork

本仓库基于开源 LeRobot/XLeRobot 项目修改而来。上游 LeRobot 的训练、录制、策略、数据集和硬件抽象仍然保留；本 fork 重点增加了适配我当前 XLeRobot 硬件结构的单臂 follower、独立 leader、稳定设备名、校准和遥操作流程。

This repository is a fork of the open-source LeRobot/XLeRobot codebase. It keeps the upstream LeRobot training, recording, policy, dataset, and hardware abstractions, while adding project-specific support for my XLeRobot hardware layout and teleoperation workflow.

本 README 重点说明“相对上游项目改了什么”，不是完整介绍 LeRobot 官方项目。

This README focuses on what changed in this fork, rather than reintroducing the full upstream LeRobot project.

## 主要改动 / Main Changes

### 1. 新增单臂 follower 入口 / Single-Arm Follower Robot Types

新增两个 robot 类型，用于单独控制 XLeRobot 的左臂或右臂：

Two robot entry points were added so each XLeRobot arm can be controlled independently:

```bash
--robot.type=xlerobot_right_arm
--robot.type=xlerobot_left_arm
```

这两个入口只暴露 6 个机械臂关节作为 policy/teleop action：

They expose only the 6 arm joints as policy/teleop action features:

```text
*_arm_shoulder_pan
*_arm_shoulder_lift
*_arm_elbow_flex
*_arm_wrist_flex
*_arm_wrist_roll
*_arm_gripper
```

右臂 wrapper 仍然会打开包含底盘电机的右侧总线，但 action 只控制右臂 6 个关节。底盘电机不会作为机械臂 action 使用。

The right-arm wrapper still opens the right bus that also contains base motors, but action control is limited to the 6 right-arm joints. Base motors are not used as arm actions.

### 2. 新增独立 leader 入口 / Dedicated Leader Teleoperators

新增两个 teleoperator 类型：

Two teleoperator entry points were added:

```bash
--teleop.type=xlerobot_right_arm_leader
--teleop.type=xlerobot_left_arm_leader
```

它们把 6 自由度 SO-style leader 的普通关节名映射到 XLeRobot follower 的左右臂关节名：

They map a 6-DOF SO-style leader arm to the corresponding XLeRobot follower arm:

```text
shoulder_pan  -> left/right_arm_shoulder_pan
shoulder_lift -> left/right_arm_shoulder_lift
elbow_flex    -> left/right_arm_elbow_flex
wrist_flex    -> left/right_arm_wrist_flex
wrist_roll    -> left/right_arm_wrist_roll
gripper       -> left/right_arm_gripper
```

默认 leader 端口：

Default leader ports:

```text
xlerobot_left_arm_leader  -> /dev/xlerobot_leader_left
xlerobot_right_arm_leader -> /dev/xlerobot_leader_right
```

独立 leader 默认使用自己的校准文件，不再复用整机 XLeRobot 校准。

Dedicated leaders use their own calibration files by default and do not reuse the full XLeRobot robot calibration.

### 3. 稳定设备名 / Stable Device Names

项目默认使用稳定 udev 设备名，不再依赖易变化的 `/dev/ttyACM*` 和 `/dev/video*`：

The project expects stable udev names instead of raw `/dev/ttyACM*` and `/dev/video*` paths:

```text
/dev/xlerobot_arm_left
/dev/xlerobot_arm_right
/dev/xlerobot_leader_left
/dev/xlerobot_leader_right
/dev/xlerobot_cam_left
/dev/xlerobot_cam_right
/dev/xlerobot_cam_top
```

Rerun 中的相机名统一为：

Camera names shown in Rerun are standardized as:

```text
left
right
center
```

相机默认设置为 640x480、MJPG、30 FPS。

All cameras default to 640x480 MJPG at 30 FPS.

### 4. 校准行为 / Calibration Behavior

如果找到校准文件，程序不会静默跳过，而是提示用户确认：

When a calibration file exists, the code no longer silently skips calibration. It prompts:

```text
Press ENTER to use provided calibration file associated with the id ..., or type 'CALIBRATE' and press ENTER to run calibration:
```

按 Enter 使用已有校准；只有输入 `CALIBRATE` 才会重新校准。

Press Enter to use the existing calibration. Only the exact word `CALIBRATE` triggers recalibration.

Leader 校准文件位置：

Leader calibration files are stored under:

```text
~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/
```

整机 XLeRobot 校准文件位置：

The full XLeRobot robot calibration is stored under:

```text
~/.cache/huggingface/lerobot/calibration/robots/xlerobot/
```

不要误删整机校准文件，除非明确要重新做整机校准。

Do not delete the full robot calibration unless intentionally recalibrating the whole robot.

### 5. leader/follower 启动对齐 / Startup Alignment

leader teleoperator 会在启动时读取 follower 当前观测，并计算初始 offset。这样 leader 和 follower 起始姿态略有差异时，不会一开始就让 follower 突然大幅跳动。

Leader teleoperators align their initial command to the current follower observation at startup. This allows the leader and follower to start from slightly different physical poses without immediately commanding a large jump.

### 6. display data 显示增强 / Display Data Improvements

`--display_data=true` 现在会同时显示 leader、follower 当前值和 follower 目标值：

`--display_data=true` now prints leader, follower current state, and follower goal separately:

```text
left_arm_leader
left_arm_follower_now
left_arm_follower_goal
```

或：

or:

```text
right_arm_leader
right_arm_follower_now
right_arm_follower_goal
```

这样可以更容易判断问题发生在 leader 读取、映射、目标发送，还是 follower 机械/电机侧。

This makes it easier to tell whether an issue is in leader reading, mapping, follower command, or physical motion.

### 7. leader 跨 0/4095 边界读取 / Wrap-Aware Leader Reading

独立 leader 的某些关节校准范围可能跨过 Feetech 编码器的 `0/4095` 边界。自定义 leader wrapper 会读取原始电机位置，并使用支持跨界的归一化逻辑，避免 `shoulder_pan` 突然跳变。

Dedicated leader arms can have a joint calibration range that crosses the Feetech `0/4095` encoder boundary. The custom leader wrappers read raw motor positions and normalize them with boundary-aware logic to avoid sudden jumps in `shoulder_pan`.

### 8. PID 和扭矩相关调整 / PID and Torque Adjustments

XLeRobot 机械臂位置控制 PID 调整为更接近官方 SO-style follower 的设置：

XLeRobot arm PID settings were adjusted to match the official SO-style follower settings more closely:

```text
P_Coefficient = 16
I_Coefficient = 0
D_Coefficient = 32
```

右臂单臂 follower 中，启用扭矩时只对右臂关节启用，不对底盘电机启用。

For the right-arm-only follower, torque enable is applied only to right-arm motors, not base motors.

## 环境 / Environment

使用 Python 3.12。本项目主要通过 `uv` 运行。

Use Python 3.12. This repository uses `uv` for the project environment.

```bash
cd /data/cdzhitu/robot_dev/xlerobot_lerobot
uv run python -V
```

期望输出：

Expected:

```text
Python 3.12.x
```

运行项目命令时优先使用 `uv run`：

Run project commands with `uv run`:

```bash
uv run lerobot-teleoperate
uv run lerobot-record
uv run lerobot-train
```

## 遥操作 / Teleoperation

### 右 leader 控制右 follower / Right Leader Controls Right Follower

```bash
uv run lerobot-teleoperate \
  --robot.type=xlerobot_right_arm \
  --robot.id=xlerobot \
  --teleop.type=xlerobot_right_arm_leader \
  --teleop.id=xlerobot_right_leader \
  --display_data=true
```

### 左 leader 控制左 follower / Left Leader Controls Left Follower

```bash
uv run lerobot-teleoperate \
  --robot.type=xlerobot_left_arm \
  --robot.id=xlerobot \
  --teleop.type=xlerobot_left_arm_leader \
  --teleop.id=xlerobot_left_leader \
  --display_data=true
```

左右 leader 应该使用不同的 `teleop.id`。推荐：

Use different `teleop.id` values for left and right leaders. Recommended:

```text
xlerobot_left_leader
xlerobot_right_leader
```

不要混用旧 id，例如 `right_arm_leader`，除非你明确想创建另一个独立校准文件。

Do not mix old ids such as `right_arm_leader` unless intentionally creating a separate calibration file.

## 数据采集 / Data Collection

遥操作稳定后，用 `lerobot-record` 录制 demonstration。录制时使用和 teleop 相同的 robot/teleop 类型。

After teleoperation is stable, record demonstrations with `lerobot-record`. Use the same robot and teleop types as teleoperation.

右臂示例：

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

根据实际任务调整 `repo_id`、episode 数量和 episode 时长。

Adjust `repo_id`, episode count, and episode duration for the actual task.

## SmolVLA 基础模型 / SmolVLA Base Policy

可以这样测试官方 `lerobot/smolvla_base` 是否能正常加载：

The official `lerobot/smolvla_base` policy can be loaded with:

```bash
uv run python -c 'from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy; policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base"); print("loaded smolvla_base ok"); print(type(policy))'
```

这会通过 Hugging Face 下载并缓存官方基础权重。

This downloads and caches the official base weights locally through Hugging Face.

## 训练 / Training

本地 demonstration 录制完成后，可以基于 SmolVLA fine-tune：

After local demonstrations are recorded, fine-tune from SmolVLA:

```bash
uv run lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=cdzhitu/xlerobot_right_arm_demo \
  --output_dir=outputs/train/xlerobot_right_arm_smolvla
```

具体训练参数需要根据数据集 features、GPU 显存和当前 LeRobot 配置进一步调整。

Training command details may need adjustment depending on dataset features, GPU memory, and the current LeRobot training config.
