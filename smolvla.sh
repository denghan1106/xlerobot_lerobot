#!/usr/bin/env bash
set -euo pipefail

cd /data/cdzhitu/robot_dev/xlerobot_lerobot

MODE="${1:-check}"
DEFAULT_HF_USER="${DEFAULT_HF_USER:-Denghan}"
ARM="${ARM:-right}"
TASK="${TASK:-pick up the red box place it inside the basket on the robot cart.}"
RECORD_VENV="${RECORD_VENV:-.venv}"
PYTHON="${RECORD_VENV}/bin/python"
HF="${RECORD_VENV}/bin/hf"
LEROBOT_RECORD="${RECORD_VENV}/bin/lerobot-record"
LEROBOT_TRAIN="${RECORD_VENV}/bin/lerobot-train"
export PATH="${RECORD_VENV}/bin:${PATH}"
DATASET_ROOT="${DATASET_ROOT:-/data/cdzhitu/robot_dev/lerobot_datasets}"
CALIBRATION_ROOT="${CALIBRATION_ROOT:-/home/cdzhitu/.cache/huggingface/lerobot/calibration}"
NUM_EPISODES="${NUM_EPISODES:-10}"
FPS="${FPS:-20}"
EPISODE_TIME_S="${EPISODE_TIME_S:-30}"
RESET_TIME_S="${RESET_TIME_S:-15}"
DISPLAY_DATA="${DISPLAY_DATA:-true}"
DISPLAY_IP="${DISPLAY_IP:-}"
DISPLAY_PORT="${DISPLAY_PORT:-}"
DISPLAY_COMPRESSED_IMAGES="${DISPLAY_COMPRESSED_IMAGES:-true}"
RERUN_MEMORY_LIMIT="${RERUN_MEMORY_LIMIT:-3GB}"
CAMERA_PRESET="${CAMERA_PRESET:-right_task}"
CONFIRM_UPLOAD="${CONFIRM_UPLOAD:-true}"
export HF_LEROBOT_HOME="$DATASET_ROOT"
export HF_LEROBOT_CALIBRATION="$CALIBRATION_ROOT"
export LEROBOT_RERUN_MEMORY_LIMIT="$RERUN_MEMORY_LIMIT"

if [[ "${SETUP:-0}" == "1" ]]; then
  uv sync --locked --extra smolvla
fi

hf_user() {
  NO_COLOR=1 "$HF" auth whoami | awk -F': *' 'NR==1 {print ($2 != "" ? $2 : $1)}'
}

case "$MODE" in
  setup-record)
    uv pip install --python "$PYTHON" -e ".[core_scripts,smolvla,feetech,gamepad]"
    ;;

  check)
    "$PYTHON" -c 'from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy; policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base"); print("loaded smolvla_base ok"); print(type(policy))'
    ;;

  record)
    HF_USER="${HF_USER:-$DEFAULT_HF_USER}"
    if [[ -z "$HF_USER" ]]; then
      echo "Could not detect Hugging Face username. Set HF_USER manually, for example:" >&2
      echo "  HF_USER=Denghan $0 record" >&2
      exit 1
    fi

    if [[ "$ARM" == "right" ]]; then
      ROBOT_TYPE="xlerobot_right_arm"
      TELEOP_TYPE="xlerobot_right_arm_leader"
      TELEOP_ID="xlerobot_right_leader"
      DATASET_NAME="${DATASET_NAME:-xlerobot_right_arm_record}"
    elif [[ "$ARM" == "left" ]]; then
      ROBOT_TYPE="xlerobot_left_arm"
      TELEOP_TYPE="xlerobot_left_arm_leader"
      TELEOP_ID="xlerobot_left_leader"
      DATASET_NAME="${DATASET_NAME:-xlerobot_left_arm_record}"
    else
      echo "ARM must be 'right' or 'left'. Current: $ARM" >&2
      exit 1
    fi

    CAMERA_ARGS=()
    if [[ "$CAMERA_PRESET" == "right_task" ]]; then
      CAMERA_ARGS=(
        --robot.cameras="{ center: {type: opencv, index_or_path: /dev/xlerobot_cam_top, width: 640, height: 480, fps: 30, fourcc: MJPG, rotation: 180}, right: {type: opencv, index_or_path: /dev/xlerobot_cam_right, width: 640, height: 480, fps: 30, fourcc: MJPG, rotation: 180}}"
      )
    elif [[ "$CAMERA_PRESET" != "all" ]]; then
      echo "CAMERA_PRESET must be 'right_task' or 'all'. Current: $CAMERA_PRESET" >&2
      exit 1
    fi

    DISPLAY_ARGS=(--display_data="$DISPLAY_DATA" --display_compressed_images="$DISPLAY_COMPRESSED_IMAGES")
    if [[ -n "$DISPLAY_IP" && -n "$DISPLAY_PORT" ]]; then
      DISPLAY_ARGS+=(--display_ip="$DISPLAY_IP" --display_port="$DISPLAY_PORT")
    fi

    mkdir -p "$DATASET_ROOT"

    echo "Recording configuration:"
    echo "  HF user: ${HF_USER}"
    echo "  Dataset: ${HF_USER}/${DATASET_NAME}"
    echo "  Dataset root: ${DATASET_ROOT}"
    echo "  Calibration root: ${CALIBRATION_ROOT}"
    echo "  Task: ${TASK}"
    echo "  Arm: ${ARM}"
    echo "  Robot type: ${ROBOT_TYPE}"
    echo "  Teleop type: ${TELEOP_TYPE}"
    echo "  Teleop id: ${TELEOP_ID}"
    echo "  Camera preset: ${CAMERA_PRESET}"
    echo "  FPS: ${FPS}"
    echo "  Episodes: ${NUM_EPISODES}"
    echo "  Episode time: ${EPISODE_TIME_S}s"
    echo "  Reset time: ${RESET_TIME_S}s"
    echo "  Display data: ${DISPLAY_DATA}"
    echo "  Display compressed images: ${DISPLAY_COMPRESSED_IMAGES}"
    echo "  Rerun memory limit: ${RERUN_MEMORY_LIMIT}"
    echo "  Confirm upload: ${CONFIRM_UPLOAD}"

    "$LEROBOT_RECORD" \
      --robot.type="$ROBOT_TYPE" \
      --robot.id=xlerobot \
      "${CAMERA_ARGS[@]}" \
      --teleop.type="$TELEOP_TYPE" \
      --teleop.id="$TELEOP_ID" \
      --dataset.repo_id="${HF_USER}/${DATASET_NAME}" \
      --dataset.single_task="$TASK" \
      --dataset.num_episodes="$NUM_EPISODES" \
      --dataset.fps="$FPS" \
      --dataset.episode_time_s="$EPISODE_TIME_S" \
      --dataset.reset_time_s="$RESET_TIME_S" \
      --confirm_upload="$CONFIRM_UPLOAD" \
      "${DISPLAY_ARGS[@]}"
    ;;

  train)
    HF_USER="${HF_USER:-$DEFAULT_HF_USER}"
    if [[ -z "$HF_USER" ]]; then
      echo "Could not detect Hugging Face username. Set HF_USER manually, for example:" >&2
      echo "  HF_USER=Denghan $0 train" >&2
      exit 1
    fi
    DATASET_REPO="${DATASET_REPO:-${HF_USER}/xlerobot_right_arm_my_task}"
    DEVICE="${DEVICE:-cuda}"

    "$LEROBOT_TRAIN" \
      --policy.path=lerobot/smolvla_base \
      --dataset.repo_id="$DATASET_REPO" \
      --output_dir=outputs/train/xlerobot_right_arm_smolvla \
      --job_name=xlerobot_right_arm_smolvla \
      --policy.device="$DEVICE" \
      --batch_size=8 \
      --steps=20000 \
      --wandb.enable=true
    ;;

  *)
    echo "Usage: $0 [check|record|train]" >&2
    echo "Examples:" >&2
    echo "  $0 setup-record" >&2
    echo "  SETUP=1 $0 check" >&2
    echo "  TASK='$TASK' $0 record" >&2
    echo "  DISPLAY_DATA=true TASK='$TASK' $0 record" >&2
    echo "  DISPLAY_DATA=true DISPLAY_IP=192.168.1.10 DISPLAY_PORT=9876 TASK='$TASK' $0 record" >&2
    echo "  CAMERA_PRESET=all TASK='$TASK' $0 record" >&2
    echo "  FPS=30 TASK='$TASK' $0 record" >&2
    echo "  DATASET_NAME=xlerobot_right_arm_my_task NUM_EPISODES=50 TASK='$TASK' $0 record" >&2
    echo "  CONFIRM_UPLOAD=false DATASET_NAME=xlerobot_right_arm_my_task TASK='$TASK' $0 record" >&2
    echo "  ARM=left DATASET_NAME=xlerobot_left_arm_my_task TASK='$TASK' $0 record" >&2
    echo "  DATASET_REPO=${DEFAULT_HF_USER}/xlerobot_right_arm_my_task $0 train" >&2
    echo "  DEVICE=cpu DATASET_REPO=${DEFAULT_HF_USER}/xlerobot_right_arm_my_task $0 train" >&2
    exit 1
    ;;
esac
