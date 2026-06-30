#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$SCRIPT_DIR}"
cd "$REPO_DIR"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$REPO_DIR/.." && pwd)}"

MODE="${1:-check}"
DEFAULT_HF_USER="${DEFAULT_HF_USER:-Denghan}"
ARM="${ARM:-right}"
TASK="${TASK:-pick up the red box place it inside the basket on the robot cart.}"
RECORD_VENV="${RECORD_VENV:-.venv}"
PYTHON="${RECORD_VENV}/bin/python"
HF="${RECORD_VENV}/bin/hf"
LEROBOT_RECORD="${RECORD_VENV}/bin/lerobot-record"
LEROBOT_TRAIN="${RECORD_VENV}/bin/lerobot-train"
JETSON_VENV="${JETSON_VENV:-.venv-jetson}"
JETSON_PYTHON="${JETSON_VENV}/bin/python"
JETSON_CUDA_LIB="${JETSON_CUDA_LIB:-${REPO_DIR}/${JETSON_VENV}/lib/python3.10/site-packages/nvidia/cu12/lib}"
export PATH="${RECORD_VENV}/bin:${PATH}"
DATASET_ROOT="${DATASET_ROOT:-${WORKSPACE_ROOT}/lerobot_datasets}"
TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-${WORKSPACE_ROOT}/traindata}"
CALIBRATION_ROOT="${CALIBRATION_ROOT:-${HOME}/.cache/huggingface/lerobot/calibration}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${WORKSPACE_ROOT}/hf_datasets_cache}"
TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-${WORKSPACE_ROOT}/train_outputs}"
TRAIN_LOG_ROOT="${TRAIN_LOG_ROOT:-${WORKSPACE_ROOT}/train_logs}"
POLICY_PATH="${POLICY_PATH:-${TRAIN_OUTPUT_ROOT}/020000/pretrained_model}"
RENAME_MAP="${RENAME_MAP:-{\"observation.images.center\":\"observation.images.camera1\",\"observation.images.right\":\"observation.images.camera2\"}}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
NUM_EPISODES="${NUM_EPISODES:-10}"
FPS="${FPS:-20}"
EPISODE_TIME_S="${EPISODE_TIME_S:-30}"
RESET_TIME_S="${RESET_TIME_S:-15}"
ROLLOUT_DURATION="${ROLLOUT_DURATION:-10}"
DISPLAY_DATA="${DISPLAY_DATA:-true}"
DEPLOY_DISPLAY_DATA="${DEPLOY_DISPLAY_DATA:-false}"
DISPLAY_IP="${DISPLAY_IP:-}"
DISPLAY_PORT="${DISPLAY_PORT:-}"
DISPLAY_COMPRESSED_IMAGES="${DISPLAY_COMPRESSED_IMAGES:-true}"
RERUN_MEMORY_LIMIT="${RERUN_MEMORY_LIMIT:-3GB}"
CAMERA_PRESET="${CAMERA_PRESET:-right_task}"
CONFIRM_UPLOAD="${CONFIRM_UPLOAD:-true}"
export HF_LEROBOT_HOME="$DATASET_ROOT"
export HF_LEROBOT_CALIBRATION="$CALIBRATION_ROOT"
export HF_DATASETS_CACHE
export HF_HUB_DISABLE_XET
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

  train|smoke-train|mac-train)
    HF_USER="${HF_USER:-$DEFAULT_HF_USER}"
    if [[ -z "$HF_USER" ]]; then
      echo "Could not detect Hugging Face username. Set HF_USER manually, for example:" >&2
      echo "  HF_USER=Denghan $0 train" >&2
      exit 1
    fi
    DATASET_REPO="${DATASET_REPO:-local/xlerobot_right_arm_train1}"
    DATASET_PATH="${DATASET_PATH:-${TRAIN_DATA_ROOT}/xlerobot_right_arm_train1}"
    DEVICE="${DEVICE:-cuda}"
    TRAIN_STEPS="${TRAIN_STEPS:-10000}"
    BATCH_SIZE="${BATCH_SIZE:-8}"
    NUM_WORKERS="${NUM_WORKERS:-4}"
    SAVE_FREQ="${SAVE_FREQ:-1000}"
    LOG_FREQ="${LOG_FREQ:-50}"
    EVAL_FREQ="${EVAL_FREQ:-0}"
    WANDB_ENABLE="${WANDB_ENABLE:-true}"
    PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
    OUTPUT_DIR="${OUTPUT_DIR:-${TRAIN_OUTPUT_ROOT}/xlerobot_right_arm_train1_smolvla_10k}"
    JOB_NAME="${JOB_NAME:-xlerobot_right_arm_train1_smolvla_10k}"
    LOG_DIR="${LOG_DIR:-${TRAIN_LOG_ROOT}}"
    EMPTY_CAMERAS="${EMPTY_CAMERAS:-1}"

    if [[ "$MODE" == "smoke-train" ]]; then
      TRAIN_STEPS="${SMOKE_TRAIN_STEPS:-1}"
      BATCH_SIZE="${SMOKE_BATCH_SIZE:-1}"
      NUM_WORKERS="${SMOKE_NUM_WORKERS:-0}"
      SAVE_FREQ="${SMOKE_SAVE_FREQ:-1}"
      LOG_FREQ="${SMOKE_LOG_FREQ:-1}"
      DEVICE="${SMOKE_DEVICE:-cpu}"
      OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-${TRAIN_OUTPUT_ROOT}/smoke_xlerobot_right_arm_train1}"
      JOB_NAME="${SMOKE_JOB_NAME:-smoke_xlerobot_right_arm_train1}"
    fi
    if [[ "$MODE" == "mac-train" ]]; then
      TRAIN_STEPS="${MAC_TRAIN_STEPS:-20000}"
      BATCH_SIZE="${MAC_BATCH_SIZE:-4}"
      NUM_WORKERS="${MAC_NUM_WORKERS:-0}"
      SAVE_FREQ="${MAC_SAVE_FREQ:-2500}"
      LOG_FREQ="${MAC_LOG_FREQ:-200}"
      DEVICE="${MAC_DEVICE:-mps}"
      OUTPUT_DIR="${MAC_OUTPUT_DIR:-outputs/train/mac_xlerobot_right_arm_smolvla_20k_2}"
      JOB_NAME="${MAC_JOB_NAME:-mac_xlerobot_right_arm_smolvla_20k_2}"
    fi
    mkdir -p "$LOG_DIR" "$(dirname "$OUTPUT_DIR")"

    echo "Training configuration:"
    echo "  Dataset repo: ${DATASET_REPO}"
    echo "  Dataset root: ${DATASET_PATH}"
    echo "  Video backend: pyav"
    echo "  Rename map: ${RENAME_MAP}"
    echo "  Empty cameras: ${EMPTY_CAMERAS}"
    echo "  Policy: lerobot/smolvla_base"
    echo "  Device: ${DEVICE}"
    echo "  Steps: ${TRAIN_STEPS}"
    echo "  Batch size: ${BATCH_SIZE}"
    echo "  Num workers: ${NUM_WORKERS}"
    echo "  Output dir: ${OUTPUT_DIR}"
    echo "  Log file: ${LOG_DIR}/${JOB_NAME}.log"
    echo "  W&B enabled: ${WANDB_ENABLE}"
    echo "  Push to hub: ${PUSH_TO_HUB}"

    if [[ "$WANDB_ENABLE" == "true" ]]; then
      unset WANDB_DISABLED
    else
      export WANDB_DISABLED=true
    fi

    "$LEROBOT_TRAIN" \
      --policy.path=lerobot/smolvla_base \
      --dataset.repo_id="$DATASET_REPO" \
      --dataset.root="$DATASET_PATH" \
      --dataset.video_backend=pyav \
      --rename_map="$RENAME_MAP" \
      --policy.empty_cameras="$EMPTY_CAMERAS" \
      --output_dir="$OUTPUT_DIR" \
      --job_name="$JOB_NAME" \
      --policy.device="$DEVICE" \
      --policy.push_to_hub="$PUSH_TO_HUB" \
      --batch_size="$BATCH_SIZE" \
      --num_workers="$NUM_WORKERS" \
      --steps="$TRAIN_STEPS" \
      --save_freq="$SAVE_FREQ" \
      --log_freq="$LOG_FREQ" \
      --eval_freq="$EVAL_FREQ" \
      --wandb.enable="$WANDB_ENABLE" \
      2>&1 | tee "${LOG_DIR}/${JOB_NAME}.log"
    ;;

  deploy|rollout)
    HF_USER="${HF_USER:-$DEFAULT_HF_USER}"

    if [[ "$ARM" == "right" ]]; then
      ROBOT_TYPE="xlerobot_right_arm"
    elif [[ "$ARM" == "left" ]]; then
      ROBOT_TYPE="xlerobot_left_arm"
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

    DISPLAY_ARGS=(--display_data="$DEPLOY_DISPLAY_DATA")
    if [[ -n "$DISPLAY_IP" && -n "$DISPLAY_PORT" ]]; then
      DISPLAY_ARGS+=(--display_ip="$DISPLAY_IP" --display_port="$DISPLAY_PORT")
    fi

    echo "Deployment configuration:"
    echo "  Policy path: ${POLICY_PATH}"
    echo "  Jetson venv: ${JETSON_VENV}"
    echo "  Jetson CUDA lib: ${JETSON_CUDA_LIB}"
    echo "  Task: ${TASK}"
    echo "  Arm: ${ARM}"
    echo "  Robot type: ${ROBOT_TYPE}"
    echo "  Camera preset: ${CAMERA_PRESET}"
    echo "  Rename map: ${RENAME_MAP}"
    echo "  FPS: ${FPS}"
    echo "  Duration: ${ROLLOUT_DURATION}s"
    echo "  Display data: ${DEPLOY_DISPLAY_DATA}"

    export LD_LIBRARY_PATH="${JETSON_CUDA_LIB}:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
    export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    export TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT="${TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT:-0}"

    "$JETSON_PYTHON" -m lerobot.scripts.lerobot_rollout \
      --strategy.type=base \
      --policy.path="$POLICY_PATH" \
      --policy.device=cuda \
      --robot.type="$ROBOT_TYPE" \
      --robot.id=xlerobot \
      --robot.port1=/dev/xlerobot_arm_left \
      --robot.port2=/dev/xlerobot_arm_right \
      --robot.reuse_full_xlerobot_calibration=true \
      "${CAMERA_ARGS[@]}" \
      --rename_map="$RENAME_MAP" \
      --task="$TASK" \
      --fps="$FPS" \
      --duration="$ROLLOUT_DURATION" \
      "${DISPLAY_ARGS[@]}"
    ;;

  *)
    echo "Usage: $0 [check|record|train|smoke-train|mac-train|deploy|rollout]" >&2
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
    echo "  $0 smoke-train" >&2
    echo "  $0 mac-train" >&2
    echo "  DATASET_REPO=local/xlerobot_right_arm_train1 $0 train" >&2
    echo "  DEVICE=cpu TRAIN_STEPS=1 BATCH_SIZE=1 NUM_WORKERS=0 $0 train" >&2
    echo "  $0 deploy" >&2
    echo "  POLICY_PATH=${POLICY_PATH} ROLLOUT_DURATION=10 TASK='$TASK' $0 deploy" >&2
    exit 1
    ;;
esac
