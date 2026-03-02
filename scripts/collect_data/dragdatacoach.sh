#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
A1_SDK_ROOT="${A1_SDK_ROOT:-${PROJECT_ROOT}/third_party/A1_SDK}"

if [[ ! -d "${A1_SDK_ROOT}" ]]; then
  echo "A1_SDK root not found: ${A1_SDK_ROOT}"
  echo "Run: scripts/collect_data/sync_a1_sdk.sh /home/eric/A1_SDK"
  exit 1
fi

set +u
original_args=("$@")
set --
# shellcheck disable=SC1091
if [[ -f "/opt/ros/noetic/setup.bash" ]]; then
  source "/opt/ros/noetic/setup.bash"
fi
# shellcheck disable=SC1091
source "${A1_SDK_ROOT}/install/setup.bash"
set -- "${original_args[@]}"
unset original_args
set -u

usage() {
  cat <<'EOF'
Usage:
  scripts/collect_data/dragdatacoach.sh <command> [args...]

Commands:
  launch-driver [serial]          # default serial=/dev/a1
  launch-ee-record [serial]       # driver + /end_effector_pose publisher
  drag-start [kp] [kd] [mode]
  drag-stop
  gripper-keyboard [args...]
  gripper-open [args...]
  gripper-close [args...]
  gripper-stop                    # stop background keyboard gripper publishers
  record-start [tag]
  record-stop
  launch-tracker
  replay --bag <path> [--rate 1.0] [--gripper-mode position]
  replay-infer --input <path> [--source auto|csv|pkl] [--rate 15] [--speed 1.0] [--loop]
  collect                        # start DataCoach replay bridge + cameras + collector
  require-cameras [context]      # run camera preflight check
  doctor                         # check runtime dependencies and selected python
  which-python                   # print selected DataCoach python
EOF
}

pick_datacoach_python() {
  if [[ -n "${DATACOACH_PYTHON:-}" ]] && [[ -x "${DATACOACH_PYTHON}" ]]; then
    echo "${DATACOACH_PYTHON}"
    return 0
  fi

  local candidates=(
    "${PROJECT_ROOT}/.conda/envs/dragdatacoach/bin/python"
    "/home/pengyue/miniconda3/envs/datacoach/bin/python"
    "/home/jolia/miniconda3/envs/datacoach/bin/python"
    "$(command -v python || true)"
  )

  local py
  for py in "${candidates[@]}"; do
    [[ -n "${py}" && -x "${py}" ]] || continue
    if "${py}" - <<'PY' >/dev/null 2>&1
import importlib.util
required = ("hydra", "zmq", "cv2")
missing = [m for m in required if importlib.util.find_spec(m) is None]
raise SystemExit(1 if missing else 0)
PY
    then
      echo "${py}"
      return 0
    fi
  done
  return 1
}

check_python_deps() {
  local py="$1"
  "${py}" - <<'PY'
import importlib.util
required = ("hydra", "zmq", "cv2", "numpy", "scipy")
optional = ("lerobot", "pyrealsense2")

missing = [m for m in required if importlib.util.find_spec(m) is None]
print("Required:", ", ".join(required))
if missing:
    print("Missing required:", ", ".join(missing))
    raise SystemExit(1)
print("Missing required: <none>")

missing_optional = [m for m in optional if importlib.util.find_spec(m) is None]
if missing_optional:
    print("Missing optional:", ", ".join(missing_optional))
else:
    print("Missing optional: <none>")
PY
}

require_all_cameras() {
  local context="${1:-recording or replay}"
  local py=""
  local camera_cfg="${A1_CAMERA_CHECK_CONFIG:-${PROJECT_ROOT}/configs/drag_replay.yaml}"
  local timeout_s="${A1_CAMERA_CHECK_TIMEOUT_S:-2.0}"

  if [[ "${A1_ALLOW_MISSING_CAMERAS:-0}" == "1" ]]; then
    echo "[WARN] Camera preflight bypass enabled (A1_ALLOW_MISSING_CAMERAS=1)."
    return 0
  fi

  py="$(pick_datacoach_python || true)"
  if [[ -z "${py}" ]]; then
    echo "[ERROR] Could not find a usable DataCoach python for camera checks."
    return 1
  fi

  echo "[CHECK] Verifying required cameras before ${context} ..."
  if ! DATACOACH_PYTHON="${py}" "${py}" "${PROJECT_ROOT}/scripts/collect_data/test_camera_connections.py" \
    --config "${camera_cfg}" \
    --timeout-s "${timeout_s}" \
    --no-save; then
    cat <<EOF
[ERROR] Camera preflight failed.
[ERROR] Both cam_0 and cam_1 must be connected before ${context}.
[ERROR] Fix the camera connection first, then retry.
EOF
    return 1
  fi
}

cmd="${1:-}"
case "${cmd}" in
  launch-driver)
    serial="${2:-/dev/a1}"
    echo "Using serial port: ${serial}"
    roslaunch signal_arm single_arm_node.launch "single_arm_serial_port_path:=${serial}"
    ;;
  launch-ee-record)
    serial="${2:-/dev/a1}"
    echo "Using serial port: ${serial}"
    roslaunch "${A1_SDK_ROOT}/install/share/mobiman/launch/simpleExample/ee_record_only.launch" "serial_port_path:=${serial}"
    ;;
  drag-start)
    shift
    "${A1_SDK_ROOT}/tools/a1_drag_mode.sh" start "$@"
    ;;
  drag-stop)
    "${A1_SDK_ROOT}/tools/a1_drag_mode.sh" stop
    ;;
  gripper-keyboard)
    shift
    "${A1_SDK_ROOT}/tools/a1_gripper_keyboard.py" "$@"
    ;;
  gripper-open)
    shift
    "${PROJECT_ROOT}/scripts/collect_data/a1_gripper_command.py" open "$@"
    ;;
  gripper-close)
    shift
    "${PROJECT_ROOT}/scripts/collect_data/a1_gripper_command.py" close "$@"
    ;;
  gripper-stop)
    if pgrep -f "a1_gripper_keyboard.py" >/dev/null 2>&1; then
      pkill -f "a1_gripper_keyboard.py" || true
      echo "Stopped a1_gripper_keyboard.py"
    else
      echo "No running a1_gripper_keyboard.py found."
    fi
    ;;
  require-cameras)
    shift || true
    require_all_cameras "${*:-replay}"
    ;;
  record-start)
    tag="${2:-drag}"
    if command -v rostopic >/dev/null 2>&1; then
      if rostopic list >/dev/null 2>&1; then
        if ! rostopic list | grep -qx "/end_effector_pose"; then
          cat <<'EOF'
[ERROR] /end_effector_pose is not being published.
[ERROR] This recording will NOT replay arm motion (gripper-only replay).
[ERROR] Start one of these before recording:
  1) scripts/collect_data/dragdatacoach.sh launch-ee-record /dev/a1
  2) roslaunch mobiman jointTrackerdemo.launch
EOF
          if [[ "${A1_RECORD_ALLOW_NO_EE_POSE:-0}" != "1" ]]; then
            echo "[ERROR] Aborting record-start. Set A1_RECORD_ALLOW_NO_EE_POSE=1 to bypass."
            exit 1
          fi
          echo "[WARN] Bypass enabled (A1_RECORD_ALLOW_NO_EE_POSE=1)."
        fi
      fi
    fi
    "${A1_SDK_ROOT}/tools/a1_record.sh" start "${tag}"
    ;;
  record-stop)
    "${A1_SDK_ROOT}/tools/a1_record.sh" stop
    ;;
  launch-tracker)
    roslaunch mobiman jointTrackerdemo.launch
    ;;
  replay)
    shift
    if [[ "${A1_SKIP_CAMERA_PREFLIGHT:-0}" != "1" ]]; then
      require_all_cameras "replay"
    fi
    if pgrep -f "a1_gripper_keyboard.py" >/dev/null 2>&1; then
      cat <<'EOF'
[ERROR] Detected running a1_gripper_keyboard.py.
[ERROR] It continuously publishes gripper commands and will conflict with bag replay.
[ERROR] Stop it first:
  scripts/collect_data/dragdatacoach.sh gripper-stop
EOF
      if [[ "${A1_REPLAY_ALLOW_GRIPPER_KEYBOARD:-0}" != "1" ]]; then
        echo "[ERROR] Aborting replay. Set A1_REPLAY_ALLOW_GRIPPER_KEYBOARD=1 to bypass."
        exit 1
      fi
      echo "[WARN] Bypass enabled (A1_REPLAY_ALLOW_GRIPPER_KEYBOARD=1)."
    fi

    replay_bag_path=""
    replay_rate="1.0"
    replay_gripper_mode="position"
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --bag)
          replay_bag_path="${2:-}"
          shift 2
          ;;
        --rate)
          replay_rate="${2:-1.0}"
          shift 2
          ;;
        --gripper-mode)
          replay_gripper_mode="${2:-position}"
          shift 2
          ;;
        *)
          echo "Unknown replay option: $1"
          echo "Usage: replay --bag <path> [--rate 1.0] [--gripper-mode position]"
          exit 1
          ;;
      esac
    done

    if [[ -z "${replay_bag_path}" ]]; then
      replay_bag_path="$(ls -t "${A1_SDK_ROOT}"/data/records/*.bag 2>/dev/null | head -n 1 || true)"
    fi
    if [[ -z "${replay_bag_path}" || ! -f "${replay_bag_path}" ]]; then
      echo "Replay bag not found: ${replay_bag_path}"
      exit 1
    fi

    if command -v rostopic >/dev/null 2>&1 && rostopic list >/dev/null 2>&1; then
      if ! rostopic list | grep -qx "/arm_joint_target_position"; then
        cat <<'EOF'
[WARN] /arm_joint_target_position topic is missing. jointTrackerdemo may not be running.
[WARN] Start:
  scripts/collect_data/dragdatacoach.sh launch-tracker
EOF
      fi
    fi

    bag_has_gripper_topic=0
    replay_gripper_topic="/gripper_position_control/host"
    replay_gripper_remap=1
    if command -v rosbag >/dev/null 2>&1; then
      if ! rosbag info "${replay_bag_path}" | awk '/topics:/{p=1;next} p && $1=="/joint_states_host"{found=1} END{exit !found}'; then
        cat <<'EOF'
[WARN] Bag does not contain /joint_states_host.
[WARN] Arm trajectory may not replay.
EOF
      fi
      if rosbag info "${replay_bag_path}" | awk '/topics:/{p=1;next} p && $1=="/gripper_position_control_host"{found=1} END{exit !found}'; then
        bag_has_gripper_topic=1
        replay_gripper_topic="/gripper_position_control_host"
        replay_gripper_remap=0
      elif rosbag info "${replay_bag_path}" | awk '/topics:/{p=1;next} p && $1=="/gripper_position_control/host"{found=1} END{exit !found}'; then
        bag_has_gripper_topic=1
        replay_gripper_topic="/gripper_position_control/host"
        replay_gripper_remap=1
      fi
    fi

    if [[ "${bag_has_gripper_topic}" -eq 0 ]]; then
      cat <<'EOF'
[WARN] Bag does not contain gripper position control topic.
[WARN] Supported: /gripper_position_control_host or /gripper_position_control/host
[WARN] Gripper may not replay.
EOF
    fi

    echo "Replay mode: gripper-mode=${replay_gripper_mode} (kept for compatibility)"
    if [[ "${replay_gripper_remap}" -eq 1 ]]; then
      rosbag play "${replay_bag_path}" \
        --topics /joint_states_host "${replay_gripper_topic}" \
        /joint_states_host:=/arm_joint_target_position \
        /gripper_position_control/host:=/gripper_position_control_host \
        -r "${replay_rate}"
    else
      rosbag play "${replay_bag_path}" \
        --topics /joint_states_host "${replay_gripper_topic}" \
        /joint_states_host:=/arm_joint_target_position \
        -r "${replay_rate}"
    fi
    ;;
  replay-infer)
    shift
    py="$(pick_datacoach_python || true)"
    if [[ -z "${py}" ]]; then
      echo "Could not find a usable DataCoach python interpreter."
      echo "Set DATACOACH_PYTHON explicitly, or prepare env with hydra/zmq/cv2."
      exit 1
    fi
    DATACOACH_PYTHON="${py}" "${py}" "${PROJECT_ROOT}/scripts/collect_data/replay_inferred_trajectory.py" "$@"
    ;;
  collect)
    py="$(pick_datacoach_python || true)"
    if [[ -z "${py}" ]]; then
      echo "Could not find a usable DataCoach python interpreter."
      echo "Set DATACOACH_PYTHON explicitly, or prepare env with hydra/zmq/cv2."
      exit 1
    fi
    DATACOACH_PYTHON="${py}" "${py}" "${PROJECT_ROOT}/scripts/collect_data/run_drag_replay_collection.py"
    ;;
  doctor)
    echo "A1_SDK_ROOT=${A1_SDK_ROOT}"
    if ! command -v roscore >/dev/null 2>&1; then
      echo "ROS tools not found in PATH. Source ROS setup first."
    fi
    py="$(pick_datacoach_python || true)"
    if [[ -z "${py}" ]]; then
      echo "No usable DataCoach python found."
      exit 1
    fi
    echo "DATACOACH_PYTHON=${py}"
    check_python_deps "${py}"
    ;;
  which-python)
    py="$(pick_datacoach_python || true)"
    if [[ -z "${py}" ]]; then
      echo "NOT_FOUND"
      exit 1
    fi
    echo "${py}"
    ;;
  *)
    usage
    exit 1
    ;;
esac
