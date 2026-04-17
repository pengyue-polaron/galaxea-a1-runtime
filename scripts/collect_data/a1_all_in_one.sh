#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SESSION="a1-collect"
SERIAL="/dev/ttyACM0"
TAG="drag_demo"
BAG=""
RATE="1.0"
GRIPPER_MODE="position"
SKIP_RECORD=0
WITH_GRIPPER_KEYBOARD=1
AUTO_STOP=1
ON_EXISTING="restart"
RUN_STATUS_DIR="$(mktemp -d "${TMPDIR:-/tmp}/a1-collect.XXXXXX")"
COLLECT_STATUS_FILE="${RUN_STATUS_DIR}/collect.exit"
COLLECT_PID_FILE="${RUN_STATUS_DIR}/collect.pid"
COLLECT_LOG_FILE="${RUN_STATUS_DIR}/collect.log"
KEEP_RUN_STATUS_DIR=0

usage() {
  cat <<'EOF'
Usage:
  scripts/collect_data/a1_all_in_one.sh [options]

Options:
  --serial <path>            Serial device for A1 driver (default: /dev/ttyACM0)
  --tag <name>               Bag record tag for drag stage (default: drag_demo)
  --bag <path>               Replay bag path. If omitted, use latest recorded bag.
  --rate <float>             Replay rate (default: 1.0)
  --gripper-mode <mode>      Replay gripper mode (default: position)
  --session <name>           tmux session name (default: a1-collect)
  --skip-record              Skip drag recording stage, replay from provided/latest bag.
  --no-gripper-keyboard      Do not launch keyboard gripper during drag stage.
  --no-auto-stop             Keep replay launch windows running at the end.
  --on-existing <policy>     Existing tmux session policy:
                             ask | restart | attach | new | abort
                             (default: restart)
  -h, --help                 Show this help.
EOF
}

cleanup() {
  local rc=$?
  if [[ "${KEEP_RUN_STATUS_DIR}" -eq 1 || "${rc}" -ne 0 ]]; then
    echo "[INFO] Preserving collector artifacts: ${RUN_STATUS_DIR}"
    if [[ -f "${COLLECT_LOG_FILE}" ]]; then
      echo "[INFO] Collector log: ${COLLECT_LOG_FILE}"
    fi
    return
  fi
  rm -rf "${RUN_STATUS_DIR}"
}

trap cleanup EXIT

next_session_name() {
  local base="$1"
  local idx=2
  local candidate
  candidate="${base}_${idx}"
  while tmux has-session -t "${candidate}" 2>/dev/null; do
    idx=$((idx + 1))
    candidate="${base}_${idx}"
  done
  echo "${candidate}"
}

handle_existing_session() {
  local base_session="$1"
  local policy="$2"
  local choice=""

  case "${policy}" in
    attach)
      tmux attach -t "${base_session}"
      exit 0
      ;;
    restart)
      echo "[INFO] Removing existing session: ${base_session}"
      tmux kill-session -t "${base_session}"
      return
      ;;
    new)
      SESSION="$(next_session_name "${base_session}")"
      echo "[INFO] Existing session detected, switching to: ${SESSION}"
      return
      ;;
    abort)
      echo "tmux session '${base_session}' already exists."
      echo "Use --on-existing <restart|attach|new>."
      exit 1
      ;;
    ask)
      if [[ -t 0 && -t 1 ]]; then
        echo "tmux session '${base_session}' already exists."
        echo "Choose action: [r]estart, [a]ttach, [n]ew, [q]uit"
        read -r -p "Action (default: r): " choice
      else
        choice="r"
      fi

      case "${choice}" in
        ""|r|R)
          echo "[INFO] Removing existing session: ${base_session}"
          tmux kill-session -t "${base_session}"
          ;;
        a|A)
          tmux attach -t "${base_session}"
          exit 0
          ;;
        n|N)
          SESSION="$(next_session_name "${base_session}")"
          echo "[INFO] Existing session detected, switching to: ${SESSION}"
          ;;
        q|Q)
          echo "Abort."
          exit 1
          ;;
        *)
          echo "Unknown choice: ${choice}"
          exit 1
          ;;
      esac
      ;;
    *)
      echo "Invalid --on-existing policy: ${policy}"
      echo "Valid values: ask, restart, attach, new, abort"
      exit 1
      ;;
  esac
}

pick_latest_bag() {
  ls -t "${PROJECT_ROOT}"/third_party/A1_SDK/data/records/*.bag 2>/dev/null | head -n 1 || true
}

pick_latest_bag_since() {
  local since_ts="$1"
  local tag="$2"
  local bag=""
  local best_m=0
  local f
  local m
  local name

  shopt -s nullglob
  for f in "${PROJECT_ROOT}"/third_party/A1_SDK/data/records/*.bag; do
    name="$(basename "${f}")"
    if [[ "${name}" != *"${tag}"* ]]; then
      continue
    fi
    m="$(stat -c %Y "${f}" 2>/dev/null || echo 0)"
    if (( m >= since_ts && m > best_m )); then
      best_m="${m}"
      bag="${f}"
    fi
  done

  if [[ -z "${bag}" ]]; then
    for f in "${PROJECT_ROOT}"/third_party/A1_SDK/data/records/*.bag; do
      m="$(stat -c %Y "${f}" 2>/dev/null || echo 0)"
      if (( m >= since_ts && m > best_m )); then
        best_m="${m}"
        bag="${f}"
      fi
    done
  fi
  shopt -u nullglob

  echo "${bag}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --serial)
      SERIAL="$2"
      shift 2
      ;;
    --tag)
      TAG="$2"
      shift 2
      ;;
    --bag)
      BAG="$2"
      shift 2
      ;;
    --rate)
      RATE="$2"
      shift 2
      ;;
    --gripper-mode)
      GRIPPER_MODE="$2"
      shift 2
      ;;
    --session)
      SESSION="$2"
      shift 2
      ;;
    --skip-record)
      SKIP_RECORD=1
      shift
      ;;
    --no-gripper-keyboard)
      WITH_GRIPPER_KEYBOARD=0
      shift
      ;;
    --no-auto-stop)
      AUTO_STOP=0
      shift
      ;;
    --on-existing)
      ON_EXISTING="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

if ! command -v just >/dev/null 2>&1; then
  echo "just is not installed or not in PATH."
  exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed. Install tmux first."
  exit 1
fi

cd "${PROJECT_ROOT}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  handle_existing_session "${SESSION}" "${ON_EXISTING}"
fi

tmux_new_window() {
  local window_name="$1"
  local command="$2"
  tmux new-window -t "${SESSION}:" -n "${window_name}" "cd '${PROJECT_ROOT}'; ${command}"
}

tmux_ctrl_c() {
  local window_name="$1"
  tmux send-keys -t "${SESSION}:${window_name}" C-c || true
}

read_status_file() {
  local status_file="$1"
  local timeout_s="${2:-10}"
  local waited=0

  while [[ ! -f "${status_file}" && "${waited}" -lt $((timeout_s * 10)) ]]; do
    sleep 0.1
    waited=$((waited + 1))
  done

  if [[ ! -f "${status_file}" ]]; then
    echo "[ERROR] Timed out waiting for status file: ${status_file}" >&2
    return 1
  fi

  cat "${status_file}"
}

collect_window_alive() {
  tmux list-panes -t "${SESSION}:collect" >/dev/null 2>&1
}

tmux_kill_window() {
  local window_name="$1"
  tmux kill-window -t "${SESSION}:${window_name}" >/dev/null 2>&1 || true
}

print_collect_failure_details() {
  if [[ ! -f "${COLLECT_LOG_FILE}" ]]; then
    echo "[ERROR] Collector log file not found: ${COLLECT_LOG_FILE}"
    return
  fi

  local detail=""
  detail="$(grep -E "Validation failed|aborting save|ended too early|coverage too short|Missing camera frames|captured too few frames|No robot states captured|No commanded states captured|produced no frames|stopped producing frames|read failed|camera_server failed" "${COLLECT_LOG_FILE}" | tail -n 1 || true)"
  if [[ -n "${detail}" ]]; then
    echo "[ERROR] ${detail}"
    echo "[ERROR] Collector log preserved at: ${COLLECT_LOG_FILE}"
    return
  fi

  echo "[ERROR] Last collector log lines:"
  tail -n 20 "${COLLECT_LOG_FILE}" || true
  echo "[ERROR] Collector log preserved at: ${COLLECT_LOG_FILE}"
}

collect_failure_detail() {
  if [[ ! -f "${COLLECT_LOG_FILE}" ]]; then
    return 0
  fi
  grep -E "Validation failed|aborting save|ended too early|coverage too short|Missing camera frames|captured too few frames|No robot states captured|No commanded states captured|produced no frames|stopped producing frames|read failed|camera_server failed" "${COLLECT_LOG_FILE}" | tail -n 1 || true
}

is_retryable_camera_failure() {
  local detail="${1:-}"
  [[ "${detail}" == *"Camera preflight failed"* ]] \
    || [[ "${detail}" == *"Both cam_0 and cam_1 must be connected"* ]] \
    || \
  [[ "${detail}" == *"camera_server failed:"* ]] \
    || [[ "${detail}" == *"Camera cam_"* ]] \
    || [[ "${detail}" == *"Missing camera frames"* ]] \
    || [[ "${detail}" == *"produced no frames"* ]] \
    || [[ "${detail}" == *"stopped producing frames"* ]] \
    || [[ "${detail}" == *"read failed"* ]]
}

stop_stage2_windows() {
  tmux_ctrl_c "collect"
  tmux_ctrl_c "tracker"
  tmux_ctrl_c "replay_driver"
  sleep 1
  tmux_kill_window "collect"
  tmux_kill_window "tracker"
  tmux_kill_window "replay_driver"
}

prompt_retry_same_replay() {
  local reason="$1"
  if ! is_retryable_camera_failure "${reason}"; then
    return 1
  fi
  if [[ ! -t 0 || ! -t 1 ]]; then
    echo "[ERROR] Camera-related replay failure is retryable, but no interactive terminal is available."
    return 1
  fi

  echo "[WARN] Camera-related replay failure detected."
  if [[ -n "${reason}" ]]; then
    echo "[WARN] ${reason}"
  fi
  echo "[WARN] Reconnect the camera, then press Enter to replay the same bag again."

  local choice=""
  read -r -p "Press Enter to retry replay, or type q to abort: " choice
  [[ ! "${choice}" =~ ^[Qq]$ ]]
}

run_stage2_attempt() {
  local attempt="$1"
  local collect_rc=""
  local replay_pid=""
  local replay_rc=0
  local detail=""

  echo "[Attempt ${attempt}] Starting replay services..."
  tmux_new_window "replay_driver" "just launch ee-record '${SERIAL}'"
  sleep 2
  tmux_new_window "tracker" "just launch tracker"
  sleep 2
  if ! scripts/collect_data/a1.sh require-cameras "replay attempt ${attempt}"; then
    KEEP_RUN_STATUS_DIR=1
    stop_stage2_windows
    if prompt_retry_same_replay "Camera preflight failed for replay attempt ${attempt}."; then
      return 10
    fi
    return 1
  fi
  rm -f "${COLLECT_STATUS_FILE}" "${COLLECT_PID_FILE}" "${COLLECT_LOG_FILE}"
  tmux_new_window "collect" "${collect_cmd}"
  sleep 4
  if [[ -f "${COLLECT_STATUS_FILE}" ]]; then
    collect_rc="$(cat "${COLLECT_STATUS_FILE}")"
    echo "[ERROR] Collector exited before replay started (rc=${collect_rc})."
    KEEP_RUN_STATUS_DIR=1
    detail="$(collect_failure_detail)"
    print_collect_failure_details
    stop_stage2_windows
    if prompt_retry_same_replay "${detail}"; then
      return 10
    fi
    return "${collect_rc:-1}"
  fi

  read -r -p "Press Enter to START replay..."
  if [[ -f "${COLLECT_STATUS_FILE}" ]] || ! collect_window_alive; then
    collect_rc="$(read_status_file "${COLLECT_STATUS_FILE}" 5 || echo 1)"
    echo "[ERROR] Collector is not running before replay start (rc=${collect_rc})."
    KEEP_RUN_STATUS_DIR=1
    detail="$(collect_failure_detail)"
    print_collect_failure_details
    stop_stage2_windows
    if prompt_retry_same_replay "${detail}"; then
      return 10
    fi
    return "${collect_rc}"
  fi
  tmux send-keys -t "${SESSION}:collect" Enter
  sleep 1

  A1_SKIP_CAMERA_PREFLIGHT=1 just replay "${BAG}" "${RATE}" "${GRIPPER_MODE}" &
  replay_pid=$!
  while kill -0 "${replay_pid}" >/dev/null 2>&1; do
    if [[ -f "${COLLECT_STATUS_FILE}" ]]; then
      collect_rc="$(cat "${COLLECT_STATUS_FILE}" 2>/dev/null || echo 1)"
      echo "[ERROR] Collector exited during replay (rc=${collect_rc}). Stopping replay..."
      KEEP_RUN_STATUS_DIR=1
      kill -INT "${replay_pid}" >/dev/null 2>&1 || true
      set +e
      wait "${replay_pid}"
      replay_rc=$?
      set -e
      detail="$(collect_failure_detail)"
      print_collect_failure_details
      stop_stage2_windows
      if prompt_retry_same_replay "${detail}"; then
        return 10
      fi
      return "${collect_rc:-1}"
    fi
    sleep 0.2
  done
  set +e
  wait "${replay_pid}"
  replay_rc=$?
  set -e
  if [[ "${replay_rc}" != "0" ]]; then
    echo "[ERROR] Replay failed (rc=${replay_rc})."
    KEEP_RUN_STATUS_DIR=1
    stop_stage2_windows
    return "${replay_rc}"
  fi

  echo "Replay finished. Stopping collector..."
  if [[ -f "${COLLECT_PID_FILE}" ]]; then
    kill -INT "$(cat "${COLLECT_PID_FILE}")" >/dev/null 2>&1 || true
  else
    tmux_ctrl_c "collect"
  fi
  sleep 2
  collect_rc="$(read_status_file "${COLLECT_STATUS_FILE}" 15 || echo 1)"
  if [[ "${collect_rc}" != "0" ]]; then
    echo "[ERROR] Collector failed (rc=${collect_rc}). Check tmux window '${SESSION}:collect'."
    KEEP_RUN_STATUS_DIR=1
    detail="$(collect_failure_detail)"
    print_collect_failure_details
    stop_stage2_windows
    if prompt_retry_same_replay "${detail}"; then
      return 10
    fi
    return "${collect_rc}"
  fi

  tmux_ctrl_c "replay_driver"
  sleep 2
  tmux send-keys -t "${SESSION}:replay_driver" "just launch driver '${SERIAL}'" Enter
  sleep 2

  if [[ "${AUTO_STOP}" -eq 1 ]]; then
    tmux_ctrl_c "tracker"
    tmux_ctrl_c "replay_driver"
  fi

  return 0
}

collect_cmd="scripts/collect_data/run_collect_supervised.sh '${COLLECT_STATUS_FILE}' '${COLLECT_PID_FILE}' '${COLLECT_LOG_FILE}'"
if ! id -nG | grep -qw video; then
  video_members="$(getent group video | awk -F: '{print $4}' || true)"
  if [[ ",${video_members}," == *",${USER},"* ]]; then
    collect_cmd="sg video -c 'cd \"${PROJECT_ROOT}\" && scripts/collect_data/run_collect_supervised.sh \"${COLLECT_STATUS_FILE}\" \"${COLLECT_PID_FILE}\" \"${COLLECT_LOG_FILE}\"'"
  else
    echo "[WARN] Current user is not in video group. Camera permissions may fail."
    echo "[WARN] Run: sudo usermod -aG video ${USER} && relogin"
  fi
fi

tmux new-session -d -s "${SESSION}" -n control "cd '${PROJECT_ROOT}'; bash"

echo "[Stage 1/2] Drag Record"
if [[ "${SKIP_RECORD}" -eq 0 ]]; then
  tmux_new_window "record_driver" "just launch ee-record '${SERIAL}'"
  sleep 3

  # Previous interrupted runs may leave drag mode alive outside tmux.
  # Always clear it before starting a new drag session.
  just drag stop >/dev/null 2>&1 || true
  if [[ "${WITH_GRIPPER_KEYBOARD}" -eq 1 ]]; then
    A1_DRAG_HOLD_GRIPPER_POSITION=0 just drag start
  else
    just drag start
  fi

  read -r -p "Press Enter to START bag recording..."
  record_start_ts="$(date +%s)"
  just record start "${TAG}"

  if [[ "${WITH_GRIPPER_KEYBOARD}" -eq 1 ]]; then
    echo "Recording now. Control gripper with keyboard in THIS terminal."
    echo "Press Enter to stop recording and exit gripper control."
    just gripper start --quit-on-enter || true
    just record stop || true
  else
    echo "Recording now. Move the arm. Press Enter when finished."
    read -r -p "Press Enter to STOP bag recording..."
    just record stop || true
  fi
  just drag stop || true
  just gripper stop >/dev/null 2>&1 || true

  if [[ -z "${BAG}" ]]; then
    BAG="$(pick_latest_bag_since "${record_start_ts}" "${TAG}")"
    if [[ -z "${BAG}" ]]; then
      BAG="$(pick_latest_bag)"
    fi
  fi
  echo "Recorded bag: ${BAG}"

  tmux_ctrl_c "record_driver"
  sleep 2
else
  echo "Skip record stage enabled."
fi

if [[ -z "${BAG}" ]]; then
  BAG="$(pick_latest_bag)"
fi

if [[ -z "${BAG}" || ! -f "${BAG}" ]]; then
  echo "Replay bag not found: ${BAG}"
  exit 1
fi

echo "[Stage 2/2] Replay + Collect"
echo "Using bag: ${BAG}"

attempt=1
while true; do
  if run_stage2_attempt "${attempt}"; then
    break
  fi
  stage2_rc=$?
  if [[ "${stage2_rc}" -eq 10 ]]; then
    attempt=$((attempt + 1))
    continue
  fi
  exit "${stage2_rc}"
done

echo
echo "All-in-one flow finished."
echo "tmux session: ${SESSION}"
echo "Attach logs: tmux attach -t ${SESSION}"
echo "Cleanup:     tmux kill-session -t ${SESSION}"
