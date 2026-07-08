#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BASE_RUNTIME="${ROOT}/scripts/runtime/a1_runtime.sh"
CONFIG_PATH="${A1_LINGBOT_CONFIG:-${ROOT}/configs/inference/lingbot_va_a1.toml}"

if [[ "${1:-}" == "--config" ]]; then
  if [[ -z "${2:-}" ]]; then
    echo "Usage: $0 --config <path> <start|services|tmux|stop|doctor|status|logs>" >&2
    exit 2
  fi
  CONFIG_PATH="$2"
  shift 2
fi

PYTHON_BIN="${ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

eval "$(
  PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" -m galaxea_a1_runtime.apps.lingbot.config \
    --repo-root "${ROOT}" \
    --shell \
    "${CONFIG_PATH}"
)"

check_lingbot_app() {
  if [[ "${WRIST_CAMERA}" != "auto" && ! -e "${WRIST_CAMERA}" ]]; then
    echo "[FAIL] Wrist camera not found: ${WRIST_CAMERA}" >&2
    exit 2
  fi
  if ! uv run --project "${ROOT}" python - "${LINGBOT_HOST}" "${LINGBOT_PORT}" >/dev/null 2>&1 <<'PY'
import sys
import websockets.sync.client

host = sys.argv[1]
port = int(sys.argv[2])
with websockets.sync.client.connect(
    f"ws://{host}:{port}",
    compression=None,
    max_size=None,
    ping_interval=None,
    close_timeout=2,
    open_timeout=2,
) as ws:
    ws.recv(timeout=2)
PY
  then
    echo "[FAIL] LingBot server is not listening on ${LINGBOT_HOST}:${LINGBOT_PORT}." >&2
    exit 2
  fi
}

start_services() {
  echo "Using LingBot config: ${CONFIG_PATH}"
  A1_STAGED_COMMAND_TOPIC="${STAGED_TOPIC}" \
    A1_RELAY_ENABLE_TOPIC="${RELAY_ENABLE_TOPIC}" \
    A1_RELAY_STATUS_TOPIC="${RELAY_STATUS_TOPIC}" \
    "${BASE_RUNTIME}" services
}

start_tmux() {
  check_lingbot_app
  echo "Using LingBot config: ${CONFIG_PATH}"
  tmux kill-session -t "${SESSION}" >/dev/null 2>&1 || true
  local bridge_command=(
    uv run --project "${ROOT}" python "${ROOT}/scripts/apps/lingbot/lingbot_va_ee_bridge.py"
    "${BRIDGE_ARGS[@]}"
  )
  local bridge_command_q
  printf -v bridge_command_q "%q " "${bridge_command[@]}"
  tmux new-session -d -s "${SESSION}" -c "${ROOT}" \
    "bash -lc 'export PYTHONPATH=\"${ROOT}/.cache/ros1_python_overlay:${ROOT}/third_party/A1_SDK/install/lib/python3/dist-packages:\${PYTHONPATH:-}\"; ${bridge_command_q}; rc=\$?; echo BRIDGE_EXIT=\$rc; exec bash'"
  sleep 4
  if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "[FAIL] tmux session exited during startup." >&2
    exit 2
  fi
  tmux capture-pane -pt "${SESSION}" -S -40
}

doctor() {
  local args=("$@")
  A1_STAGED_COMMAND_TOPIC="${STAGED_TOPIC}" \
    A1_RELAY_ENABLE_TOPIC="${RELAY_ENABLE_TOPIC}" \
    A1_RELAY_STATUS_TOPIC="${RELAY_STATUS_TOPIC}" \
    "${BASE_RUNTIME}" doctor "${args[@]}"
  PYTHONPATH="${ROOT}/.cache/ros1_python_overlay:${ROOT}/third_party/A1_SDK/install/lib/python3/dist-packages:${PYTHONPATH:-}" \
    uv run --project "${ROOT}" python "${ROOT}/scripts/apps/lingbot/a1_lingbot_doctor.py" \
      --lingbot-host "${LINGBOT_HOST}" \
      --lingbot-port "${LINGBOT_PORT}" \
      --wrist-camera "${WRIST_CAMERA}" \
      --staged-command-topic "${STAGED_TOPIC}" \
      --relay-status-topic "${RELAY_STATUS_TOPIC}" \
      "${args[@]}"
}

stop_runtime() {
  tmux kill-session -t "${SESSION}" >/dev/null 2>&1 || true
  "${BASE_RUNTIME}" stop
  echo "LingBot A1 bridge stopped."
}

status() {
  A1_STAGED_COMMAND_TOPIC="${STAGED_TOPIC}" \
    A1_RELAY_ENABLE_TOPIC="${RELAY_ENABLE_TOPIC}" \
    A1_RELAY_STATUS_TOPIC="${RELAY_STATUS_TOPIC}" \
    "${BASE_RUNTIME}" status
  echo
  echo "tmux:"
  tmux list-sessions 2>/dev/null | grep "${SESSION}" || echo "${SESSION}: not running"
}

case "${1:-help}" in
  start)
    check_lingbot_app
    start_services
    start_tmux
    echo
    echo "Attach with: tmux attach -t ${SESSION}"
    ;;
  services)
    start_services
    ;;
  tmux)
    start_tmux
    ;;
  stop)
    stop_runtime
    ;;
  doctor)
    shift
    doctor "$@"
    ;;
  status)
    status
    ;;
  logs)
    "${BASE_RUNTIME}" logs
    ;;
  *)
    cat <<EOF
Usage: $0 [--config configs/inference/lingbot_va_a1.toml] <start|services|tmux|stop|doctor|status|logs>

  start     Start A1 base runtime, then open the interactive LingBot bridge tmux
  services  Start only the decoupled A1 base runtime
  tmux      Start only the interactive LingBot bridge
  stop      Stop the LingBot tmux and A1 base runtime
  doctor    Run base runtime checks plus LingBot app checks
  status    Base runtime status plus tmux state
  logs      Tail base runtime logs

Config:
  ${CONFIG_PATH}
EOF
    ;;
esac
