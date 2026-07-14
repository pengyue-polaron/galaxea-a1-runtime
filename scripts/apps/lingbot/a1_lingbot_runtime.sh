#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BASE_RUNTIME="${ROOT}/scripts/runtime/a1_runtime.sh"
CONFIG_PATH="${ROOT}/configs/deployments/lingbot_va.toml"

if [[ "${1:-}" == "--config" ]]; then
  if [[ -z "${2:-}" ]]; then
    echo "Usage: $0 --config <path> <start|server|server-stop|server-logs|services|tmux|stop|doctor|status|logs>" >&2
    exit 2
  fi
  CONFIG_PATH="$2"
  shift 2
fi

PYTHON_BIN="${ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi
BRIDGE_SCRIPT="${ROOT}/scripts/apps/lingbot/lingbot_va_ee_bridge.py"
BRIDGE_GUARD="${ROOT}/scripts/apps/lingbot/a1_lingbot_bridge_guard.sh"
BRIDGE_ENVIRONMENT_CHECKED=false

eval "$(
  PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" -m galaxea_a1_runtime.apps.lingbot.config \
    --repo-root "${ROOT}" \
    --shell \
    "${CONFIG_PATH}"
)"

check_lingbot_server() {
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

check_lingbot_app() {
  if [[ "${WRIST_BACKEND}" == "v4l2" && "${WRIST_CAMERA}" != "auto" && ! -e "${WRIST_CAMERA}" ]]; then
    echo "[FAIL] Wrist camera not found: ${WRIST_CAMERA}" >&2
    exit 2
  fi
  check_lingbot_server
}

check_bridge_environment() {
  if [[ "${BRIDGE_ENVIRONMENT_CHECKED}" == "true" ]]; then
    return
  fi
  if [[ ! -x "${BRIDGE_GUARD}" ]]; then
    echo "[FAIL] LingBot bridge guard is missing or not executable: ${BRIDGE_GUARD}" >&2
    exit 2
  fi
  if ! PYTHONPATH="${ROOT}/third_party/A1_SDK/install/lib/python3/dist-packages:${ROOT}/.cache/ros1_python_overlay:${PYTHONPATH:-}" \
    uv run --project "${ROOT}" python "${BRIDGE_SCRIPT}" --help >/dev/null; then
    echo "[FAIL] LingBot bridge Python environment is incomplete. Run: just setup" >&2
    exit 2
  fi
  BRIDGE_ENVIRONMENT_CHECKED=true
}

link_model_component() {
  local source="$1"
  local destination="$2"
  if [[ -L "${destination}" ]]; then
    if [[ "$(readlink -f "${destination}")" != "$(readlink -f "${source}")" ]]; then
      echo "[FAIL] Model link points somewhere unexpected: ${destination}" >&2
      exit 2
    fi
  elif [[ -e "${destination}" ]]; then
    echo "[FAIL] Refusing to replace existing model component: ${destination}" >&2
    exit 2
  else
    ln -s "${source}" "${destination}"
  fi
}

prepare_model_root() {
  if [[ "${DEPLOYMENT_READY}" != "1" ]]; then
    echo "[FAIL] LingBot deployment_ready=false; update the new checkpoint, prompt, and q01/q99 first." >&2
    exit 2
  fi
  local weight="${MODEL_CHECKPOINT}/transformer/diffusion_pytorch_model.safetensors"
  for required in \
    "${MODEL_PYTHON}" \
    "${BASE_MODEL}/vae" \
    "${BASE_MODEL}/text_encoder" \
    "${BASE_MODEL}/tokenizer" \
    "${MODEL_CHECKPOINT}/transformer/config.json" \
    "${weight}"; do
    if [[ ! -e "${required}" ]]; then
      echo "[FAIL] Missing LingBot model prerequisite: ${required}" >&2
      exit 2
    fi
  done
  local actual_size
  actual_size="$(stat -c '%s' "${weight}")"
  if [[ "${actual_size}" != "${MODEL_EXPECTED_WEIGHT_SIZE}" ]]; then
    echo "[FAIL] LingBot weight size mismatch: expected ${MODEL_EXPECTED_WEIGHT_SIZE}, got ${actual_size}" >&2
    exit 2
  fi
  mkdir -p "${MODEL_ROOT}" "${MODEL_SAVE_ROOT}"
  link_model_component "${BASE_MODEL}/vae" "${MODEL_ROOT}/vae"
  link_model_component "${BASE_MODEL}/text_encoder" "${MODEL_ROOT}/text_encoder"
  link_model_component "${BASE_MODEL}/tokenizer" "${MODEL_ROOT}/tokenizer"
  link_model_component "${MODEL_CHECKPOINT}/transformer" "${MODEL_ROOT}/transformer"
}

start_model_server() {
  prepare_model_root
  if ss -H -ltn "sport = :${LINGBOT_PORT}" | grep -q .; then
    if tmux has-session -t "${MODEL_SESSION}" 2>/dev/null; then
      check_lingbot_server
      echo "LingBot policy server is already running in tmux ${MODEL_SESSION}."
      return
    fi
    echo "[FAIL] ${LINGBOT_HOST}:${LINGBOT_PORT} is occupied by an unmanaged process." >&2
    exit 2
  fi

  tmux kill-session -t "${MODEL_SESSION}" >/dev/null 2>&1 || true
  local server_command=(
    "${MODEL_PYTHON}" -m torch.distributed.run
    --nproc_per_node=1
    --local-ranks-filter=0
    --master_port "${MODEL_MASTER_PORT}"
    --tee 3
    "${ROOT}/scripts/apps/lingbot/lingbot_va_policy_server.py"
    --repo-root "${ROOT}"
    --config "${CONFIG_PATH}"
  )
  local server_command_q
  printf -v server_command_q "%q " "${server_command[@]}"
  tmux new-session -d -s "${MODEL_SESSION}" -c "${MODEL_CHECKOUT}" \
    "bash -lc 'export PYTHONPATH=\"${MODEL_CHECKOUT}:${ROOT}:\${PYTHONPATH:-}\"; export TOKENIZERS_PARALLELISM=false; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; ${server_command_q}; rc=\$?; echo SERVER_EXIT=\$rc; exec bash'"

  local timeout_s="${MODEL_STARTUP_TIMEOUT%.*}"
  local deadline=$((SECONDS + timeout_s))
  while (( SECONDS < deadline )); do
    if curl -fsS --max-time 1 "http://${LINGBOT_HOST}:${LINGBOT_PORT}/healthz" >/dev/null 2>&1; then
      check_lingbot_server
      echo "LingBot policy server is listening on ${LINGBOT_HOST}:${LINGBOT_PORT}."
      return
    fi
    if ! tmux has-session -t "${MODEL_SESSION}" 2>/dev/null; then
      echo "[FAIL] LingBot policy server tmux exited during startup." >&2
      exit 2
    fi
    if tmux capture-pane -pt "${MODEL_SESSION}" -S -20 2>/dev/null | grep -q 'SERVER_EXIT='; then
      tmux capture-pane -pt "${MODEL_SESSION}" -S -80 >&2 || true
      echo "[FAIL] LingBot policy server process exited during startup." >&2
      exit 2
    fi
    sleep 1
  done
  tmux capture-pane -pt "${MODEL_SESSION}" -S -80 >&2 || true
  echo "[FAIL] LingBot policy server did not listen within ${MODEL_STARTUP_TIMEOUT}s." >&2
  exit 2
}

start_services() {
  echo "Using LingBot config: ${CONFIG_PATH}"
  A1_STAGED_COMMAND_TOPIC="${STAGED_TOPIC}" \
    A1_RELAY_ENABLE_TOPIC="${RELAY_ENABLE_TOPIC}" \
    A1_RELAY_STATUS_TOPIC="${RELAY_STATUS_TOPIC}" \
    "${BASE_RUNTIME}" services
}

start_tmux() {
  check_bridge_environment
  check_lingbot_app
  echo "Using LingBot config: ${CONFIG_PATH}"
  tmux kill-session -t "${SESSION}" >/dev/null 2>&1 || true
  local bridge_command=(
    uv run --project "${ROOT}" python "${BRIDGE_SCRIPT}"
    "${BRIDGE_ARGS[@]}"
  )
  local guarded_command=(
    "${BRIDGE_GUARD}" "${BASE_RUNTIME}" "${MODEL_SESSION}" --
    "${bridge_command[@]}"
  )
  local guarded_command_q
  printf -v guarded_command_q "%q " "${guarded_command[@]}"
  tmux new-session -d -s "${SESSION}" -c "${ROOT}" \
    "bash -lc 'export PYTHONPATH=\"${ROOT}/third_party/A1_SDK/install/lib/python3/dist-packages:${ROOT}/.cache/ros1_python_overlay:\${PYTHONPATH:-}\"; ${guarded_command_q}; exec bash'"
  sleep 4
  if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "[FAIL] tmux session exited during startup." >&2
    exit 2
  fi
  local pane
  pane="$(tmux capture-pane -pt "${SESSION}" -S -40)"
  printf '%s\n' "${pane}"
  if grep -q 'BRIDGE_EXIT=' <<<"${pane}"; then
    echo "[FAIL] LingBot bridge exited during startup; runtime cleanup has run." >&2
    exit 2
  fi
}

start_pipeline() {
  cleanup_failed_pipeline() {
    local status=$?
    if [[ "${status}" != "0" ]]; then
      echo "[CLEANUP] Pipeline startup failed; stopping partial LingBot/A1 runtime." >&2
      tmux kill-session -t "${SESSION}" >/dev/null 2>&1 || true
      tmux kill-session -t "${MODEL_SESSION}" >/dev/null 2>&1 || true
      "${BASE_RUNTIME}" stop >/dev/null 2>&1 || true
    fi
  }
  trap cleanup_failed_pipeline EXIT
  check_bridge_environment
  start_model_server
  start_services
  start_tmux
  trap - EXIT
}

doctor() {
  local args=("$@")
  A1_STAGED_COMMAND_TOPIC="${STAGED_TOPIC}" \
    A1_RELAY_ENABLE_TOPIC="${RELAY_ENABLE_TOPIC}" \
    A1_RELAY_STATUS_TOPIC="${RELAY_STATUS_TOPIC}" \
    "${BASE_RUNTIME}" doctor "${args[@]}"
  PYTHONPATH="${ROOT}/third_party/A1_SDK/install/lib/python3/dist-packages:${ROOT}/.cache/ros1_python_overlay:${PYTHONPATH:-}" \
    uv run --project "${ROOT}" python "${ROOT}/scripts/apps/lingbot/a1_lingbot_doctor.py" \
      --lingbot-host "${LINGBOT_HOST}" \
      --lingbot-port "${LINGBOT_PORT}" \
      --wrist-backend "${WRIST_BACKEND}" \
      --wrist-serial "${WRIST_SERIAL}" \
      --wrist-camera "${WRIST_CAMERA}" \
      --staged-command-topic "${STAGED_TOPIC}" \
      --relay-status-topic "${RELAY_STATUS_TOPIC}" \
      "${args[@]}"
}

stop_runtime() {
  tmux kill-session -t "${SESSION}" >/dev/null 2>&1 || true
  tmux kill-session -t "${MODEL_SESSION}" >/dev/null 2>&1 || true
  "${BASE_RUNTIME}" stop
  echo "LingBot A1 bridge and policy server stopped."
}

status() {
  A1_STAGED_COMMAND_TOPIC="${STAGED_TOPIC}" \
    A1_RELAY_ENABLE_TOPIC="${RELAY_ENABLE_TOPIC}" \
    A1_RELAY_STATUS_TOPIC="${RELAY_STATUS_TOPIC}" \
    "${BASE_RUNTIME}" status
  echo
  echo "tmux:"
  tmux list-sessions 2>/dev/null | grep "${SESSION}" || echo "${SESSION}: not running"
  tmux list-sessions 2>/dev/null | grep "${MODEL_SESSION}" || echo "${MODEL_SESSION}: not running"
}

case "${1:-help}" in
  start)
    start_pipeline
    echo
    echo "Attach with: tmux attach -t ${SESSION}"
    ;;
  services)
    start_services
    ;;
  server)
    start_model_server
    ;;
  server-stop)
    tmux kill-session -t "${MODEL_SESSION}" >/dev/null 2>&1 || true
    ;;
  server-logs)
    tmux capture-pane -pt "${MODEL_SESSION}" -S -160
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
Usage: $0 [--config configs/deployments/lingbot_va.toml] <start|server|server-stop|server-logs|services|tmux|stop|doctor|status|logs>

  start     Start the deployment server, A1 base runtime, and bridge
  server    Start only the managed deployment policy server
  server-stop  Stop only the managed policy server
  server-logs  Show recent policy server output
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
