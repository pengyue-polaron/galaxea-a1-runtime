#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${ROOT}/scripts/runtime/a1_config.sh"
BASE_RUNTIME="${ROOT}/scripts/runtime/a1_runtime.sh"
CONFIG_PATH=""
source "${ROOT}/scripts/runtime/a1_tmux.sh"

if [[ "${1:-}" == "--config" ]]; then
  if [[ -z "${2:-}" ]]; then
    a1_fail "--config requires a path."
    a1_usage "$0 --config <path> <start|server|server-stop|server-logs|services|tmux|stop|doctor|status|logs>" >&2
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

config_args=(--repo-root "${ROOT}" --shell)
if [[ -n "${CONFIG_PATH}" ]]; then
  config_args+=("${CONFIG_PATH}")
fi
a1_load_shell_config env \
  PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" -m galaxea_a1_runtime.apps.lingbot.config \
    "${config_args[@]}"
export A1_SYSTEM_CONFIG_PATH="${SYSTEM_CONFIG_PATH}"

check_lingbot_server() {
  if ! uv run --project "${ROOT}" python - \
    "${LINGBOT_HOST}" "${LINGBOT_PORT}" "${LINGBOT_CONNECT_TIMEOUT}" \
    >/dev/null 2>&1 <<'PY'
import sys
import websockets.sync.client

host = sys.argv[1]
port = int(sys.argv[2])
timeout = float(sys.argv[3])
with websockets.sync.client.connect(
    f"ws://{host}:{port}",
    compression=None,
    max_size=None,
    ping_interval=None,
    close_timeout=timeout,
    open_timeout=timeout,
) as ws:
    ws.recv(timeout=timeout)
PY
  then
    a1_fail "LingBot server is not listening on ${LINGBOT_HOST}:${LINGBOT_PORT}."
    exit 2
  fi
}

check_lingbot_app() {
  if [[ "${WRIST_BACKEND}" == "v4l2" && "${WRIST_CAMERA}" != "auto" && ! -e "${WRIST_CAMERA}" ]]; then
    a1_fail "Wrist camera not found: ${WRIST_CAMERA}"
    exit 2
  fi
  check_lingbot_server
}

check_bridge_environment() {
  if [[ "${BRIDGE_ENVIRONMENT_CHECKED}" == "true" ]]; then
    return
  fi
  if [[ ! -x "${BRIDGE_GUARD}" ]]; then
    a1_fail "LingBot bridge guard is missing or not executable: ${BRIDGE_GUARD}"
    exit 2
  fi
  if ! PYTHONPATH="${ROOT}/third_party/A1_SDK/install/lib/python3/dist-packages:${ROOT}/.cache/ros1_python_overlay:${PYTHONPATH:-}" \
    uv run --project "${ROOT}" python "${BRIDGE_SCRIPT}" --help >/dev/null; then
    a1_fail "LingBot bridge Python environment is incomplete. Run: just setup"
    exit 2
  fi
  BRIDGE_ENVIRONMENT_CHECKED=true
}

link_model_component() {
  local source="$1"
  local destination="$2"
  if [[ -L "${destination}" ]]; then
    if [[ "$(readlink -f "${destination}")" != "$(readlink -f "${source}")" ]]; then
      a1_fail "Model link points somewhere unexpected: ${destination}"
      exit 2
    fi
  elif [[ -e "${destination}" ]]; then
    a1_fail "Refusing to replace existing model component: ${destination}"
    exit 2
  else
    ln -s "${source}" "${destination}"
  fi
}

prepare_model_root() {
  if [[ "${DEPLOYMENT_READY}" != "1" ]]; then
    a1_fail "LingBot deployment_ready=false; update the new checkpoint, prompt, and q01/q99 first."
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
      a1_fail "Missing LingBot model prerequisite: ${required}"
      exit 2
    fi
  done
  local actual_size
  actual_size="$(stat -c '%s' "${weight}")"
  if [[ "${actual_size}" != "${MODEL_EXPECTED_WEIGHT_SIZE}" ]]; then
    a1_fail "LingBot weight size mismatch: expected ${MODEL_EXPECTED_WEIGHT_SIZE}, got ${actual_size}"
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
    if a1_tmux_has_session "${MODEL_SESSION}"; then
      check_lingbot_server
      a1_info "LingBot policy server is already running in tmux ${MODEL_SESSION}."
      return
    fi
    a1_fail "${LINGBOT_HOST}:${LINGBOT_PORT} is occupied by an unmanaged process."
    exit 2
  fi

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
  a1_tmux_start "${MODEL_SESSION}" "${MODEL_CHECKOUT}" \
    "bash -lc 'export PYTHONPATH=\"${MODEL_CHECKOUT}:${ROOT}:\${PYTHONPATH:-}\"; export TOKENIZERS_PARALLELISM=false; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; ${server_command_q}; rc=\$?; echo SERVER_EXIT=\$rc; exec bash'"

  local timeout_s="${MODEL_STARTUP_TIMEOUT%.*}"
  local deadline=$((SECONDS + timeout_s))
  while (( SECONDS < deadline )); do
    if curl -fsS --max-time 1 "http://${LINGBOT_HOST}:${LINGBOT_PORT}/healthz" >/dev/null 2>&1; then
      check_lingbot_server
      a1_success "LingBot policy server is listening on ${LINGBOT_HOST}:${LINGBOT_PORT}."
      return
    fi
    if ! a1_tmux_has_session "${MODEL_SESSION}"; then
      a1_fail "LingBot policy server tmux exited during startup."
      exit 2
    fi
    if a1_tmux_capture "${MODEL_SESSION}" 20 | grep -q 'SERVER_EXIT='; then
      a1_tmux_capture "${MODEL_SESSION}" 80 >&2 || true
      a1_fail "LingBot policy server process exited during startup."
      exit 2
    fi
    sleep 1
  done
  a1_tmux_capture "${MODEL_SESSION}" 80 >&2 || true
  a1_fail "LingBot policy server did not listen within ${MODEL_STARTUP_TIMEOUT}s."
  exit 2
}

start_services() {
  a1_info "Config: ${CONFIG_PATH}"
  "${BASE_RUNTIME}" services
}

start_tmux() {
  check_bridge_environment
  check_lingbot_app
  a1_info "Config: ${CONFIG_PATH}"
  local bridge_command=(
    uv run --project "${ROOT}" python "${BRIDGE_SCRIPT}"
    --config "${CONFIG_PATH}"
  )
  local guarded_command=(
    "${BRIDGE_GUARD}" "${BASE_RUNTIME}" "${MODEL_SESSION}" --
    "${bridge_command[@]}"
  )
  local guarded_command_q
  printf -v guarded_command_q "%q " "${guarded_command[@]}"
  a1_tmux_start "${SESSION}" "${ROOT}" \
    "bash -lc 'export PYTHONPATH=\"${ROOT}/third_party/A1_SDK/install/lib/python3/dist-packages:${ROOT}/.cache/ros1_python_overlay:\${PYTHONPATH:-}\"; ${guarded_command_q}; exec bash'"
  if ! a1_tmux_verify_startup \
    "${SESSION}" "BRIDGE_EXIT=" "LingBot bridge" "${TMUX_STARTUP_GRACE_S}" 40; then
    a1_fail "LingBot runtime cleanup has run."
    exit 2
  fi
}

start_pipeline() {
  cleanup_failed_pipeline() {
    local status=$?
    if [[ "${status}" != "0" ]]; then
      a1_cleanup "Pipeline startup failed; stopping partial LingBot/A1 runtime."
      a1_tmux_stop "${SESSION}"
      a1_tmux_stop "${MODEL_SESSION}"
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
  "${BASE_RUNTIME}" doctor "${args[@]}"
  PYTHONPATH="${ROOT}/third_party/A1_SDK/install/lib/python3/dist-packages:${ROOT}/.cache/ros1_python_overlay:${PYTHONPATH:-}" \
    uv run --project "${ROOT}" python "${ROOT}/scripts/apps/lingbot/a1_lingbot_doctor.py" \
      --config "${CONFIG_PATH}" \
      "${args[@]}"
}

stop_runtime() {
  a1_tmux_stop "${SESSION}"
  a1_tmux_stop "${MODEL_SESSION}"
  "${BASE_RUNTIME}" stop
  a1_success "LingBot A1 bridge and policy server stopped."
}

status() {
  local rc=0
  "${BASE_RUNTIME}" status || rc=$?
  echo
  a1_info "LingBot tmux sessions"
  a1_tmux_status "${SESSION}" || rc=$?
  a1_tmux_status "${MODEL_SESSION}" || rc=$?
  return "${rc}"
}

case "${1:-help}" in
  start)
    start_pipeline
    echo
    a1_success "LingBot runtime started."
    a1_info "Attach with: tmux attach -t ${SESSION}"
    ;;
  services)
    start_services
    ;;
  server)
    start_model_server
    ;;
  server-stop)
    a1_tmux_stop "${MODEL_SESSION}"
    ;;
  server-logs)
    a1_tmux_capture "${MODEL_SESSION}" 160
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
    a1_usage "$0 [--config PATH] <start|server|server-stop|server-logs|services|tmux|stop|doctor|status|logs>"
    cat <<EOF
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
    if [[ "${1:-help}" != "help" && "${1:-}" != "-h" && "${1:-}" != "--help" ]]; then
      a1_fail "Unknown LingBot command: ${1:-}"
      exit 2
    fi
    ;;
esac
