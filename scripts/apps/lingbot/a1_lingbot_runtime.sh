#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${ROOT}/scripts/runtime/a1_config.sh"
BASE_RUNTIME="${ROOT}/scripts/runtime/a1_joint_runtime.sh"
CONFIG_PATH=""
source "${ROOT}/scripts/runtime/a1_processes.sh"

if [[ "${1:-}" == "--config" ]]; then
  if [[ -z "${2:-}" ]]; then
    a1_fail "--config requires a path."
    a1_usage "$0 --config <path> <setup|verify|run|server|smoke|server-stop|server-logs|services|stop|doctor|status|logs>" >&2
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
SETUP_SCRIPT="${ROOT}/scripts/apps/lingbot/setup_lingbot_inference.py"
SMOKE_SCRIPT="${ROOT}/scripts/apps/lingbot/smoke_lingbot_inference.py"
VERIFY_SCRIPT="${ROOT}/scripts/apps/lingbot/verify_lingbot_inference.py"
PROBE_SCRIPT="${ROOT}/scripts/apps/lingbot/probe_lingbot_server.py"
BRIDGE_ENVIRONMENT_CHECKED=false
SELECTED_TASK_ID=""

config_args=(--repo-root "${ROOT}" --shell)
if [[ -n "${CONFIG_PATH}" ]]; then
  config_args+=("${CONFIG_PATH}")
fi
a1_load_shell_config env \
  PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" -m galaxea_a1_runtime.apps.lingbot.config \
    "${config_args[@]}"
export A1_SYSTEM_CONFIG_PATH="${SYSTEM_CONFIG_PATH}"
MODEL_PROCESS_NAME="lingbot-policy-server"
MODEL_LOG="${MODEL_SAVE_ROOT}/policy_server.log"
PIPELINE_CLEANED_UP=false

check_lingbot_server() {
  if ! PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" \
    "${PROBE_SCRIPT}" --repo-root "${ROOT}" --config "${CONFIG_PATH}"; then
    a1_fail "LingBot server contract check failed on ${LINGBOT_HOST}:${LINGBOT_PORT}."
    return 2
  fi
}

check_lingbot_app() {
  if [[ "${WRIST_BACKEND}" == "v4l2" && "${WRIST_CAMERA}" != "auto" && ! -e "${WRIST_CAMERA}" ]]; then
    a1_fail "Wrist camera not found: ${WRIST_CAMERA}"
    return 2
  fi
  check_lingbot_server
}

check_bridge_environment() {
  if [[ "${BRIDGE_ENVIRONMENT_CHECKED}" == "true" ]]; then
    return
  fi
  if ! PYTHONPATH="${ROOT}/third_party/A1_SDK/install/lib/python3/dist-packages:${ROOT}/.cache/ros1_python_overlay:${PYTHONPATH:-}" \
    "${PYTHON_BIN}" "${BRIDGE_SCRIPT}" --help >/dev/null; then
    a1_fail "LingBot bridge Python environment is incomplete. Run: just setup"
    return 2
  fi
  BRIDGE_ENVIRONMENT_CHECKED=true
}

prepare_model_root() {
  if [[ "${DEPLOYMENT_READY}" != "1" ]]; then
    a1_fail "LingBot deployment_ready=false; update the new checkpoint, prompt, and q01/q99 first."
    return 2
  fi
  verify_inference
  for required in \
    "${MODEL_PYTHON}" \
    "${MODEL_ROOT}/vae/diffusion_pytorch_model.safetensors" \
    "${MODEL_ROOT}/text_encoder/model.safetensors.index.json" \
    "${MODEL_ROOT}/tokenizer/tokenizer.json" \
    "${MODEL_ROOT}/transformer/config.json" \
    "${MODEL_ROOT}/transformer/diffusion_pytorch_model.safetensors"; do
    if [[ ! -e "${required}" ]]; then
      a1_fail "Missing LingBot model prerequisite: ${required}"
      return 2
    fi
  done
  mkdir -p "${MODEL_SAVE_ROOT}"
}

tail_model_log() {
  if [[ -f "${MODEL_LOG}" ]]; then
    a1_info "LingBot policy server log: ${MODEL_LOG}"
    tail -n "${1:-80}" "${MODEL_LOG}" || true
  fi
}

wait_for_model_health() {
  local timeout_s="${MODEL_STARTUP_TIMEOUT%.*}"
  local deadline=$((SECONDS + timeout_s))
  while (( SECONDS < deadline )); do
    if curl -fsS --max-time 1 \
      "http://${LINGBOT_HOST}:${LINGBOT_PORT}/healthz" >/dev/null 2>&1; then
      return 0
    fi
    if ! a1_process_is_running "${MODEL_PROCESS_NAME}"; then
      tail_model_log 120 >&2
      a1_fail "LingBot policy server exited during startup."
      return 2
    fi
    sleep 1
  done
  tail_model_log 120 >&2
  a1_fail "LingBot policy server did not become healthy within ${MODEL_STARTUP_TIMEOUT}s."
  return 2
}

start_model_server() {
  prepare_model_root
  if ss -H -ltn "sport = :${LINGBOT_PORT}" | grep -q .; then
    if a1_process_is_running "${MODEL_PROCESS_NAME}"; then
      check_lingbot_server
      a1_info "LingBot policy server is already running (log: ${MODEL_LOG})."
      return
    fi
    a1_fail "${LINGBOT_HOST}:${LINGBOT_PORT} is occupied by an unmanaged process."
    return 2
  fi
  if a1_process_is_running "${MODEL_PROCESS_NAME}"; then
    a1_warn "Stopping a marked LingBot server that is no longer listening."
    a1_process_stop "${MODEL_PROCESS_NAME}" "${MODEL_SHUTDOWN_TIMEOUT}"
  fi

  local server_command=(
    env
    "PYTHONPATH=${MODEL_CHECKOUT}:${ROOT}:${PYTHONPATH:-}"
    TOKENIZERS_PARALLELISM=false
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    "${MODEL_PYTHON}" -m torch.distributed.run
    --nproc_per_node="${MODEL_WORLD_SIZE}"
    --local-ranks-filter=0
    --master_port "${MODEL_MASTER_PORT}"
    --tee 3
    "${ROOT}/scripts/apps/lingbot/lingbot_va_policy_server.py"
    --repo-root "${ROOT}"
    --config "${CONFIG_PATH}"
  )
  a1_process_start \
    "${MODEL_PROCESS_NAME}" "${MODEL_CHECKOUT}" "${MODEL_LOG}" \
    "${server_command[@]}"
  if ! wait_for_model_health; then
    a1_process_stop "${MODEL_PROCESS_NAME}" "${MODEL_SHUTDOWN_TIMEOUT}" || true
    return 2
  fi
  if ! check_lingbot_server; then
    a1_process_stop "${MODEL_PROCESS_NAME}" "${MODEL_SHUTDOWN_TIMEOUT}" || true
    return 2
  fi
  a1_success "LingBot policy server is listening on ${LINGBOT_HOST}:${LINGBOT_PORT}."
  a1_info "Policy server output: ${MODEL_LOG}"
}

setup_inference() {
  PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" \
    "${SETUP_SCRIPT}" --repo-root "${ROOT}" --config "${CONFIG_PATH}"
}

verify_inference() {
  PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" \
    "${VERIFY_SCRIPT}" --repo-root "${ROOT}" --config "${CONFIG_PATH}"
}

smoke_inference() {
  start_model_server
  PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" \
    "${SMOKE_SCRIPT}" --repo-root "${ROOT}" --config "${CONFIG_PATH}"
}

start_services() {
  a1_info "Config: ${CONFIG_PATH}"
  "${BASE_RUNTIME}" services
}

select_task() {
  if [[ -n "${SELECTED_TASK_ID}" ]]; then
    return
  fi
  local selected
  if ! selected="$(
    PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" \
      -m galaxea_a1_runtime.apps.task_selection \
      --catalog "${TASK_CATALOG_PATH}"
  )"; then
    a1_fail "LingBot task selection cancelled; no model or hardware was started."
    return 2
  fi
  SELECTED_TASK_ID="${selected}"
  a1_info "Selected LingBot task: ${SELECTED_TASK_ID}"
}

run_bridge_foreground() {
  local bridge_command=(
    "${PYTHON_BIN}" "${BRIDGE_SCRIPT}"
    --config "${CONFIG_PATH}"
    --task-id "${SELECTED_TASK_ID}"
  )
  a1_success "LingBot runtime started in the current terminal."
  a1_info "AgentView dashboard: http://${WEB_PREVIEW_BIND}:${WEB_PREVIEW_PORT}"
  local rc=0
  PYTHONPATH="${ROOT}/third_party/A1_SDK/install/lib/python3/dist-packages:${ROOT}/.cache/ros1_python_overlay:${PYTHONPATH:-}" \
    "${bridge_command[@]}" || rc=$?
  echo "BRIDGE_EXIT=${rc}"
  return "${rc}"
}

cleanup_pipeline() {
  if [[ "${PIPELINE_CLEANED_UP}" == "true" ]]; then
    return
  fi
  PIPELINE_CLEANED_UP=true
  a1_cleanup "LingBot foreground run ended; locking and stopping the A1 runtime and policy server."
  if ! "${BASE_RUNTIME}" stop >/dev/null 2>&1; then
    a1_fail "Cleanup failed to stop the A1 runtime: ${BASE_RUNTIME}"
  fi
  if ! a1_process_stop "${MODEL_PROCESS_NAME}" "${MODEL_SHUTDOWN_TIMEOUT}"; then
    a1_fail "Cleanup failed to stop the marked LingBot policy server."
  fi
}

run_pipeline() {
  select_task
  PIPELINE_CLEANED_UP=false
  trap cleanup_pipeline EXIT
  trap 'exit 129' HUP
  trap 'exit 130' INT
  trap 'exit 143' TERM
  check_bridge_environment
  start_model_server
  check_lingbot_app
  start_services
  run_bridge_foreground
}

doctor() {
  local args=("$@")
  "${BASE_RUNTIME}" doctor "${args[@]}"
  PYTHONPATH="${ROOT}/third_party/A1_SDK/install/lib/python3/dist-packages:${ROOT}/.cache/ros1_python_overlay:${PYTHONPATH:-}" \
    "${PYTHON_BIN}" "${ROOT}/scripts/apps/lingbot/a1_lingbot_doctor.py" \
      --config "${CONFIG_PATH}" \
      "${args[@]}"
}

stop_runtime() {
  local rc=0
  "${BASE_RUNTIME}" stop || rc=$?
  a1_process_stop "${MODEL_PROCESS_NAME}" "${MODEL_SHUTDOWN_TIMEOUT}" || rc=$?
  if (( rc == 0 )); then
    a1_success "LingBot A1 runtime and marked policy server stopped."
  fi
  return "${rc}"
}

status() {
  local rc=0
  "${BASE_RUNTIME}" status || rc=$?
  echo
  a1_info "LingBot host process"
  a1_process_status "${MODEL_PROCESS_NAME}" || rc=$?
  return "${rc}"
}

case "${1:-help}" in
  setup)
    setup_inference
    ;;
  verify)
    verify_inference
    ;;
  run)
    run_pipeline
    ;;
  services)
    start_services
    ;;
  server)
    start_model_server
    ;;
  smoke)
    smoke_inference
    ;;
  server-stop)
    a1_process_stop "${MODEL_PROCESS_NAME}" "${MODEL_SHUTDOWN_TIMEOUT}"
    ;;
  server-logs)
    tail_model_log 160
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
    tail_model_log 160
    ;;
  *)
    a1_usage "$0 [--config PATH] <setup|verify|run|server|smoke|server-stop|server-logs|services|stop|doctor|status|logs>"
    cat <<EOF
  setup     Clone, pin, install, download, and verify LingBot
  verify    Hash-check registered LingBot inputs without opening hardware
  run       Run the complete deployment in the current terminal
  server    Start only the marked background policy server (offline workflows)
  smoke     Start the server and run one synthetic inference (no ROS/cameras)
  server-stop  Stop only the managed policy server
  server-logs  Show recent policy server output
  services  Start only the decoupled A1 base runtime
  stop      Stop the A1 runtime and marked LingBot policy server
  doctor    Run base runtime checks plus LingBot app checks
  status    Base runtime and marked policy-server status
  logs      Tail base runtime and policy-server logs

Config:
  ${CONFIG_PATH}
EOF
    if [[ "${1:-help}" != "help" && "${1:-}" != "-h" && "${1:-}" != "--help" ]]; then
      a1_fail "Unknown LingBot command: ${1:-}"
      exit 2
    fi
    ;;
esac
