#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${ROOT}/scripts/runtime/a1_config.sh"
BASE_RUNTIME="${ROOT}/scripts/runtime/a1_joint_runtime.sh"
CAMERA_RUNTIME="${ROOT}/scripts/apps/cameras/a1_camera_web_runtime.sh"
CONFIG_PATH=""
MODEL_SELECTOR=""
TASK_SELECTOR=""
SCENE_NOTE_INPUT=""
source "${ROOT}/scripts/runtime/a1_processes.sh"
A1_REPO_PYTHONPATH="${ROOT}"

runtime_args=()
while (( $# > 0 )); do
  case "$1" in
    --config)
      if [[ -n "${CONFIG_PATH}" ]]; then
        a1_fail "--config may be provided only once."
        exit 2
      fi
      if [[ -z "${2:-}" ]]; then
        a1_fail "--config requires a path."
        exit 2
      fi
      CONFIG_PATH="$2"
      shift 2
      ;;
    --model)
      if [[ -n "${MODEL_SELECTOR}" ]]; then
        a1_fail "--model may be provided only once."
        exit 2
      fi
      if [[ -z "${2:-}" ]]; then
        a1_fail "--model requires a registered model id or descriptor name."
        exit 2
      fi
      MODEL_SELECTOR="$2"
      shift 2
      ;;
    --task)
      if [[ -n "${TASK_SELECTOR}" || -z "${2:-}" ]]; then
        a1_fail "--task requires one tracked task id."
        exit 2
      fi
      TASK_SELECTOR="$2"
      shift 2
      ;;
    --scene-note)
      if [[ -n "${SCENE_NOTE_INPUT}" || -z "${2:-}" ]]; then
        a1_fail "--scene-note requires one non-empty scene description."
        exit 2
      fi
      SCENE_NOTE_INPUT="$2"
      shift 2
      ;;
    *)
      runtime_args+=("$1")
      shift
      ;;
  esac
done
set -- "${runtime_args[@]}"

case "${1:-}" in
  run)
    ;;
  batch)
    if [[ -n "${TASK_SELECTOR}" ]]; then
      a1_fail "--task is not valid with batch; tasks come from the selected plan."
      exit 2
    fi
    ;;
  *)
    if [[ -n "${TASK_SELECTOR}" || -n "${SCENE_NOTE_INPUT}" ]]; then
      a1_fail "--task and --scene-note are valid only with run or batch."
      exit 2
    fi
    ;;
esac

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
SCENE_NOTE=""
RESET_BEFORE_RUN_PATH=""
BATCH_ID=""
BATCH_RESET_POSE=""
BATCH_RETRIES_PER_PROMPT=0
BATCH_ATTEMPTS_PER_PROMPT=1
BATCH_TOTAL_ATTEMPTS=0
BATCH_TASK_IDS_CSV=""
BATCH_TASK_POSITION=""
BATCH_TASK_COUNT=""
BATCH_ATTEMPT=""
BATCH_SEQUENCE=""
BATCH_COMPLETED_SEQUENCES_CSV=""
BATCH_COMPLETED_COUNT=0
BATCH_PENDING_COUNT=0

config_args=(--repo-root "${ROOT}" --shell)
if [[ -n "${MODEL_SELECTOR}" ]]; then
  config_args+=(--model "${MODEL_SELECTOR}")
fi
if [[ -n "${CONFIG_PATH}" ]]; then
  config_args+=("${CONFIG_PATH}")
fi
a1_load_shell_config env \
  PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" -m galaxea_a1_runtime.apps.lingbot.config \
    "${config_args[@]}"
export A1_SYSTEM_CONFIG_PATH="${SYSTEM_CONFIG_PATH}"
if [[ -n "${TASK_SELECTOR}" ]]; then
  SELECTED_TASK_ID="$(
    PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" \
      -m galaxea_a1_runtime.apps.task_selection \
      --catalog "${TASK_CATALOG_PATH}" \
      --task-id "${TASK_SELECTOR}"
  )"
fi
if [[ -n "${SCENE_NOTE_INPUT}" ]]; then
  SCENE_NOTE="$(
    PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" \
      -m galaxea_a1_runtime.apps.lingbot.operator_input \
      --value "${SCENE_NOTE_INPUT}"
  )"
fi
MODEL_PROCESS_NAME="lingbot-policy-server"
MODEL_LOG="${MODEL_SAVE_ROOT}/policy_server.log"
PIPELINE_CLEANED_UP=false
RUN_ID=""
RUN_LOG_STAGING_DIR=""
RUN_RUNTIME_RAW_LOG=""
RUN_POLICY_LOG=""
RUN_FRONT_VIDEO_FILENAME=""
RUN_WRIST_VIDEO_FILENAME=""
RUN_ARTIFACTS_PREPARED=false
RUN_ARTIFACTS_FINALIZED=false
RUN_EXIT_CODE=2
RUN_STATUS=""
RUN_EVALUATION_DECISION=""

check_lingbot_server() {
  if ! PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" \
    "${PROBE_SCRIPT}" --repo-root "${ROOT}" --config "${CONFIG_PATH}" \
    --model "${MODEL_ID}"; then
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
  if ! command -v script >/dev/null 2>&1; then
    a1_fail "util-linux script is required for per-run foreground logging."
    return 2
  fi
  if ! command -v tee >/dev/null 2>&1; then
    a1_fail "tee is required for per-run foreground logging."
    return 2
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
  local force_restart="${1:-false}"
  prepare_model_root
  if ss -H -ltn "sport = :${LINGBOT_PORT}" | grep -q .; then
    if a1_process_is_running "${MODEL_PROCESS_NAME}"; then
      if [[ "${force_restart}" == "true" ]]; then
        a1_info "Restarting the marked policy server for an independent live run."
        a1_process_stop "${MODEL_PROCESS_NAME}" "${MODEL_SHUTDOWN_TIMEOUT}"
      else
        check_lingbot_server
        a1_info "LingBot policy server is already running (log: ${MODEL_LOG})."
        return
      fi
    else
      a1_fail "${LINGBOT_HOST}:${LINGBOT_PORT} is occupied by an unmanaged process."
      return 2
    fi
  fi
  if a1_process_is_running "${MODEL_PROCESS_NAME}"; then
    a1_warn "Stopping a marked LingBot server that is no longer listening."
    a1_process_stop "${MODEL_PROCESS_NAME}" "${MODEL_SHUTDOWN_TIMEOUT}"
  fi

  local server_command=(
    env
    "PYTHONPATH=${MODEL_CHECKOUT}:${A1_REPO_PYTHONPATH}:${PYTHONPATH:-}"
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
    --model "${MODEL_ID}"
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
    "${SETUP_SCRIPT}" --repo-root "${ROOT}" --config "${CONFIG_PATH}" \
    --model "${MODEL_ID}"
}

verify_inference() {
  PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" \
    "${VERIFY_SCRIPT}" --repo-root "${ROOT}" --config "${CONFIG_PATH}" \
    --model "${MODEL_ID}"
}

smoke_inference() {
  start_model_server
  PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" \
    "${SMOKE_SCRIPT}" --repo-root "${ROOT}" --config "${CONFIG_PATH}" \
    --model "${MODEL_ID}"
}

start_services() {
  a1_info "Config: ${CONFIG_PATH}"
  "${BASE_RUNTIME}" services
}

ensure_camera_monitor() {
  "${CAMERA_RUNTIME}" --config "${SYSTEM_CONFIG_PATH}"
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
    a1_fail "LingBot task selection cancelled; no model or motion runtime was started; the camera monitor remains available."
    return 2
  fi
  SELECTED_TASK_ID="${selected}"
  a1_info "Selected LingBot task: ${SELECTED_TASK_ID}"
}

read_scene_note() {
  if [[ -n "${SCENE_NOTE}" ]]; then
    return
  fi
  local note
  if ! note="$(
    PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" \
      -m galaxea_a1_runtime.apps.lingbot.operator_input
  )"; then
    a1_fail "Scene note cancelled; no model or motion runtime was started; the camera monitor remains available."
    return 2
  fi
  SCENE_NOTE="${note}"
}

run_bridge_foreground() {
  local run_id="$1"
  local bridge_command=(
    "${PYTHON_BIN}" "${BRIDGE_SCRIPT}"
    --config "${CONFIG_PATH}"
    --model "${MODEL_ID}"
    --task-id "${SELECTED_TASK_ID}"
    --run-id "${run_id}"
    --front-video-filename "${RUN_FRONT_VIDEO_FILENAME}"
    --wrist-video-filename "${RUN_WRIST_VIDEO_FILENAME}"
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
  ensure_camera_monitor >/dev/null 2>&1 || \
    a1_fail "Persistent Camera Bridge became unavailable during cleanup."
}

run_pipeline_foreground() {
  local run_id="$1"
  local reset_pose="$2"
  PIPELINE_CLEANED_UP=false
  trap cleanup_pipeline EXIT
  trap 'exit 129' HUP
  trap 'exit 130' INT
  trap 'exit 143' TERM
  check_bridge_environment
  ensure_camera_monitor
  a1_info "LingBot run: ${run_id}"
  a1_info "Selected LingBot task: ${SELECTED_TASK_ID}"
  start_model_server true
  start_services
  if [[ -n "${reset_pose}" ]]; then
    a1_step "Resetting A1 to the tracked batch start pose before inference."
    PYTHONPATH="${ROOT}/third_party/A1_SDK/install/lib/python3/dist-packages:${ROOT}/.cache/ros1_python_overlay:${PYTHONPATH:-}" \
      "${PYTHON_BIN}" "${ROOT}/scripts/apps/reset/a1_reset.py" \
      --system-config "${SYSTEM_CONFIG_PATH}" \
      --pose "${reset_pose}"
  fi
  check_lingbot_app
  a1_step "Reading uncompressed frames from the persistent Camera Bridge."
  run_bridge_foreground "${run_id}"
}

prepare_run_artifacts() {
  local batch_args=()
  if [[ -n "${BATCH_ID}" ]]; then
    batch_args=(
      --batch-id "${BATCH_ID}"
      --task-position "${BATCH_TASK_POSITION}"
      --task-count "${BATCH_TASK_COUNT}"
      --attempt "${BATCH_ATTEMPT}"
      --attempt-count "${BATCH_ATTEMPTS_PER_PROMPT}"
      --sequence "${BATCH_SEQUENCE}"
      --total "${BATCH_TOTAL_ATTEMPTS}"
    )
  fi
  a1_load_shell_config env \
    PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" \
    -m galaxea_a1_runtime.apps.lingbot.run_artifacts prepare \
    --repo-root "${ROOT}" \
    --config "${CONFIG_PATH}" \
    --model "${MODEL_ID}" \
    --task-id "${SELECTED_TASK_ID}" \
    --scene-note "${SCENE_NOTE}" \
    "${batch_args[@]}" \
    --shell
}

finalize_run_artifacts() {
  local exit_code="$1"
  local final_dir
  if ! final_dir="$(
    PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" \
      -m galaxea_a1_runtime.apps.lingbot.run_artifacts finalize \
      --output-root "${RECORDING_OUTPUT_ROOT}" \
      --run-id "${RUN_ID}" \
      --exit-code "${exit_code}"
  )"; then
    a1_fail "Failed to finalize LingBot run artifacts from ${RUN_LOG_STAGING_DIR}."
    return 2
  fi
  RUN_ARTIFACTS_FINALIZED=true
  a1_success "LingBot run artifacts saved: ${final_dir}"
}

load_run_result() {
  a1_load_shell_config env \
    PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" \
    -m galaxea_a1_runtime.apps.lingbot.run_artifacts inspect \
    --output-root "${RECORDING_OUTPUT_ROOT}" \
    --run-id "${RUN_ID}" \
    --shell
}

record_evaluation_decision() {
  local decision="$1"
  PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" \
    -m galaxea_a1_runtime.apps.lingbot.run_artifacts decide \
    --output-root "${RECORDING_OUTPUT_ROOT}" \
    --run-id "${RUN_ID}" \
    --decision "${decision}" >/dev/null
  RUN_EVALUATION_DECISION="${decision}"
}

finalize_run_artifacts_on_exit() {
  local shell_status=$?
  if [[ "${RUN_ARTIFACTS_PREPARED}" != "true" || "${RUN_ARTIFACTS_FINALIZED}" == "true" ]]; then
    return
  fi
  local exit_code="${RUN_EXIT_CODE}"
  if [[ ! "${exit_code}" =~ ^[0-9]+$ ]] || (( exit_code > 255 )); then
    exit_code="${shell_status}"
  fi
  finalize_run_artifacts "${exit_code}" || true
}

run_pipeline() {
  ensure_camera_monitor || return $?
  check_bridge_environment || return $?
  a1_info "LingBot model: ${MODEL_ID}"
  select_task || return $?
  read_scene_note || return $?
  prepare_run_artifacts || return $?
  RUN_ARTIFACTS_PREPARED=true
  RUN_ARTIFACTS_FINALIZED=false
  RUN_EXIT_CODE=2
  RUN_STATUS=""
  RUN_EVALUATION_DECISION=""
  trap finalize_run_artifacts_on_exit EXIT
  trap 'RUN_EXIT_CODE=129; exit 129' HUP
  trap 'RUN_EXIT_CODE=130; exit 130' INT
  trap 'RUN_EXIT_CODE=143; exit 143' TERM

  local internal_command=(
    "$0" --config "${CONFIG_PATH}" --model "${MODEL_ID}" __run
    "${SELECTED_TASK_ID}" "${RUN_ID}" "${RUN_POLICY_LOG}"
    "${RUN_FRONT_VIDEO_FILENAME}" "${RUN_WRIST_VIDEO_FILENAME}"
    "${RESET_BEFORE_RUN_PATH}"
  )
  local internal_command_q
  printf -v internal_command_q "%q " "${internal_command[@]}"
  local statuses rc
  set +e
  SHELL=/bin/bash script --quiet --flush --return \
    --command "${internal_command_q}" /dev/null |
    tee --ignore-interrupts "${RUN_RUNTIME_RAW_LOG}"
  statuses=("${PIPESTATUS[@]}")
  set -e
  rc="${statuses[0]}"
  if (( statuses[1] != 0 )); then
    a1_fail "LingBot runtime log writer exited with status ${statuses[1]}."
    if (( rc == 0 )); then
      rc="${statuses[1]}"
    fi
  fi
  RUN_EXIT_CODE="${rc}"
  if ! finalize_run_artifacts "${rc}"; then
    if (( rc == 0 )); then
      rc=2
    fi
    RUN_EXIT_CODE="${rc}"
  elif ! load_run_result; then
    a1_fail "Failed to inspect finalized LingBot run ${RUN_ID}."
    rc=2
    RUN_EXIT_CODE="${rc}"
  fi
  if [[ "${RUN_ARTIFACTS_FINALIZED}" == "true" ]]; then
    trap - EXIT
  fi
  trap - HUP INT TERM
  return "${rc}"
}

load_batch_config() {
  local batch_config="$1"
  a1_load_shell_config env \
    PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" \
    -m galaxea_a1_runtime.apps.lingbot.batch_config \
    --repo-root "${ROOT}" --model "${MODEL_ID}" --shell "${batch_config}"
  if [[ "${BATCH_DEPLOYMENT_CONFIG}" != "${CONFIG_PATH}" ]]; then
    a1_fail "Batch deployment ${BATCH_DEPLOYMENT_CONFIG} does not match ${CONFIG_PATH}."
    return 2
  fi
}

load_batch_progress() {
  local batch_config="$1"
  a1_load_shell_config env \
    PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" \
    -m galaxea_a1_runtime.apps.lingbot.batch_progress \
    --repo-root "${ROOT}" \
    --model "${MODEL_ID}" \
    --scene-note "${SCENE_NOTE}" \
    --shell "${batch_config}"
}

batch_sequence_is_complete() {
  local sequence="$1"
  [[ ",${BATCH_COMPLETED_SEQUENCES_CSV}," == *",${sequence},"* ]]
}

confirm_batch_attempt() {
  local label="$1"
  local answer
  while true; do
    a1_step "${label}"
    if [[ "${OPERATOR_PANEL_PROTOCOL:-}" == "1" ]]; then
      printf '%s\n' '@@OPERATOR_PANEL {"input":["enter","quit"]}'
    fi
    if ! IFS= read -r -p "Enter=reset A1 and run, q=stop batch > " answer; then
      a1_info "Batch input closed; stopping before the next reset."
      return 1
    fi
    if [[ -z "${answer}" ]]; then
      return 0
    fi
    if [[ "${answer,,}" =~ ^(q|quit|exit)$ ]]; then
      return 1
    fi
    a1_warn "Unknown input; press Enter to run or q to stop."
  done
}

decide_safety_stopped_attempt() {
  local answer
  a1_warn "This evaluation ended on a rejected model target; the target was not published."
  while true; do
    if [[ "${OPERATOR_PANEL_PROTOCOL:-}" == "1" ]]; then
      printf '%s\n' '@@OPERATOR_PANEL {"input":["enter","discard","quit"]}'
    fi
    if ! IFS= read -r -p "Enter=count as evaluated, d=discard and retry this slot, q=stop batch > " answer; then
      a1_info "Decision input closed; leaving this evaluation pending."
      return 20
    fi
    case "${answer,,}" in
      ""|k|keep|count)
        if ! record_evaluation_decision counted; then
          a1_fail "Failed to record the counted evaluation decision."
          return 2
        fi
        a1_success "Evaluation counted; the next slot will wait for Enter/reset."
        return 0
        ;;
      d|discard|retry)
        if ! record_evaluation_decision discarded; then
          a1_fail "Failed to record the discarded evaluation decision."
          return 2
        fi
        a1_info "Evaluation discarded; this same slot will wait for Enter/reset again."
        return 10
        ;;
      q|quit|exit)
        a1_info "Batch stopped; this evaluation remains pending for resume."
        return 20
        ;;
      *)
        a1_warn "Unknown input; press Enter to count, d to discard/retry, or q to stop."
        ;;
    esac
  done
}

run_batch() {
  local batch_config="${1:-${ROOT}/configs/runs/lingbot/fruit_placement.toml}"
  local resume="${2:-false}"
  ensure_camera_monitor
  check_bridge_environment
  load_batch_config "${batch_config}"
  read_scene_note
  if [[ "${resume}" == "true" ]]; then
    load_batch_progress "${batch_config}"
    a1_info "Resume found ${BATCH_COMPLETED_COUNT}/${BATCH_TOTAL_ATTEMPTS} completed slots for scene note ${SCENE_NOTE}; pending=${BATCH_PENDING_COUNT}."
  fi

  local task_ids=()
  IFS=',' read -r -a task_ids <<<"${BATCH_TASK_IDS_CSV}"
  local task_count="${#task_ids[@]}"
  local sequence=0 task_index attempt task_id rc decision_rc
  a1_info "LingBot model: ${MODEL_ID}"
  a1_info "LingBot batch ${BATCH_ID}: tasks=${task_count} retries_per_prompt=${BATCH_RETRIES_PER_PROMPT} total=${BATCH_TOTAL_ATTEMPTS}"
  a1_info "Scene note: ${SCENE_NOTE}"
  for (( task_index = 0; task_index < task_count; task_index++ )); do
    task_id="${task_ids[task_index]}"
    for (( attempt = 1; attempt <= BATCH_ATTEMPTS_PER_PROMPT; attempt++ )); do
      sequence=$((sequence + 1))
      if [[ "${resume}" == "true" ]] && batch_sequence_is_complete "${sequence}"; then
        a1_success "Skipping completed ${sequence}/${BATCH_TOTAL_ATTEMPTS}: task=${task_id} attempt=${attempt}/${BATCH_ATTEMPTS_PER_PROMPT}"
        continue
      fi
      while true; do
        if ! confirm_batch_attempt \
          "Next ${sequence}/${BATCH_TOTAL_ATTEMPTS}: task=${task_id} attempt=${attempt}/${BATCH_ATTEMPTS_PER_PROMPT}"; then
          a1_success "LingBot batch stopped before attempt ${sequence}."
          return 0
        fi
        SELECTED_TASK_ID="${task_id}"
        RESET_BEFORE_RUN_PATH="${BATCH_RESET_POSE}"
        BATCH_TASK_POSITION=$((task_index + 1))
        BATCH_TASK_COUNT="${task_count}"
        BATCH_ATTEMPT="${attempt}"
        BATCH_SEQUENCE="${sequence}"
        rc=0
        run_pipeline || rc=$?
        if (( rc != 0 )); then
          a1_fail "Batch aborted after attempt ${sequence}/${BATCH_TOTAL_ATTEMPTS} failed with status ${rc}."
          return "${rc}"
        fi
        if [[ "${RUN_STATUS}" != "safety_stopped" ]]; then
          break
        fi
        decision_rc=0
        decide_safety_stopped_attempt || decision_rc=$?
        if (( decision_rc == 0 )); then
          break
        fi
        if (( decision_rc == 10 )); then
          continue
        fi
        if (( decision_rc == 20 )); then
          return 0
        fi
        return "${decision_rc}"
      done
    done
  done
  a1_success "LingBot batch ${BATCH_ID} completed ${BATCH_TOTAL_ATTEMPTS} attempts."
}

stop_runtime() {
  local rc=0
  "${BASE_RUNTIME}" stop || rc=$?
  a1_process_stop "${MODEL_PROCESS_NAME}" "${MODEL_SHUTDOWN_TIMEOUT}" || rc=$?
  if (( rc == 0 )); then
    a1_success "LingBot A1 runtime and marked policy server stopped."
  fi
  ensure_camera_monitor || rc=$?
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
  batch)
    shift
    batch_resume=false
    if [[ "${1:-}" == "--resume" ]]; then
      batch_resume=true
      shift
    fi
    if (( $# > 1 )); then
      a1_fail "batch accepts [--resume] and at most one tracked plan path."
      exit 2
    fi
    run_batch "${1:-}" "${batch_resume}"
    ;;
  __run)
    if (( $# != 7 )); then
      a1_fail "Internal LingBot run expects <task-id> <run-id> <policy-log> <front-video-filename> <wrist-video-filename> <reset-pose>."
      exit 2
    fi
    SELECTED_TASK_ID="$2"
    RUN_ID="$3"
    MODEL_LOG="$4"
    RUN_FRONT_VIDEO_FILENAME="$5"
    RUN_WRIST_VIDEO_FILENAME="$6"
    RESET_BEFORE_RUN_PATH="$7"
    expected_policy_log="${RECORDING_OUTPUT_ROOT}/.${RUN_ID}.logs/policy_server.log"
    if [[ "${MODEL_LOG}" != "${expected_policy_log}" || ! -f "${MODEL_LOG}" ]]; then
      a1_fail "Internal LingBot policy log does not match prepared run ${RUN_ID}."
      exit 2
    fi
    run_pipeline_foreground "${RUN_ID}" "${RESET_BEFORE_RUN_PATH}"
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
    "${BASE_RUNTIME}" doctor "$@"
    ;;
  status)
    status
    ;;
  logs)
    "${BASE_RUNTIME}" logs
    tail_model_log 160
    ;;
  *)
    a1_usage "$0 [--config PATH] [--model REGISTERED_MODEL] [--task TASK_ID] [--scene-note TEXT] <setup|verify|run|batch|server|smoke|server-stop|server-logs|services|stop|doctor|status|logs>"
    cat <<EOF
  setup     Clone, pin, install, download, and verify LingBot
  verify    Hash-check registered LingBot inputs without opening hardware
  run       Run the complete deployment in the current terminal
  batch     Run a tracked plan; add --resume to skip finished matching slots
  server    Start only the marked background policy server (offline workflows)
  smoke     Start the server and run one synthetic inference (no ROS/cameras)
  server-stop  Stop only the managed policy server
  server-logs  Show recent policy server output
  services  Start only the decoupled A1 base runtime
  stop      Stop the A1 runtime and marked LingBot policy server
  doctor    Run the shared layered runtime checks
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
