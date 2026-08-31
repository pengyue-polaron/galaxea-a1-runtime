#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/scripts/runtime/a1_config.sh"
SYSTEM_CONFIG_PATH="${A1_SYSTEM_CONFIG_PATH:-}"
PYTHON_BIN="${ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi
if [[ "${1:-help}" != "stop" && "${1:-help}" != "logs" ]]; then
  config_args=(--repo-root "${ROOT}" --shell)
  if [[ -n "${SYSTEM_CONFIG_PATH}" ]]; then
    config_args+=("${SYSTEM_CONFIG_PATH}")
  fi
  a1_load_shell_config env \
    PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" \
      -m galaxea_a1_runtime.configuration.system "${config_args[@]}"
fi
source "${ROOT}/scripts/runtime/a1_services.sh"
ROSCORE_CONTAINER="${A1_OBSERVABILITY_ROSCORE_CONTAINER}"
OBSERVABILITY_CONTAINER="${A1_OBSERVABILITY_TELEMETRY_CONTAINER}"
FOXGLOVE_CONTAINER="${A1_OBSERVABILITY_FOXGLOVE_CONTAINER}"

stop_observability() {
  a1_remove_runtime_containers \
    "${FOXGLOVE_CONTAINER}" \
    "${OBSERVABILITY_CONTAINER}"
  a1_stop_observability_roscore_if_unused
  a1_success "Standalone read-only observability stopped."
}

start_observability() {
  local startup_complete=0
  cleanup_failed_start() {
    if [[ "${startup_complete}" != "1" ]]; then
      a1_cleanup "Observability startup failed; stopping partial resources."
      stop_observability >/dev/null
    fi
  }
  trap cleanup_failed_start ERR

  a1_info "Config: ${SYSTEM_CONFIG_PATH}"
  a1_preflight_container_host
  if [[ "${OBSERVABILITY_ENABLED}" != "true" ]]; then
    a1_fail "Read-only observability is disabled by System config."
    return 2
  fi
  if a1_observability_stack_is_ready \
    "${OBSERVABILITY_CONTAINER}" "${FOXGLOVE_CONTAINER}"; then
    startup_complete=1
    trap - ERR
    a1_success "Persistent Foxglove is already ready at ws://127.0.0.1:${FOXGLOVE_PORT}."
    return 0
  fi
  a1_step "1/2 Ensuring ROS master without an A1 driver"
  a1_ensure_roscore "${ROSCORE_CONTAINER}"
  a1_step "2/2 Starting read-only telemetry and Foxglove Bridge"
  a1_start_observability "${OBSERVABILITY_CONTAINER}" "${FOXGLOVE_CONTAINER}"
  startup_complete=1
  trap - ERR
  a1_success "Foxglove is ready at ws://127.0.0.1:${FOXGLOVE_PORT}."
}

status_observability() {
  a1_info "Standalone observability containers"
  docker ps -a --format '{{.Names}}\t{{.Status}}' |
    grep -E "^${A1_OBSERVABILITY_PREFIX}-" ||
    a1_info "No ${A1_OBSERVABILITY_PREFIX}-* containers."
  if a1_observability_stack_is_ready \
    "${OBSERVABILITY_CONTAINER}" "${FOXGLOVE_CONTAINER}"; then
    a1_success "Persistent Foxglove stack is healthy at ws://127.0.0.1:${FOXGLOVE_PORT}."
    return 0
  fi
  if a1_foxglove_port_is_listening; then
    a1_fail "Port ${FOXGLOVE_PORT} is listening without the complete marked Foxglove stack."
  else
    a1_info "Foxglove endpoint is not listening on port ${FOXGLOVE_PORT}."
  fi
  return 1
}

logs_observability() {
  for name in "${OBSERVABILITY_CONTAINER}" "${FOXGLOVE_CONTAINER}" "${ROSCORE_CONTAINER}"; do
    a1_info "Logs: ${name}"
    docker logs --tail "${A1_LOG_TAIL:-120}" "${name}" 2>&1 || true
  done
}

case "${1:-help}" in
  start)
    start_observability
    ;;
  restart)
    stop_observability
    start_observability
    ;;
  stop)
    stop_observability
    ;;
  status)
    status_observability
    ;;
  logs)
    logs_observability
    ;;
  help|-h|--help)
    a1_usage "$0 <start|restart|stop|status|logs>"
    cat <<EOF
  start   Start ROS master, telemetry adapter, and read-only Foxglove Bridge
  restart Reload the tracked observability configuration
  stop    Stop only the standalone observability containers
  status  Show standalone containers and endpoint state
  logs    Tail standalone observability logs

This entrypoint never starts the A1 driver, tracker, relay, or a command publisher.
The public camera lifecycle starts this shared stack automatically.
EOF
    ;;
  *)
    a1_fail "Unknown observability command: ${1:-}"
    a1_usage "$0 <start|restart|stop|status|logs>" >&2
    exit 2
    ;;
esac
