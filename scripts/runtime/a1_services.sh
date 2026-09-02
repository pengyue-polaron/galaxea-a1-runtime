#!/usr/bin/env bash
# Shared, app-agnostic Docker/ROS service primitives for A1 runtime entrypoints.
# Callers own lifecycle, tracker selection, UI, and failure cleanup policy.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/a1_console.sh"

A1_ROS_PREFIX='source /opt/ros/noetic/setup.bash && source "${A1_SDK_ROOT}/install/setup.bash" && source /opt/foxglove_bridge_ws/install/local_setup.bash && export ROS_PACKAGE_PATH="/opt/foxglove_bridge_ws/install/share:${ROS_PACKAGE_PATH}"'
A1_MANAGED_CONTAINER_LABEL='io.galaxea.a1-runtime.managed=true'
A1_CONTAINER_PYTHONPATH='/workspace:/workspace/external/embodied-ops/src'
A1_OBSERVABILITY_PREFIX="${A1_OBSERVABILITY_PREFIX:-a1-observability}"
A1_OBSERVABILITY_ROSCORE_CONTAINER="${A1_OBSERVABILITY_PREFIX}-roscore"
A1_OBSERVABILITY_TELEMETRY_CONTAINER="${A1_OBSERVABILITY_PREFIX}-telemetry"
A1_OBSERVABILITY_FOXGLOVE_CONTAINER="${A1_OBSERVABILITY_PREFIX}-foxglove"

a1_require_runtime_value() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    a1_fail "Required runtime value ${name} is unset."
    return 2
  fi
}

a1_preflight_container_host() {
  a1_require_runtime_value ROOT
  a1_require_runtime_value IMAGE
  if ! command -v docker >/dev/null 2>&1; then
    a1_fail "Docker CLI is not installed."
    return 2
  fi
  if ! docker info >/dev/null 2>&1; then
    a1_fail "Docker daemon is unavailable."
    return 2
  fi
  if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    a1_fail "Runtime image is missing: ${IMAGE}"
    return 2
  fi
  if [[ ! -d "${ROOT}/third_party/A1_SDK" ]]; then
    a1_fail "Vendored A1 SDK is missing under ${ROOT}/third_party/A1_SDK."
    return 2
  fi
}

a1_preflight_container_runtime() {
  a1_preflight_container_host || return
  a1_require_runtime_value SERIAL
  if [[ ! -c "${SERIAL}" ]]; then
    a1_fail "A1 serial path is not a character device: ${SERIAL}"
    return 2
  fi
}

a1_container_run() {
  if (( $# != 3 )); then
    a1_fail "a1_container_run expects <profile> <name> <command>."
    return 2
  fi
  local profile="$1"
  local name="$2"
  local command="$3"
  a1_require_runtime_value ROOT
  a1_require_runtime_value IMAGE
  local access_args=(--network host -v "${ROOT}:/workspace:ro")
  local environment_args=()
  case "${profile}" in
    core|relay)
      ;;
    driver)
      a1_require_runtime_value SERIAL
      access_args+=(--device "${SERIAL}:${SERIAL}")
      ;;
    tracker)
      access_args=(--network host --ipc host -v "${ROOT}:/workspace:rw")
      ;;
    output-writer)
      mkdir -p "${ROOT}/outputs"
      access_args=(
        --network host
        -v "${ROOT}:/workspace:ro"
        -v "${ROOT}/outputs:/workspace/outputs:rw"
      )
      ;;
    telemetry)
      local host_state_root="${A1_PROCESS_STATE_ROOT:-${XDG_RUNTIME_DIR:-/tmp}/galaxea-a1-runtime-${UID}}"
      mkdir -p "${host_state_root}"
      access_args+=(
        -v "${host_state_root}:/run/galaxea-a1-runtime:ro"
      )
      environment_args+=(
        -e A1_PROCESS_STATE_ROOT=/run/galaxea-a1-runtime
      )
      ;;
    *)
      a1_fail "Unknown A1 container profile: ${profile}"
      return 2
      ;;
  esac
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker run -d \
    --name "${name}" \
    --label "${A1_MANAGED_CONTAINER_LABEL}" \
    "${access_args[@]}" \
    -e A1_SDK_ROOT=/workspace/third_party/A1_SDK \
    -e "PYTHONPATH=${A1_CONTAINER_PYTHONPATH}" \
    "${environment_args[@]}" \
    "${IMAGE}" \
    bash -lc "${command}" \
    >/dev/null
}

a1_remove_runtime_containers() {
  docker rm -f "$@" >/dev/null 2>&1 || true
}

a1_remove_all_managed_containers() {
  if ! command -v docker >/dev/null 2>&1; then
    return 0
  fi
  local exclusions=("$@")
  local container_ids=()
  local removable_ids=()
  local listing
  if ! listing="$(docker ps -aq --filter "label=${A1_MANAGED_CONTAINER_LABEL}" 2>/dev/null)"; then
    a1_fail "Could not list managed A1 runtime containers."
    return 2
  fi
  mapfile -t container_ids <<<"${listing}"
  if (( ${#container_ids[@]} == 1 )) && [[ -z "${container_ids[0]}" ]]; then
    container_ids=()
  fi
  local container_id container_name exclusion excluded
  for container_id in "${container_ids[@]}"; do
    container_name="$(docker inspect --format '{{.Name}}' "${container_id}" 2>/dev/null)" || {
      a1_fail "Could not identify managed container ${container_id}."
      return 2
    }
    container_name="${container_name#/}"
    excluded=false
    for exclusion in "${exclusions[@]}"; do
      if [[ "${container_name}" == "${exclusion}" ]]; then
        excluded=true
        break
      fi
    done
    if [[ "${excluded}" != "true" ]]; then
      removable_ids+=("${container_id}")
    fi
  done
  if (( ${#removable_ids[@]} > 0 )); then
    if ! docker rm -f "${removable_ids[@]}" >/dev/null; then
      a1_fail "Could not remove every managed A1 runtime container."
      return 2
    fi
  fi
  if ! listing="$(docker ps -aq --filter "label=${A1_MANAGED_CONTAINER_LABEL}" 2>/dev/null)"; then
    a1_fail "Could not verify managed A1 runtime container shutdown."
    return 2
  fi
  if [[ -n "${listing}" ]]; then
    mapfile -t container_ids <<<"${listing}"
    for container_id in "${container_ids[@]}"; do
      container_name="$(docker inspect --format '{{.Name}}' "${container_id}" 2>/dev/null)" || return 2
      container_name="${container_name#/}"
      excluded=false
      for exclusion in "${exclusions[@]}"; do
        if [[ "${container_name}" == "${exclusion}" ]]; then
          excluded=true
          break
        fi
      done
      if [[ "${excluded}" != "true" ]]; then
        a1_fail "Managed A1 runtime container remains after shutdown: ${container_name}."
        return 2
      fi
    done
  fi
}

a1_container_is_running() {
  [[ "$(docker inspect --format '{{.State.Running}}' "$1" 2>/dev/null || true)" == "true" ]]
}

a1_foxglove_port_is_listening() {
  a1_require_runtime_value FOXGLOVE_PORT || return
  timeout 1 bash -c "</dev/tcp/127.0.0.1/${FOXGLOVE_PORT}" 2>/dev/null
}

a1_observability_stack_is_ready() {
  local telemetry_container="$1"
  local foxglove_container="$2"
  a1_container_is_running "${telemetry_container}" &&
    a1_container_is_running "${foxglove_container}" &&
    a1_foxglove_port_is_listening
}

a1_stop_observability_roscore_if_unused() {
  if a1_container_is_running "${A1_OBSERVABILITY_TELEMETRY_CONTAINER}" ||
    a1_container_is_running "${A1_OBSERVABILITY_FOXGLOVE_CONTAINER}"; then
    return 0
  fi

  local active_name
  while IFS= read -r active_name; do
    [[ -z "${active_name}" ]] && continue
    if [[ "${active_name}" != "${A1_OBSERVABILITY_ROSCORE_CONTAINER}" ]]; then
      return 0
    fi
  done < <(
    docker ps \
      --filter "label=${A1_MANAGED_CONTAINER_LABEL}" \
      --format '{{.Names}}' 2>/dev/null || true
  )
  a1_remove_runtime_containers "${A1_OBSERVABILITY_ROSCORE_CONTAINER}"
}

a1_cleanup_shared_ros_nodes() {
  local ros_container
  ros_container="$(docker ps --format '{{.Names}}' | grep -E '^galaxea-a1-runtime-a1-noetic-run-' | head -n 1 || true)"
  if [[ -n "${ros_container}" ]]; then
    docker exec "${ros_container}" bash -lc \
      'source /opt/ros/noetic/setup.bash; rosnode cleanup <<< y >/dev/null 2>&1 || true' \
      >/dev/null 2>&1 || true
  fi
}

a1_ensure_roscore() {
  local container="$1"
  if timeout 1 bash -c '</dev/tcp/127.0.0.1/11311' >/dev/null 2>&1; then
    return 0
  fi
  a1_container_run core "${container}" "${A1_ROS_PREFIX} && exec roscore"
  a1_require_runtime_value ROS_MASTER_STARTUP_TIMEOUT_S
  local deadline=$((SECONDS + ${ROS_MASTER_STARTUP_TIMEOUT_S%.*}))
  until timeout 1 bash -c '</dev/tcp/127.0.0.1/11311' >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      a1_fail "ROS master did not start."
      return 2
    fi
    sleep 0.5
  done
}

a1_start_driver() {
  local container="$1"
  a1_require_runtime_value SERIAL
  a1_container_run driver "${container}" \
    "${A1_ROS_PREFIX} && exec roslaunch signal_arm single_arm_node.launch single_arm_serial_port_path:=${SERIAL}"
}

a1_require_running_container() {
  if (( $# != 2 )); then
    a1_fail "a1_require_running_container expects <container> <wait-description>."
    return 2
  fi
  local container="$1"
  local wait_description="$2"
  local state
  if ! state="$(docker inspect --format \
    '{{if .State.Running}}running{{else}}{{.State.Status}} (exit {{.State.ExitCode}}){{end}}' \
    "${container}" 2>/dev/null)"; then
    state="missing"
  fi
  if [[ "${state}" == "running" ]]; then
    return 0
  fi
  a1_fail "Container ${container} is ${state} while waiting for ${wait_description}."
  if [[ "${state}" != "missing" ]]; then
    a1_warn "Last ${A1_LOG_TAIL:-120} log lines from ${container}:"
    docker logs --tail "${A1_LOG_TAIL:-120}" "${container}" >&2 || true
  fi
  return 1
}

a1_wait_valid_joint_feedback() {
  local container="$1"
  local topic="$2"
  a1_require_runtime_value JOINT_FEEDBACK_STARTUP_TIMEOUT_S
  local deadline=$((SECONDS + ${JOINT_FEEDBACK_STARTUP_TIMEOUT_S%.*}))
  while (( SECONDS < deadline )); do
    if docker exec "${container}" bash -lc \
      "${A1_ROS_PREFIX}; timeout 2 rostopic echo -n1 '${topic}' | grep -Eq '^position: \\[[^]]+\\]'" \
      >/dev/null 2>&1; then
      return 0
    fi
    if ! a1_require_running_container "${container}" "non-empty ${topic}"; then
      return 1
    fi
    sleep 1
  done
  a1_fail "No non-empty ${topic} after ${JOINT_FEEDBACK_STARTUP_TIMEOUT_S}s."
  return 1
}

a1_wait_topic() {
  local container="$1"
  local topic="$2"
  a1_require_runtime_value TOPIC_STARTUP_TIMEOUT_S
  local deadline=$((SECONDS + ${TOPIC_STARTUP_TIMEOUT_S%.*}))
  while (( SECONDS < deadline )); do
    if docker exec "${container}" bash -lc \
      "${A1_ROS_PREFIX}; timeout 2 rostopic echo -n1 '${topic}' >/dev/null" \
      >/dev/null 2>&1; then
      return 0
    fi
    if ! a1_require_running_container "${container}" "a message on ${topic}"; then
      return 1
    fi
    sleep 1
  done
  a1_fail "No message on ${topic} after ${TOPIC_STARTUP_TIMEOUT_S}s."
  return 1
}

a1_start_command_relay() {
  local container="$1"
  a1_require_runtime_value SYSTEM_CONFIG_PATH
  a1_require_runtime_value ROOT
  local relative_config="${SYSTEM_CONFIG_PATH#${ROOT}/}"
  if [[ "${relative_config}" == "${SYSTEM_CONFIG_PATH}" ]]; then
    a1_fail "System config must be inside the repository for Docker: ${SYSTEM_CONFIG_PATH}"
    return 2
  fi
  a1_container_run relay "${container}" \
    "${A1_ROS_PREFIX} && exec /workspace/scripts/runtime/safe_arm_command_relay.py \
      --config '/workspace/${relative_config}'"
}

a1_start_observability() {
  if (( $# != 2 )); then
    a1_fail "a1_start_observability expects <telemetry-container> <foxglove-container>."
    return 2
  fi
  local telemetry_container="$1"
  local foxglove_container="$2"
  a1_require_runtime_value OBSERVABILITY_ENABLED
  if [[ "${OBSERVABILITY_ENABLED}" != "true" ]]; then
    return 0
  fi
  for name in \
    SYSTEM_CONFIG_PATH ROOT FOXGLOVE_BIND FOXGLOVE_PORT FOXGLOVE_STARTUP_TIMEOUT_S \
    FOXGLOVE_GRAPH_UPDATE_MS FOXGLOVE_SEND_BUFFER_LIMIT_BYTES \
    FOXGLOVE_TOPIC_WHITELIST_YAML FOXGLOVE_SERVICE_WHITELIST_YAML \
    FOXGLOVE_NO_MATCH_ALLOWLIST_YAML \
    FOXGLOVE_CAPABILITIES_YAML FOXGLOVE_ASSET_URI_ALLOWLIST_YAML \
    OBSERVABILITY_DIAGNOSTICS_TOPIC; do
    a1_require_runtime_value "${name}" || return
  done
  if a1_observability_stack_is_ready \
    "${telemetry_container}" "${foxglove_container}"; then
    a1_success "Persistent Foxglove observability is already ready on port ${FOXGLOVE_PORT}."
    return 0
  fi
  if a1_foxglove_port_is_listening; then
    a1_fail "Foxglove port ${FOXGLOVE_PORT} is owned by an unexpected or unhealthy service."
    return 2
  fi
  a1_remove_runtime_containers "${foxglove_container}" "${telemetry_container}"
  local relative_config="${SYSTEM_CONFIG_PATH#${ROOT}/}"
  if [[ "${relative_config}" == "${SYSTEM_CONFIG_PATH}" ]]; then
    a1_fail "System config must be inside the repository for Docker: ${SYSTEM_CONFIG_PATH}"
    return 2
  fi
  a1_container_run telemetry "${telemetry_container}" \
    "${A1_ROS_PREFIX} && exec python3.12 \
      /workspace/scripts/runtime/a1_observability.py \
      --config '/workspace/${relative_config}'"
  a1_wait_topic "${telemetry_container}" "${OBSERVABILITY_DIAGNOSTICS_TOPIC}"

  local bind_q port_q topics_q services_q no_match_q capabilities_q assets_q update_q buffer_q
  printf -v bind_q '%q' "${FOXGLOVE_BIND}"
  printf -v port_q '%q' "${FOXGLOVE_PORT}"
  printf -v topics_q '%q' "${FOXGLOVE_TOPIC_WHITELIST_YAML}"
  printf -v services_q '%q' "${FOXGLOVE_SERVICE_WHITELIST_YAML}"
  printf -v no_match_q '%q' "${FOXGLOVE_NO_MATCH_ALLOWLIST_YAML}"
  printf -v capabilities_q '%q' "${FOXGLOVE_CAPABILITIES_YAML}"
  printf -v assets_q '%q' "${FOXGLOVE_ASSET_URI_ALLOWLIST_YAML}"
  printf -v update_q '%q' "${FOXGLOVE_GRAPH_UPDATE_MS}"
  printf -v buffer_q '%q' "${FOXGLOVE_SEND_BUFFER_LIMIT_BYTES}"
  a1_container_run core "${foxglove_container}" \
    "${A1_ROS_PREFIX} && exec roslaunch --screen \
      /workspace/scripts/runtime/foxglove_bridge_scoped.launch \
      address:=${bind_q} port:=${port_q} topic_whitelist:=${topics_q} \
      service_whitelist:=${services_q} \
      no_match_allowlist:=${no_match_q} capabilities:=${capabilities_q} \
      asset_uri_allowlist:=${assets_q} max_update_ms:=${update_q} \
      send_buffer_limit:=${buffer_q}"
  local deadline=$((SECONDS + ${FOXGLOVE_STARTUP_TIMEOUT_S%.*}))
  while ! timeout 1 bash -c "</dev/tcp/127.0.0.1/${FOXGLOVE_PORT}" 2>/dev/null; do
    if ! a1_require_running_container "${foxglove_container}" "Foxglove Bridge"; then
      return 1
    fi
    if (( SECONDS >= deadline )); then
      a1_fail "Foxglove Bridge did not listen on port ${FOXGLOVE_PORT}."
      return 1
    fi
    sleep 0.5
  done
}
