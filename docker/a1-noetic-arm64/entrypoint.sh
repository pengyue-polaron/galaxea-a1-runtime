#!/usr/bin/env bash
set -e

if [[ -f /opt/ros/noetic/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/noetic/setup.bash
fi

if [[ -n "${A1_SDK_ROOT:-}" ]] && [[ -f "${A1_SDK_ROOT}/install/setup.bash" ]]; then
  # shellcheck disable=SC1090
  source "${A1_SDK_ROOT}/install/setup.bash"
fi

exec "$@"
