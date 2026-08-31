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

if [[ -f /opt/foxglove_bridge_ws/install/local_setup.bash ]]; then
  # Extend the active A1 SDK workspace without replacing it.
  # shellcheck disable=SC1091
  source /opt/foxglove_bridge_ws/install/local_setup.bash
  export ROS_PACKAGE_PATH="/opt/foxglove_bridge_ws/install/share:${ROS_PACKAGE_PATH}"
fi

exec "$@"
