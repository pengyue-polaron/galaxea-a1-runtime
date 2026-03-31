#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OVERLAY_ROOT="${PROJECT_ROOT}/.cache/ros1_python_overlay"
SYSTEM_DIST_PACKAGES="/usr/lib/python3/dist-packages"

ROS_MODULES=(
  catkin
  catkin_pkg
  genmsg
  genpy
  geometry_msgs
  rosgraph
  rosgraph_msgs
  roslaunch
  roslib
  rosmaster
  rosnode
  roscpp
  rospkg
  rospy
  rostopic
  sensor_msgs
  std_msgs
)

mkdir -p "${OVERLAY_ROOT}"
find "${OVERLAY_ROOT}" -mindepth 1 -maxdepth 1 -exec rm -rf {} +

for name in "${ROS_MODULES[@]}"; do
  src="${SYSTEM_DIST_PACKAGES}/${name}"
  dst="${OVERLAY_ROOT}/${name}"
  if [[ ! -e "${src}" ]]; then
    echo "[WARN] missing ROS python module: ${src}" >&2
    continue
  fi
  ln -snf "${src}" "${dst}"
done

echo "${OVERLAY_ROOT}"
