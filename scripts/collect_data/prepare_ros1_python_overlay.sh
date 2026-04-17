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

# Patch rosgraph/roslogging.py for Python 3.12+ compatibility.
# The original findCaller() enters an infinite loop because Python 3.12
# changed logging.Logger.findCaller return values.  We copy rosgraph
# (instead of symlinking) and apply a max-depth guard.
_rg="${OVERLAY_ROOT}/rosgraph"
if [[ -L "${_rg}" ]]; then
  _rg_target="$(readlink -f "${_rg}")"
  rm "${_rg}"
  cp -r "${_rg_target}" "${_rg}"
  sed -i 's/while hasattr(f, "f_code"):/max_depth = 20\n        while hasattr(f, "f_code") and max_depth > 0:\n            max_depth -= 1/' "${_rg}/roslogging.py"
fi

# Generate signal_arm Python message bindings from A1 SDK .msg definitions.
# genpy produces partial output even when Header.msg is missing from the overlay
# (Header-dependent messages get empty stubs), but the package becomes importable
# which is sufficient for the doctor check and teleop bridge at runtime.
A1_SDK_SHARE="${PROJECT_ROOT}/third_party/A1_SDK/install/share"
SA_MSG_DIR="${A1_SDK_SHARE}/signal_arm/msg"
SA_PKG_DIR="${OVERLAY_ROOT}/signal_arm"
SA_OUT_DIR="${SA_PKG_DIR}/msg"

if [[ -d "${SA_MSG_DIR}" ]]; then
  mkdir -p "${SA_OUT_DIR}"
  touch "${SA_PKG_DIR}/__init__.py"

  OVERLAY_ROOT="${OVERLAY_ROOT}" SA_MSG_DIR="${SA_MSG_DIR}" SA_OUT_DIR="${SA_OUT_DIR}" \
    PYTHONPATH="${OVERLAY_ROOT}:${PYTHONPATH:-}" python3 - 2>/dev/null <<'GENPY'
import os, sys

overlay = os.environ["OVERLAY_ROOT"]
msg_dir = os.environ["SA_MSG_DIR"]
outdir  = os.environ["SA_OUT_DIR"]

sys.path.insert(0, overlay)
import genpy.generator

msg_defs = os.path.join(os.path.dirname(overlay), 'ros1_msg_defs')
search_path = {
    'std_msgs':    [os.path.join(msg_defs, 'std_msgs')] if os.path.isdir(os.path.join(msg_defs, 'std_msgs')) else [os.path.join(overlay, 'std_msgs', 'msg')],
    'sensor_msgs': [os.path.join(msg_defs, 'sensor_msgs')] if os.path.isdir(os.path.join(msg_defs, 'sensor_msgs')) else [os.path.join(overlay, 'sensor_msgs', 'msg')],
    'signal_arm':  [msg_dir],
}
gen = genpy.generator.MsgGenerator()
for f in sorted(os.listdir(msg_dir)):
    if f.endswith('.msg'):
        gen.generate_messages('signal_arm', [os.path.join(msg_dir, f)], outdir, search_path)

imports = []
for f in sorted(os.listdir(outdir)):
    if f.startswith('_') and f.endswith('.py') and f != '__init__.py':
        imports.append(f'from .{f[:-3]} import *')
with open(os.path.join(outdir, '__init__.py'), 'w') as fp:
    fp.write('\n'.join(imports) + '\n')
GENPY
fi

echo "${OVERLAY_ROOT}"
