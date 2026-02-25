set shell := ["bash", "-eu", "-o", "pipefail", "-c"]
set quiet := true

default:
    @just --list

# ---------- Environment ----------

doctor:
    scripts/collect_data/dragdatacoach.sh doctor

which-python:
    scripts/collect_data/dragdatacoach.sh which-python

camera-test:
    PY=$(scripts/collect_data/dragdatacoach.sh which-python); "$PY" scripts/collect_data/test_camera_connections.py --config configs/drag_replay.yaml --timeout-s 6.0

camera-test-raw *args:
    PY=$(scripts/collect_data/dragdatacoach.sh which-python); "$PY" scripts/collect_data/test_camera_connections.py {{args}}

# ---------- Recording ----------

launch-driver serial="/dev/ttyACM0":
    scripts/collect_data/dragdatacoach.sh launch-driver "{{serial}}"

launch-ee-record serial="/dev/ttyACM0":
    scripts/collect_data/dragdatacoach.sh launch-ee-record "{{serial}}"

drag-start:
    scripts/collect_data/dragdatacoach.sh drag-start

drag-stop:
    scripts/collect_data/dragdatacoach.sh drag-stop

gripper-keyboard:
    scripts/collect_data/dragdatacoach.sh gripper-keyboard

gripper-stop:
    scripts/collect_data/dragdatacoach.sh gripper-stop

record-start tag="drag_demo":
    scripts/collect_data/dragdatacoach.sh record-start "{{tag}}"

record-stop:
    scripts/collect_data/dragdatacoach.sh record-stop

# ---------- Replay ----------

launch-tracker:
    scripts/collect_data/dragdatacoach.sh launch-tracker

replay bag rate="1.0" gripper_mode="position":
    scripts/collect_data/dragdatacoach.sh replay --bag "{{bag}}" --gripper-mode "{{gripper_mode}}" --rate "{{rate}}"

replay-arm-only bag rate="1.0":
    scripts/collect_data/dragdatacoach.sh replay --bag "{{bag}}" --gripper-mode none --rate "{{rate}}"

collect:
    trap '' INT; set +e; scripts/collect_data/dragdatacoach.sh collect; rc=$?; trap - INT; set -e; if [ "$rc" -eq 130 ]; then exit 0; fi; exit "$rc"

# ---------- Utilities ----------

latest-bag:
    ls -t third_party/A1_SDK/data/records/*.bag | head -n 1

bag-info bag:
    set +u; source /opt/ros/noetic/setup.bash; set -u; rosbag info "{{bag}}"
