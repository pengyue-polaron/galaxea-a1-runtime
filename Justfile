set shell := ["bash", "-eu", "-o", "pipefail", "-c"]
set quiet := true

default:
    @just --list

# ---------- Environment ----------

doctor:
    scripts/collect_data/dragdatacoach.sh doctor

which-python:
    scripts/collect_data/dragdatacoach.sh which-python

# ---------- Command Groups ----------

launch target="driver" serial="/dev/ttyACM0":
    case "{{target}}" in driver) scripts/collect_data/dragdatacoach.sh launch-driver "{{serial}}" ;; ee-record) scripts/collect_data/dragdatacoach.sh launch-ee-record "{{serial}}" ;; tracker) scripts/collect_data/dragdatacoach.sh launch-tracker ;; *) echo "Usage: just launch <driver|ee-record|tracker> [serial]"; exit 1 ;; esac

drag action="start" *args:
    case "{{action}}" in start) scripts/collect_data/dragdatacoach.sh drag-start {{args}} ;; stop) scripts/collect_data/dragdatacoach.sh drag-stop ;; *) echo "Usage: just drag <start|stop> [args...]"; exit 1 ;; esac

gripper action="start" *args:
    case "{{action}}" in start|keyboard) scripts/collect_data/dragdatacoach.sh gripper-keyboard {{args}} ;; stop) scripts/collect_data/dragdatacoach.sh gripper-stop ;; *) echo "Usage: just gripper <start|stop> [args...]"; exit 1 ;; esac

record action="start" tag="drag_demo":
    case "{{action}}" in start) scripts/collect_data/dragdatacoach.sh record-start "{{tag}}" ;; stop) scripts/collect_data/dragdatacoach.sh record-stop ;; *) echo "Usage: just record <start|stop> [tag]"; exit 1 ;; esac

replay bag="" rate="1.0" gripper_mode="position":
    BAG="{{bag}}"; if [ -z "$BAG" ]; then BAG=$(ls -t third_party/A1_SDK/data/records/*.bag | head -n 1); fi; scripts/collect_data/dragdatacoach.sh replay --bag "$BAG" --gripper-mode "{{gripper_mode}}" --rate "{{rate}}"

collect:
    trap '' INT; set +e; scripts/collect_data/dragdatacoach.sh collect; rc=$?; trap - INT; set -e; if [ "$rc" -eq 130 ]; then exit 0; fi; exit "$rc"

drag-collect *args:
    scripts/collect_data/dragdatacoach_all_in_one.sh {{args}}

camera action="test" *args:
    PY=$(scripts/collect_data/dragdatacoach.sh which-python); case "{{action}}" in test) "$PY" scripts/collect_data/test_camera_connections.py --config configs/drag_replay.yaml --timeout-s 6.0 {{args}} ;; raw) "$PY" scripts/collect_data/test_camera_connections.py {{args}} ;; *) echo "Usage: just camera <test|raw> [args...]"; exit 1 ;; esac

bag action="latest" bag="":
    case "{{action}}" in latest) ls -t third_party/A1_SDK/data/records/*.bag | head -n 1 ;; info) if [ -z "{{bag}}" ]; then echo "Usage: just bag info <bag>"; exit 1; fi; set +u; source /opt/ros/noetic/setup.bash; set -u; rosbag info "{{bag}}" ;; *) echo "Usage: just bag <latest|info> [bag]"; exit 1 ;; esac
