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

launch target="driver" serial="/dev/a1":
    case "{{target}}" in roscore) set +u; source /opt/ros/noetic/setup.bash; set -u; roscore ;; camera-server) set +u; source /opt/ros/noetic/setup.bash; source /home/pengyue/Codespace/DataCoach/third_party/A1_SDK/install/setup.bash; set -u; /home/jolia/.local/bin/uv run --project /home/jolia/DataCoach python scripts/collect_data/run_data_services.py service_mode=live "a1_server.components=[]" ;; ee-tracker) set +u; source /opt/ros/noetic/setup.bash; source /home/pengyue/Codespace/DataCoach/third_party/A1_SDK/install/setup.bash; set -u; roslaunch mobiman eeTrackerdemo.launch rviz:=false ;; a1-server) set +u; source /opt/ros/noetic/setup.bash; source /home/pengyue/Codespace/DataCoach/third_party/A1_SDK/install/setup.bash; set -u; /home/jolia/.local/bin/uv run --project /home/jolia/DataCoach python scripts/collect_data/run_a1_server.py ;; driver) scripts/collect_data/dragdatacoach.sh launch-driver "{{serial}}" ;; ee-record) scripts/collect_data/dragdatacoach.sh launch-ee-record "{{serial}}" ;; tracker) scripts/collect_data/dragdatacoach.sh launch-tracker ;; *) echo "Usage: just launch <roscore|camera-server|ee-tracker|a1-server|driver|ee-record|tracker> [serial]"; exit 1 ;; esac

drag action="start" *args:
    case "{{action}}" in start) scripts/collect_data/dragdatacoach.sh drag-start {{args}} ;; stop) scripts/collect_data/dragdatacoach.sh drag-stop ;; *) echo "Usage: just drag <start|stop> [args...]"; exit 1 ;; esac

gripper action="start" *args:
    case "{{action}}" in start|keyboard) scripts/collect_data/dragdatacoach.sh gripper-keyboard {{args}} ;; open) scripts/collect_data/dragdatacoach.sh gripper-open {{args}} ;; close) scripts/collect_data/dragdatacoach.sh gripper-close {{args}} ;; stop) scripts/collect_data/dragdatacoach.sh gripper-stop ;; *) echo "Usage: just gripper <start|keyboard|open|close|stop> [args...]"; exit 1 ;; esac

record action="start" tag="drag_demo":
    case "{{action}}" in start) scripts/collect_data/dragdatacoach.sh record-start "{{tag}}" ;; stop) scripts/collect_data/dragdatacoach.sh record-stop ;; *) echo "Usage: just record <start|stop> [tag]"; exit 1 ;; esac

replay bag="" rate="1.0" gripper_mode="position":
    BAG="{{bag}}"; if [ -z "$BAG" ]; then BAG=$(ls -t third_party/A1_SDK/data/records/*.bag | head -n 1); fi; scripts/collect_data/dragdatacoach.sh replay --bag "$BAG" --gripper-mode "{{gripper_mode}}" --rate "{{rate}}"

collect:
    trap '' INT; set +e; scripts/collect_data/dragdatacoach.sh collect; rc=$?; trap - INT; set -e; if [ "$rc" -eq 130 ]; then exit 0; fi; exit "$rc"

drag-collect *args:
    scripts/collect_data/dragdatacoach_all_in_one.sh {{args}}

test target="camera" *args:
    PY=$(scripts/collect_data/dragdatacoach.sh which-python); case "{{target}}" in camera) "$PY" scripts/collect_data/test_camera_connections.py --config configs/drag_replay.yaml --timeout-s 6.0 {{args}} ;; camera-raw) "$PY" scripts/collect_data/test_camera_connections.py {{args}} ;; *) echo "Usage: just test <camera|camera-raw> [args...]"; exit 1 ;; esac

print target="joints" count="0" unit="deg":
    case "{{target}}" in joints) set +u; source /opt/ros/noetic/setup.bash; source third_party/A1_SDK/install/setup.bash; set -u; python3 scripts/collect_data/print_joint_angles.py --count "{{count}}" --unit "{{unit}}" ;; *) echo "Usage: just print joints [count] [unit]"; exit 1 ;; esac

bag action="latest" bag="":
    case "{{action}}" in latest) ls -t third_party/A1_SDK/data/records/*.bag | head -n 1 ;; info) if [ -z "{{bag}}" ]; then echo "Usage: just bag info <bag>"; exit 1; fi; set +u; source /opt/ros/noetic/setup.bash; set -u; rosbag info "{{bag}}" ;; *) echo "Usage: just bag <latest|info> [bag]"; exit 1 ;; esac

# ---------- Inference ----------

policy policy_dir="/home/pengyue/29000":
    PYTHONPATH="/home/pengyue/Codespace/DataCoach:${PYTHONPATH:-}" /home/jolia/.local/bin/uv run --project /home/jolia/DataCoach python /home/pengyue/Codespace/DataCoach/scripts/inference/my_serve_policy.py policy:checkpoint --policy.config pi05_ltc_pick_twice --policy.dir "{{policy_dir}}"
