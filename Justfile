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
    case "{{target}}" in roscore) set +u; source /opt/ros/noetic/setup.bash; set -u; roscore ;; camera-server) set +u; source /opt/ros/noetic/setup.bash; source /home/pengyue/Codespace/DataCoach/third_party/A1_SDK/install/setup.bash; set -u; /home/jolia/.local/bin/uv run --project /home/jolia/DataCoach python scripts/collect_data/run_data_services.py service_mode=live "a1_server.components=[]" ;; joint-tracker) set +u; source /opt/ros/noetic/setup.bash; source /home/pengyue/Codespace/DataCoach/third_party/A1_SDK/install/setup.bash; set -u; roslaunch mobiman jointTrackerdemo.launch ;; ee-tracker) set +u; source /opt/ros/noetic/setup.bash; source /home/pengyue/Codespace/DataCoach/third_party/A1_SDK/install/setup.bash; set -u; roslaunch mobiman eeTrackerdemo.launch ;; a1-server) set +u; source /opt/ros/noetic/setup.bash; source /home/pengyue/Codespace/DataCoach/third_party/A1_SDK/install/setup.bash; set -u; /home/jolia/.local/bin/uv run --project /home/jolia/DataCoach python scripts/collect_data/run_a1_server.py "a1_server.components=[ros_subscriber,policy_action_subscriber]" ;; driver) scripts/collect_data/dragdatacoach.sh launch-driver "{{serial}}" ;; ee-record) scripts/collect_data/dragdatacoach.sh launch-ee-record "{{serial}}" ;; tracker) scripts/collect_data/dragdatacoach.sh launch-tracker ;; *) echo "Usage: just launch <roscore|camera-server|joint-tracker|ee-tracker|a1-server|driver|ee-record|tracker> [serial]"; exit 1 ;; esac

joint-tracker:
    just launch joint-tracker

ee-tracker mode="" *args:
    case "{{mode}}" in ""|run) just launch ee-tracker ;; -drag|drag) scripts/collect_data/dragdatacoach.sh ee-tracker-drag {{args}} ;; *) echo "Usage: just ee-tracker [run|-drag|drag] [args...]"; exit 1 ;; esac

drag action="start" *args:
    case "{{action}}" in start) scripts/collect_data/dragdatacoach.sh drag-start {{args}} ;; stop) scripts/collect_data/dragdatacoach.sh drag-stop ;; *) echo "Usage: just drag <start|stop> [args...]"; exit 1 ;; esac

gripper action="start" *args:
    case "{{action}}" in start|keyboard) scripts/collect_data/dragdatacoach.sh gripper-keyboard {{args}} ;; open) scripts/collect_data/dragdatacoach.sh gripper-open {{args}} ;; close) scripts/collect_data/dragdatacoach.sh gripper-close {{args}} ;; stop) scripts/collect_data/dragdatacoach.sh gripper-stop ;; *) echo "Usage: just gripper <start|keyboard|open|close|stop> [args...]"; exit 1 ;; esac

record action="start" tag="drag_demo":
    case "{{action}}" in start) scripts/collect_data/dragdatacoach.sh record-start "{{tag}}" ;; stop) scripts/collect_data/dragdatacoach.sh record-stop ;; *) echo "Usage: just record <start|stop> [tag]"; exit 1 ;; esac

replay bag="" rate="1.0" gripper_mode="position":
    BAG="{{bag}}"; if [ -z "$BAG" ]; then BAG=$(ls -t third_party/A1_SDK/data/records/*.bag | head -n 1); fi; scripts/collect_data/dragdatacoach.sh replay --bag "$BAG" --gripper-mode "{{gripper_mode}}" --rate "{{rate}}"

replay-infer input="" source="auto" rate="15" speed="1.0" *args:
    if [ -z "{{input}}" ]; then echo "Usage: just replay-infer <input> [source] [rate] [speed] [extra args...]"; exit 1; fi; scripts/collect_data/dragdatacoach.sh replay-infer --input "{{input}}" --source "{{source}}" --rate "{{rate}}" --speed "{{speed}}" {{args}}

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

policy policy_dir="/home/pengyue/6000":
    PYTHONPATH="/home/pengyue/Codespace/DataCoach/third_party/openpi/src:/home/pengyue/Codespace/DataCoach:${PYTHONPATH:-}" /home/jolia/.local/bin/uv run --project /home/jolia/DataCoach python /home/pengyue/Codespace/DataCoach/scripts/inference/my_serve_policy.py policy:checkpoint --policy.config pi05_ltc_pick_twice --policy.dir "{{policy_dir}}"

# Teacher-forcing: offline policy inference on training images → trajectory.json + trajectory.html
# Example: just teacher-forcing demo_0_20260227_225247
# Example: just teacher-forcing demo_0_20260227_225247 -- --max-steps 100
teacher-forcing demo="" processed_root="/home/jolia/DataCoach/data/processed_data/swap" policy_dir="/home/pengyue/6000" *args:
    PYTHONPATH="/home/pengyue/Codespace/DataCoach/third_party/openpi/src:/home/pengyue/Codespace/DataCoach:${PYTHONPATH:-}" /home/jolia/.local/bin/uv run --project /home/jolia/DataCoach python /home/pengyue/Codespace/DataCoach/scripts/inference/teacher_forcing_infer.py --processed-root "{{processed_root}}" --policy-dir "{{policy_dir}}" $([ -n "{{demo}}" ] && echo "--demo {{demo}}") {{args}}

# Open-loop rollout eval on processed data → per-demo trajectory.json + trajectory.html
# Example: just openloop-rollout --policy-dir /home/pengyue/6000
# Example: just openloop-rollout --policy-dir /home/pengyue/6000 --max-demos 1 --max-steps-per-demo 100
openloop-rollout policy_dir="" *args:
    if [ -z "{{policy_dir}}" ]; then echo "Usage: just openloop-rollout <policy_dir> [extra args...]"; exit 1; fi; PYTHONPATH="/home/pengyue/Codespace/DataCoach/third_party/openpi/src:/home/pengyue/Codespace/DataCoach:${PYTHONPATH:-}" /home/jolia/.local/bin/uv run --project /home/jolia/DataCoach python /home/pengyue/Codespace/DataCoach/scripts/inference/openloop_rollout.py --policy-dir "{{policy_dir}}" {{args}}

# Zero-shot DROID bridge: streams A1 state+cameras to pi05_droid WebSocket server
# Requires: pi05_droid server running (just policy-droid)
# Example: just droid-bridge
# Example: just droid-bridge --prompt "pick up the cup"
droid-bridge prompt="swap the position of the marker and the yellow block through the white plate" *args:
    PYTHONPATH="/home/pengyue/Codespace/openpi/packages/openpi-client/src:/home/pengyue/Codespace/DataCoach:${PYTHONPATH:-}" /home/jolia/.local/bin/uv run --project /home/jolia/DataCoach python /home/pengyue/Codespace/DataCoach/scripts/inference/droid_zero_shot_bridge.py --prompt "{{prompt}}" {{args}}

# Zero-shot DROID EEF bridge: streams A1 cameras+joints to pi05_droid, publishes EEF targets via ROS
# Requires: ee-tracker + camera-server + policy-droid all running
# Example: just droid-eef-bridge
# Example: just droid-eef-bridge "pick up the cup" --pos-scale 0.3
droid-eef-bridge prompt="swap the position of the marker and the yellow block through the white plate" flip_axes="y" *args:
    PYTHONPATH="/home/pengyue/Codespace/openpi/packages/openpi-client/src:/home/pengyue/Codespace/DataCoach:${PYTHONPATH:-}" /home/jolia/.local/bin/uv run --project /home/jolia/DataCoach python /home/pengyue/Codespace/DataCoach/scripts/inference/droid_eef_bridge.py "{{prompt}}" --flip-axes "{{flip_axes}}" {{args}}

# Start pi05_droid policy server (WebSocket on port 8000)
policy-droid policy_dir="/home/pengyue/pi05_droid":
    PYTHONPATH="/home/pengyue/Codespace/openpi/src:${PYTHONPATH:-}" /home/jolia/.local/bin/uv run --project /home/jolia/DataCoach python /home/pengyue/Codespace/openpi/scripts/serve_policy.py policy:checkpoint --policy.config pi05_droid --policy.dir "{{policy_dir}}"

# Teacher-forcing with pi05_droid on training data → trajectory.html
# Requires: just policy-droid running
# Example: just droid-teacher-forcing
# Example: just droid-teacher-forcing demo_0_20260227_225247
# Example: just droid-teacher-forcing demo_0_20260227_225247 -- --max-steps 100
droid-teacher-forcing demo="" *args:
    PYTHONPATH="/home/pengyue/Codespace/openpi/packages/openpi-client/src:/home/pengyue/Codespace/DataCoach:${PYTHONPATH:-}" /home/jolia/.local/bin/uv run --project /home/jolia/DataCoach python /home/pengyue/Codespace/DataCoach/scripts/inference/droid_teacher_forcing.py $([ -n "{{demo}}" ] && echo "--demo {{demo}}") {{args}}

# ---------- Debug ----------

debug target="camera" output_dir="/home/pengyue/Codespace/DataCoach/data/debug/model_input_frames" every_n="20" max_per_cam="300" duration_s="30" *args:
    case "{{target}}" in camera) /home/jolia/.local/bin/uv run --project /home/jolia/DataCoach python /home/pengyue/Codespace/DataCoach/scripts/inference/dump_model_input_images.py --output-dir "{{output_dir}}" --every-n "{{every_n}}" --max-per-cam "{{max_per_cam}}" --duration-s "{{duration_s}}" {{args}} ;; *) echo "Usage: just debug <camera> [output_dir] [every_n] [max_per_cam] [duration_s] [extra args...]"; exit 1 ;; esac
