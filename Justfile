set shell := ["bash", "-eu", "-o", "pipefail", "-c"]
set quiet := true

# ── Configuration ────────────────────────────────────────────────────────────
# Model
checkpoint   := "/home/eric/4999"
openpi       := "/home/eric/openpi"
model_config := "pi05_a1_single_arm"
# Paths
uv   := "/home/pengyue/.local/bin/uv"
repo := justfile_directory()

default:
    @just --list

# ── Environment ──────────────────────────────────────────────────────────────

doctor:
    scripts/collect_data/dragdatacoach.sh doctor

which-python:
    scripts/collect_data/dragdatacoach.sh which-python

# ── Commands ─────────────────────────────────────────────────────────────────

launch target="driver" serial="/dev/a1":
    case "{{target}}" in \
        roscore) \
            set +u; source /opt/ros/noetic/setup.bash; set -u; \
            roscore ;; \
        camera-server) \
            set +u; source /opt/ros/noetic/setup.bash; source {{repo}}/third_party/A1_SDK/install/setup.bash; set -u; \
            {{uv}} run --project {{repo}} python scripts/collect_data/run_data_services.py service_mode=live "a1_server.components=[]" ;; \
        joint-tracker) \
            set +u; source /opt/ros/noetic/setup.bash; source {{repo}}/third_party/A1_SDK/install/setup.bash; set -u; \
            roslaunch mobiman jointTrackerdemo.launch ;; \
        ee-tracker) \
            set +u; source /opt/ros/noetic/setup.bash; source {{repo}}/third_party/A1_SDK/install/setup.bash; set -u; \
            roslaunch mobiman eeTrackerdemo.launch ;; \
        a1-server) \
            set +u; source /opt/ros/noetic/setup.bash; source {{repo}}/third_party/A1_SDK/install/setup.bash; set -u; \
            {{uv}} run --project {{repo}} python scripts/collect_data/run_a1_server.py "a1_server.components=[ros_subscriber,policy_action_subscriber]" ;; \
        driver) \
            scripts/collect_data/dragdatacoach.sh launch-driver "{{serial}}" ;; \
        ee-record) \
            scripts/collect_data/dragdatacoach.sh launch-ee-record "{{serial}}" ;; \
        tracker) \
            scripts/collect_data/dragdatacoach.sh launch-tracker ;; \
        *) echo "Usage: just launch <roscore|camera-server|joint-tracker|ee-tracker|a1-server|driver|ee-record|tracker> [serial]"; exit 1 ;; \
    esac

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
    PY=$(scripts/collect_data/dragdatacoach.sh which-python)
    case "{{target}}" in \
        camera)     "$PY" scripts/collect_data/test_camera_connections.py --config configs/drag_replay.yaml --timeout-s 6.0 {{args}} ;; \
        camera-raw) "$PY" scripts/collect_data/test_camera_connections.py {{args}} ;; \
        *) echo "Usage: just test <camera|camera-raw> [args...]"; exit 1 ;; \
    esac

print target="joints" count="0" unit="deg":
    case "{{target}}" in \
        joints) \
            set +u; source /opt/ros/noetic/setup.bash; source third_party/A1_SDK/install/setup.bash; set -u; \
            python3 scripts/collect_data/print_joint_angles.py --count "{{count}}" --unit "{{unit}}" ;; \
        *) echo "Usage: just print joints [count] [unit]"; exit 1 ;; \
    esac

bag action="latest" bag="":
    case "{{action}}" in \
        latest) ls -t third_party/A1_SDK/data/records/*.bag | head -n 1 ;; \
        info) \
            if [ -z "{{bag}}" ]; then echo "Usage: just bag info <bag>"; exit 1; fi; \
            set +u; source /opt/ros/noetic/setup.bash; set -u; \
            rosbag info "{{bag}}" ;; \
        *) echo "Usage: just bag <latest|info> [bag]"; exit 1 ;; \
    esac

# ── Inference ────────────────────────────────────────────────────────────────

policy policy_dir=checkpoint:
    PYTHONPATH="{{openpi}}/src:${PYTHONPATH:-}" {{uv}} run --project {{repo}} python {{repo}}/scripts/inference/serve_policy_a1.py policy:checkpoint --policy.config {{model_config}} --policy.dir "{{policy_dir}}"

# ZMQ↔WebSocket bridge: reads state+cameras from ZMQ, calls WebSocket policy server, publishes actions to ZMQ
# Run this alongside `just policy` so the A1 server receives joint actions over ZMQ.
# Example: just zmq-bridge
# Example: just zmq-bridge --prompt "pick up the cup" --action-chunk-size 3
zmq-bridge host="127.0.0.1" port="8000" prompt="swap the position of the marker and the yellow block" *args:
    PYTHONPATH="{{openpi}}/packages/openpi-client/src:${PYTHONPATH:-}" {{uv}} run --project {{repo}} python {{repo}}/scripts/inference/zmq_ws_bridge.py --host "{{host}}" --port "{{port}}" --prompt "{{prompt}}" {{args}}

# Teacher-forcing: offline policy inference on training images → trajectory.json + trajectory.html
# Requires: just policy (WebSocket server) running first.
# Example: just teacher-forcing
# Example: just teacher-forcing demo_0
# Example: just teacher-forcing demo_0 -- --max-steps 100
teacher-forcing demo="" processed_root="{{repo}}/data/processed_data/pick_twice" *args:
    PYTHONPATH="{{openpi}}/packages/openpi-client/src:${PYTHONPATH:-}" {{uv}} run --project {{repo}} python {{repo}}/scripts/inference/teacher_forcing_infer.py --processed-root "{{processed_root}}" $([ -n "{{demo}}" ] && echo "--demo {{demo}}") {{args}}

# Open-loop rollout eval on processed data → per-demo trajectory.json + trajectory.html
# Example: just openloop-rollout --policy-dir /home/eric/4999
# Example: just openloop-rollout --policy-dir /home/eric/4999 --max-demos 1 --max-steps-per-demo 100
openloop-rollout policy_dir="" *args:
    if [ -z "{{policy_dir}}" ]; then echo "Usage: just openloop-rollout <policy_dir> [extra args...]"; exit 1; fi
    PYTHONPATH="{{openpi}}/src:{{repo}}:${PYTHONPATH:-}" {{uv}} run --project {{repo}} python {{repo}}/scripts/inference/openloop_rollout.py --policy-dir "{{policy_dir}}" {{args}}


# ── Debug ────────────────────────────────────────────────────────────────────

debug target="camera" output_dir="{{repo}}/data/debug/model_input_frames" every_n="20" max_per_cam="300" duration_s="30" *args:
    case "{{target}}" in \
        camera) {{uv}} run --project {{repo}} python {{repo}}/scripts/inference/dump_model_input_images.py --output-dir "{{output_dir}}" --every-n "{{every_n}}" --max-per-cam "{{max_per_cam}}" --duration-s "{{duration_s}}" {{args}} ;; \
        *) echo "Usage: just debug <camera> [output_dir] [every_n] [max_per_cam] [duration_s] [extra args...]"; exit 1 ;; \
    esac
