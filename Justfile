set shell := ["bash", "-eu", "-o", "pipefail", "-c"]
set quiet := true

# ── Configuration ────────────────────────────────────────────────────────────
# Model
checkpoint   := "/home/eric/4999"
openpi       := "/home/pengyue/Codespace/TFP"
model_config := "pi05_a1_single_arm"
# Paths
uv   := "/home/pengyue/.local/bin/uv"
repo := justfile_directory()
# Teleoperation
teleop_leader_port := "/dev/ttyACM2"

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

# Relay /arm_joint_target_position (JointState) → /arm_joint_command_host (arm_control)
# Required for inference when using /arm_joint_target_position as the control topic.
joint-relay:
    set +u; source /opt/ros/noetic/setup.bash; source {{repo}}/third_party/A1_SDK/install/setup.bash; set -u; \
    {{uv}} run --project {{repo}} python {{repo}}/scripts/inference/joint_target_relay.py

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

# Data collection. Choose mode: drag (bag-replay pipeline) or teleop (SO leader real-time).
# Example: just collect drag
# Example: just collect drag --skip-record
# Example: just collect teleop
# Example: just collect teleop -- --cam0-serial 123456 --task "pick block"
collect action="" *args:
    #!/usr/bin/env bash
    set -euo pipefail
    case "{{action}}" in
        drag)
            scripts/collect_data/dragdatacoach_all_in_one.sh {{args}}
            ;;
        teleop)
            echo "[collect] starting teleop services..."
            AUTO_CONFIRM=1 just teleop
            echo "[collect] starting recorder..."
            trap '' INT; set +e
            {{uv}} run --project {{repo}} python \
                {{repo}}/third_party/lerobot/src/lerobot/scripts/lerobot_a1_collect.py \
                --output-root "{{repo}}/data/a1" \
                {{args}}
            rc=$?; trap - INT; set -e
            if [ "$rc" -eq 130 ]; then exit 0; fi; exit "$rc"
            ;;
        *)
            echo "Usage: just collect <drag|teleop> [args...]"
            echo ""
            echo "  drag    Drag-teach → record bag → replay → collect (all-in-one)"
            echo "  teleop  SO leader real-time teleoperation + direct camera/joint recording"
            exit 1
            ;;
    esac

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

# ── Teleoperation ────────────────────────────────────────────────────────────

# One-time calibration for the SO leader arm.
# Run once before first `just teleop` to generate ~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/my_leader.json
# Example: just calibrate-leader
# Example: just calibrate-leader /dev/ttyACM0 my_other_leader
calibrate-leader port=teleop_leader_port id="my_leader":
    {{repo}}/third_party/lerobot/.venv/bin/lerobot-calibrate \
        --teleop.type=so100_leader \
        --teleop.port="{{port}}" \
        --teleop.id="{{id}}"

# One-time setup: create third_party/lerobot/.venv (Python 3.12) and install lerobot[feetech].
# Run this once before the first `just teleop`.
setup-teleop:
    {{uv}} venv --python 3.12 {{repo}}/third_party/lerobot/.venv
    {{uv}} pip install --python {{repo}}/third_party/lerobot/.venv/bin/python -e "{{repo}}/third_party/lerobot[feetech]"
    {{uv}} pip install --python {{repo}}/third_party/lerobot/.venv/bin/python pyrealsense2
    @echo "teleop env ready: {{repo}}/third_party/lerobot/.venv"

# SO leader → A1 jointTracker teleoperation.
# Starts: roscore → single_arm_node → jointTrackerdemo → SO leader bridge.
# Run `just setup-teleop` once before first use.
# Example: just teleop
# Example: just teleop stop
# Example: LEADER_PORT=/dev/ttyACM0 just teleop
teleop action="start" serial="/dev/a1" leader_port=teleop_leader_port:
    #!/usr/bin/env bash
    set -euo pipefail

    if [[ "{{action}}" == "stop" ]]; then
        run_dir="/tmp/lerobot-teleop"
        found=0
        for pid_file in "$run_dir"/*.pid; do
            [[ -f "$pid_file" ]] || continue
            found=1
            pid="$(cat "$pid_file")"
            name="$(basename "$pid_file" .pid)"
            if kill "$pid" 2>/dev/null; then
                echo "stopped $name (pid=$pid)"
            else
                echo "$name (pid=$pid) already exited"
            fi
            rm -f "$pid_file"
        done
        [[ "$found" -eq 0 ]] && echo "no teleop pid files found in $run_dir"
        exit 0
    fi

    sdk_dir="{{repo}}/third_party/A1_SDK/install"
    bridge_bin="{{repo}}/third_party/lerobot/.venv/bin/lerobot-a1-jointtracker-bridge"
    run_dir="/tmp/lerobot-teleop"
    log_dir="$run_dir/logs"
    tail_pid_file="$run_dir/tail.pids"
    single_arm_serial="${SINGLE_ARM_SERIAL:-{{serial}}}"
    leader_port="${LEADER_PORT:-{{leader_port}}}"

    if [[ ! -x "$bridge_bin" ]]; then
        echo "teleop env not set up. run: just setup-teleop"
        exit 1
    fi

    if [[ ! -e "$single_arm_serial" ]]; then
        echo "single arm serial not found: $single_arm_serial"
        ls -l /dev/ttyACM* /dev/a1 2>/dev/null || true
        exit 1
    fi

    if [[ ! -e "$leader_port" ]]; then
        echo "leader port not found: $leader_port"
        ls -l /dev/ttyACM* 2>/dev/null || true
        exit 1
    fi

    mkdir -p "$log_dir"
    : > "$tail_pid_file"

    cleanup_tails() {
        while IFS= read -r pid; do kill "$pid" 2>/dev/null || true; done < "$tail_pid_file"
    }
    trap cleanup_tails EXIT

    auto_confirm="${AUTO_CONFIRM:-0}"

    start_service() {
        local name="$1" cmd="$2" prompt="$3"
        local suppress_regex="${4:-}" ok_if_log_regex="${5:-}"
        local log_file="$log_dir/${name}.log"
        local pid_file="$run_dir/${name}.pid"

        echo "[$name] starting..."
        : > "$log_file"

        nohup bash -lc "$cmd" >> "$log_file" 2>&1 &
        local svc_pid=$!
        echo "$svc_pid" > "$pid_file"

        if [[ -n "$suppress_regex" ]]; then
            tail -n 0 -F "$log_file" | grep -Eav --line-buffered "$suppress_regex" | sed -u "s/^/[$name] /" &
        else
            tail -n 0 -F "$log_file" | sed -u "s/^/[$name] /" &
        fi
        echo "$!" >> "$tail_pid_file"

        sleep 1
        if ! kill -0 "$svc_pid" 2>/dev/null; then
            if [[ -n "$ok_if_log_regex" ]] && grep -Eq "$ok_if_log_regex" "$log_file"; then
                echo "[$name] detected existing instance, reusing."
                rm -f "$pid_file"
                [[ "$auto_confirm" == "1" ]] && echo "[$name] $prompt (auto-confirmed)" || read -r -p "$prompt"
                return 0
            fi
            echo "[$name] failed to start, check: $log_file"
            exit 1
        fi

        [[ "$auto_confirm" == "1" ]] && echo "[$name] $prompt (auto-confirmed)" || read -r -p "$prompt"
    }

    start_service "roscore" \
        "cd $sdk_dir && source setup.bash && roscore" \
        "roscore ready, press enter to proceed" \
        "" "another roscore/master is already running"

    start_service "single_arm_node" \
        "cd $sdk_dir && source setup.bash && roslaunch signal_arm single_arm_node.launch single_arm_serial_port_path:=$single_arm_serial" \
        "single arm node ready, press enter to proceed"

    start_service "joint_tracker" \
        "cd $sdk_dir && source setup.bash && roslaunch mobiman jointTrackerdemo.launch" \
        "joint tracker ready, press enter to proceed" \
        "model->njoints|start tracking|get target position"

    start_service "bridge" \
        "cd $sdk_dir && source setup.bash && $bridge_bin --leader-port $leader_port --leader-id my_leader --gripper-min-stroke-mm 0 --gripper-max-stroke-mm 200" \
        "bridge running, press enter to finish"

    echo "teleop services running in background."
    echo "logs: $log_dir"
    echo "stop with: just teleop stop"

# ── Inference ────────────────────────────────────────────────────────────────

policy policy_dir=checkpoint:
    PYTHONPATH="{{openpi}}/src:${PYTHONPATH:-}" {{uv}} run --project {{repo}} python {{repo}}/scripts/inference/serve_policy_a1.py policy:checkpoint --policy.config {{model_config}} --policy.dir "{{policy_dir}}"

# ZMQ↔WebSocket bridge: reads state+cameras from ZMQ, calls WebSocket policy server, publishes actions to ZMQ
# Run this alongside `just policy` so the A1 server receives joint actions over ZMQ.
# Example: just zmq-bridge
# Example: just zmq-bridge nyushrobo5090 8000 "pick up the cup"
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
