set shell := ["bash", "-eu", "-o", "pipefail", "-c"]
set quiet := true

# ── Configuration ────────────────────────────────────────────────────────────
# Model
checkpoint   := env("A1_CHECKPOINT", "checkpoints/latest")
openpi       := env("OPENPI_ROOT", justfile_directory() + "/third_party/openpi")
model_config := env("A1_MODEL_CONFIG", "pi05_a1_banana_red_plate")
# Paths
uv   := env("UV_BIN", "uv")
repo := justfile_directory()
vpy  := repo + "/.venv/bin/python"
# Teleoperation
teleop_leader_port := "/dev/ttyACM0"

default:
    @just --list

# ── Environment ──────────────────────────────────────────────────────────────

doctor:
    scripts/collect_data/a1.sh doctor

which-python:
    scripts/collect_data/a1.sh which-python

# ── Commands ─────────────────────────────────────────────────────────────────

launch target="driver" serial="/dev/a1":
    #!/usr/bin/env bash
    set -eu -o pipefail
    _source_ros() {
        set +u
        [[ -f /opt/ros/noetic/setup.bash ]] && source /opt/ros/noetic/setup.bash
        [[ -f "{{repo}}/third_party/A1_SDK/install/setup.bash" ]] && source "{{repo}}/third_party/A1_SDK/install/setup.bash"
        set -u
    }
    case "{{target}}" in
        roscore)
            _source_ros; roscore ;;
        camera-server)
            _source_ros
            {{uv}} run --project {{repo}} python scripts/collect_data/run_data_services.py service_mode=live "a1_server.components=[]" ;;
        joint-tracker)
            _source_ros; roslaunch mobiman jointTrackerdemo.launch ;;
        ee-tracker)
            _source_ros; roslaunch mobiman eeTrackerdemo.launch ;;
        a1-server)
            _source_ros
            {{uv}} run --project {{repo}} python scripts/collect_data/run_a1_server.py "a1_server.components=[ros_subscriber,policy_action_subscriber]" ;;
        driver)
            scripts/collect_data/a1.sh launch-driver "{{serial}}" ;;
        ee-record)
            scripts/collect_data/a1.sh launch-ee-record "{{serial}}" ;;
        tracker)
            scripts/collect_data/a1.sh launch-tracker ;;
        *) echo "Usage: just launch <roscore|camera-server|joint-tracker|ee-tracker|a1-server|driver|ee-record|tracker> [serial]"; exit 1 ;;
    esac

joint-tracker:
    just launch joint-tracker

# Relay /arm_joint_target_position (JointState) → /arm_joint_command_host (arm_control)
# Required for inference when using /arm_joint_target_position as the control topic.
joint-relay:
    set +u; source /opt/ros/noetic/setup.bash; source {{repo}}/third_party/A1_SDK/install/setup.bash; set -u; \
    {{uv}} run --project {{repo}} python {{repo}}/scripts/inference/joint_target_relay.py

ee-tracker mode="" *args:
    case "{{mode}}" in ""|run) just launch ee-tracker ;; -drag|drag) scripts/collect_data/a1.sh ee-tracker-drag {{args}} ;; *) echo "Usage: just ee-tracker [run|-drag|drag] [args...]"; exit 1 ;; esac

drag action="start" *args:
    case "{{action}}" in start) scripts/collect_data/a1.sh drag-start {{args}} ;; stop) scripts/collect_data/a1.sh drag-stop ;; *) echo "Usage: just drag <start|stop> [args...]"; exit 1 ;; esac

gripper action="start" *args:
    case "{{action}}" in start|keyboard) scripts/collect_data/a1.sh gripper-keyboard {{args}} ;; open) scripts/collect_data/a1.sh gripper-open {{args}} ;; close) scripts/collect_data/a1.sh gripper-close {{args}} ;; stop) scripts/collect_data/a1.sh gripper-stop ;; *) echo "Usage: just gripper <start|keyboard|open|close|stop> [args...]"; exit 1 ;; esac

record action="start" tag="drag_demo":
    case "{{action}}" in start) scripts/collect_data/a1.sh record-start "{{tag}}" ;; stop) scripts/collect_data/a1.sh record-stop ;; *) echo "Usage: just record <start|stop> [tag]"; exit 1 ;; esac

replay bag="" rate="1.0" gripper_mode="position":
    BAG="{{bag}}"; if [ -z "$BAG" ]; then BAG=$(ls -t third_party/A1_SDK/data/records/*.bag | head -n 1); fi; scripts/collect_data/a1.sh replay --bag "$BAG" --gripper-mode "{{gripper_mode}}" --rate "{{rate}}"

replay-infer input="" source="auto" rate="15" speed="1.0" *args:
    if [ -z "{{input}}" ]; then echo "Usage: just replay-infer <input> [source] [rate] [speed] [extra args...]"; exit 1; fi; scripts/collect_data/a1.sh replay-infer --input "{{input}}" --source "{{source}}" --rate "{{rate}}" --speed "{{speed}}" {{args}}

# Data collection.
# Teleop mode: starts services, then loops recording episodes.
#   First run: prompts for task description (saved to task.txt, never asked again).
#   Enter=start, Enter=save, d+Enter=discard, Ctrl+C=quit.
# Example: just collect pick_block
# Example: just collect pick_block --fps 20
collect experiment *args:
    #!/usr/bin/env bash
    set -euo pipefail
    cleanup() { echo "[collect] stopping teleop ..."; just teleop stop > /dev/null 2>&1 || true; }
    trap cleanup EXIT
    just teleop stop > /dev/null 2>&1 || true
    echo "[collect] starting teleop services ..."
    AUTO_CONFIRM=1 just teleop > /dev/null 2>&1
    echo "[collect] teleop ready"
    export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"
    # Use system Python 3.10 (rospy is built for 3.10 and hangs under 3.12).
    # Strip ROS2 Humble paths from PYTHONPATH so rospy's rosgraph_msgs isn't shadowed.
    export PYTHONPATH="$(echo "${PYTHONPATH:-}" | tr ':' '\n' | grep -v '/opt/ros/humble' | tr '\n' ':')"
    export PYTHONPATH="/usr/lib/python3/dist-packages:{{repo}}/third_party/A1_SDK/install/lib/python3/dist-packages:${PYTHONPATH}"
    PYTHONUNBUFFERED=1 python3 {{repo}}/third_party/lerobot/src/lerobot/scripts/lerobot_a1_collect.py \
        --experiment "{{experiment}}" \
        --data-root "{{repo}}/data/raw" \
        {{args}}

collect-drag *args:
    scripts/collect_data/a1_all_in_one.sh {{args}}

# Same as `just collect` but records only the RealSense (cam0); wrist cam1 disabled.
# For experiments isolating whether observation/trajectory miscorrespondence comes from cam1.
# Example: just collect1 pick_block
collect1 experiment *args:
    #!/usr/bin/env bash
    set -euo pipefail
    cleanup() { echo "[collect1] stopping teleop ..."; just teleop stop > /dev/null 2>&1 || true; }
    trap cleanup EXIT
    just teleop stop > /dev/null 2>&1 || true
    echo "[collect1] starting teleop services ..."
    AUTO_CONFIRM=1 just teleop > /dev/null 2>&1
    echo "[collect1] teleop ready"
    export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"
    PYTHONUNBUFFERED=1 {{vpy}} {{repo}}/third_party/lerobot/src/lerobot/scripts/lerobot_a1_collect.py \
        --experiment "{{experiment}}" \
        --data-root "{{repo}}/data/raw" \
        --disable-cam1 \
        {{args}}

# Convert raw episodes to LeRobot v2.1 dataset.
# Task prompt is read from data/raw/{experiment}/task.txt (created during collection).
# Example: just convert pick_block
# Example: just convert pick_block --overwrite
convert experiment *args:
    {{vpy}} {{repo}}/scripts/process_data/convert_episodes_to_lerobot_v21.py \
        --source-root "{{repo}}/data/raw/{{experiment}}" \
        --output-root "{{repo}}/data/processed/{{experiment}}" \
        {{args}}

# Same as `just convert` but for cam0-only episodes (recorded with `just collect1`).
# Example: just convert1 pick_block
convert1 experiment *args:
    {{vpy}} {{repo}}/scripts/process_data/convert_episodes_to_lerobot_v21.py \
        --source-root "{{repo}}/data/raw/{{experiment}}" \
        --output-root "{{repo}}/data/processed/{{experiment}}" \
        --disable-cam1 \
        {{args}}

test target="camera" *args:
    #!/usr/bin/env bash
    set -euo pipefail
    PY=$(scripts/collect_data/a1.sh which-python)
    case "{{target}}" in
        camera)     "$PY" scripts/collect_data/test_camera_connections.py --config configs/drag_replay.yaml --timeout-s 6.0 {{args}} ;;
        camera-raw) "$PY" scripts/collect_data/test_camera_connections.py {{args}} ;;
        *) echo "Usage: just test <camera|camera-raw> [args...]"; exit 1 ;;
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
        for cid in $(docker ps -q --filter "ancestor=a1-research/a1-noetic-arm64:local" 2>/dev/null); do
            docker kill "$cid" 2>/dev/null && echo "stopped container $cid"
        done
        exit 0
    fi

    run_dir="/tmp/lerobot-teleop"
    log_dir="$run_dir/logs"
    tail_pid_file="$run_dir/tail.pids"
    single_arm_serial="${SINGLE_ARM_SERIAL:-{{serial}}}"
    leader_port="${LEADER_PORT:-{{leader_port}}}"

    if [[ ! -x "{{repo}}/third_party/lerobot/.venv/bin/lerobot-a1-jointtracker-bridge" ]]; then
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

    a1_backend="{{repo}}/scripts/collect_data/a1_ros_backend.sh"

    start_service "roscore" \
        "$a1_backend roscore" \
        "roscore ready, press enter to proceed" \
        "" "another roscore/master is already running"

    start_service "single_arm_node" \
        "$a1_backend driver $single_arm_serial" \
        "single arm node ready, press enter to proceed"

    start_service "joint_tracker" \
        "$a1_backend tracker" \
        "joint tracker ready, press enter to proceed" \
        "model->njoints|start tracking|get target position"

    start_service "bridge" \
        "$a1_backend bridge $leader_port" \
        "bridge running, press enter to finish"

    echo "teleop services running in background."
    echo "logs: $log_dir"
    echo "stop with: just teleop stop"

# ── Inference ────────────────────────────────────────────────────────────────

policy policy_dir=checkpoint port="8001":
    PYTHONPATH="{{openpi}}/src:${PYTHONPATH:-}" {{uv}} run --project {{repo}} python {{repo}}/scripts/inference/serve_policy_a1.py --port {{port}} policy:checkpoint --policy.config {{model_config}} --policy.dir "{{policy_dir}}"

# Policy → ROS bridge: captures cameras directly, reads /joint_states_host (ROS),
# calls remote policy (WebSocket), publishes actions to /arm_joint_target_position (ROS).
# Requires: roscore + driver + tracker (via Docker), remote policy server
# Example: just policy-ros-bridge
# Example: just policy-ros-bridge 10.208.2.251 8001 "pick up the banana"
policy-ros-bridge host="127.0.0.1" port="8001" prompt="pick up the marker and place it into the red plate" *args:
    python3 {{repo}}/scripts/inference/policy_ros_bridge.py --host "{{host}}" --port "{{port}}" --prompt "{{prompt}}" {{args}}

# One-shot full inference pipeline (everything driven from py-a1):
#   1. rsync the inference-relevant files to the remote GPU box
#   2. start the policy server on the remote (skipped if already up; force-restart with A1_RESTART_SERVER=1)
#   3. wait for the websocket to come up, mirroring [server] log lines locally
#   4. run the policy-ros-bridge in the foreground
# Prereqs on this machine: ssh to {{remote}} works (key auth), driver + joint-tracker running.
# Env overrides:
#   A1_MODEL_CONFIG=<name>     train-config to serve (default: {{model_config}})
#   A1_RESTART_SERVER=1        kill the current remote server and relaunch
# Example: just infer
# Example: just infer "pick up the marker"
# Example: just infer "..." pengyue@10.208.2.251 8001 ~/checkpoints/pi05_a1_banana_red_plate/latest
infer prompt="pick up the banana and place it into the red plate" remote="pengyue@10.208.2.251" port="8001" remote_ckpt="~/checkpoints/pi05_a1_banana_red_plate/latest":
    #!/usr/bin/env bash
    set -eu -o pipefail

    config="${A1_MODEL_CONFIG:-{{model_config}}}"
    host="{{remote}}"; host="${host##*@}"

    server_up() {
        (exec 3<>"/dev/tcp/${host}/{{port}}") 2>/dev/null && { exec 3<&-; return 0; } || return 1
    }

    echo "[infer 1/4] rsync code → {{remote}}:~/A1-Research/"
    rsync -az \
        {{repo}}/a1/training/config.py     {{remote}}:~/A1-Research/a1/training/config.py
    rsync -az \
        {{repo}}/a1/training/a1_policy.py  {{remote}}:~/A1-Research/a1/training/a1_policy.py
    rsync -az \
        {{repo}}/scripts/inference/serve_policy_a1.py \
        {{remote}}:~/A1-Research/scripts/inference/serve_policy_a1.py
    rsync -az \
        {{repo}}/Justfile                  {{remote}}:~/A1-Research/Justfile

    need_start=1
    if server_up; then
        if [[ "${A1_RESTART_SERVER:-0}" == "1" ]]; then
            echo "[infer 2/4] A1_RESTART_SERVER=1 — killing existing server on ${host}:{{port}}"
            ssh {{remote}} "pkill -f 'serve_policy_a1.py.*--port {{port}}' 2>/dev/null || true" || true
            sleep 2
        else
            echo "[infer 2/4] policy server already up on ${host}:{{port}} — reusing (set A1_RESTART_SERVER=1 to force restart)"
            need_start=0
        fi
    fi

    tail_pid=""
    cleanup() { [[ -n "${tail_pid:-}" ]] && kill "${tail_pid}" 2>/dev/null || true; }
    trap cleanup EXIT INT TERM

    if [[ "${need_start}" == "1" ]]; then
        echo "[infer 2/4] starting remote policy server: config=${config} ckpt={{remote_ckpt}} port={{port}}"
        remote_cmd="cd ~/A1-Research && A1_MODEL_CONFIG=${config} nohup just policy {{remote_ckpt}} {{port}} </dev/null >/tmp/a1_policy_server.log 2>&1 &"
        ssh {{remote}} "bash -lc '${remote_cmd}'"

        ssh -o BatchMode=yes {{remote}} "tail -n 0 -F /tmp/a1_policy_server.log 2>/dev/null" 2>/dev/null \
            | sed -u 's/^/[server] /' &
        tail_pid=$!

        echo "[infer 3/4] waiting for ws://${host}:{{port}} (up to 240s)"
        deadline=$(( $(date +%s) + 240 ))
        while (( $(date +%s) < deadline )); do
            if server_up; then
                echo "[infer 3/4] policy server is up"
                break
            fi
            sleep 2
        done
        if ! server_up; then
            echo "[infer ERROR] server did not come up in time. See [server] log lines above; full log: ssh {{remote}} cat /tmp/a1_policy_server.log"
            exit 1
        fi
    else
        echo "[infer 3/4] server already up — skipping wait"
    fi

    echo "[infer 4/4] launching policy → ROS bridge (Ctrl-C to stop)"
    python3 {{repo}}/scripts/inference/policy_ros_bridge.py \
        --host "${host}" --port "{{port}}" --prompt "{{prompt}}"

# Teacher-forcing execution via ROS — same receiver pipeline as just teleop (no ZMQ).
# Feeds GT observations from recorded demo to policy, sends outputs to /arm_joint_target_position.
# Requires: just launch roscore, just launch driver, just joint-tracker, just policy
# Example: just tf-exec-ros demo_0
# Example: just tf-exec-ros demo_0 -- --no-step-mode
# Example: just tf-exec-ros demo_0 -- --dry-run
tf-exec-ros demo="" processed_root="{{repo}}/data/processed_data/pick_twice" *args:
    PYTHONPATH="{{openpi}}/packages/openpi-client/src:${PYTHONPATH:-}" {{uv}} run --project {{repo}} python {{repo}}/scripts/inference/tf_exec_ros.py --processed-root "{{processed_root}}" $([ -n "{{demo}}" ] && echo "--demo {{demo}}") {{args}}

# Teacher-forcing execution via ROS on a LeRobot v2.1 dataset.
# Requires: just launch roscore, just launch driver, just joint-tracker, just policy
# Example: just tf-exec-ros-lerobot /path/to/dataset "pick up the marker" -- --episode 0
# Example: just tf-exec-ros-lerobot /path/to/dataset "pick up the marker" -- --episode 0 --dry-run
tf-exec-ros-lerobot lerobot_root prompt="pick up the marker and place it into the red plate" *args:
    PYTHONPATH="{{openpi}}/packages/openpi-client/src:${PYTHONPATH:-}" {{uv}} run --project {{repo}} python {{repo}}/scripts/inference/tf_exec_ros.py --lerobot-root "{{lerobot_root}}" --prompt "{{prompt}}" {{args}}

# Teacher-forcing: offline policy inference on training images → trajectory.json + trajectory.html
# Requires: just policy (WebSocket server) running first.
# Example: just teacher-forcing
# Example: just teacher-forcing demo_0
# Example: just teacher-forcing demo_0 -- --max-steps 100
teacher-forcing demo="" processed_root="{{repo}}/data/processed_data/pick_twice" *args:
    PYTHONPATH="{{openpi}}/packages/openpi-client/src:${PYTHONPATH:-}" {{uv}} run --project {{repo}} python {{repo}}/scripts/inference/teacher_forcing_infer.py --processed-root "{{processed_root}}" $([ -n "{{demo}}" ] && echo "--demo {{demo}}") {{args}}

# Teacher-forcing on a LeRobot v2.1 dataset (parquet + jpg, no mp4 videos).
# Requires: just policy (WebSocket server) running first.
# Example: just tf-lerobot data/a1_lerobot
# Example: just tf-lerobot data/a1_lerobot "pick up the block" -- --episode 3
tf-lerobot *args:
    {{uv}} run --project {{repo}} python {{repo}}/scripts/inference/teacher_forcing_infer.py --lerobot-root "{{repo}}/data/processed/pick_up_the_banana_and_place_it_into_the_red_plate" --prompt "pick up the banana and place it into the red plate" {{args}}

# Teacher-forcing execution: run policy on recorded demo and send actions to A1 arm via ROS.
# Requires: roscore + driver + tracker (via Docker), remote policy server
# Add --dry-run to infer without sending to robot.
# Example: just tf-exec demo_0
# Example: just tf-exec demo_0 -- --exec-rate 5 --dry-run
tf-exec demo="" processed_root="{{repo}}/data/processed_data/pick_twice" *args:
    python3 {{repo}}/scripts/inference/teacher_forcing_exec.py --processed-root "{{processed_root}}" $([ -n "{{demo}}" ] && echo "--demo {{demo}}") {{args}}

# Teacher-forcing execution on LeRobot v2.1 dataset — sends actions to A1 arm via ROS.
# Requires: roscore + driver + tracker, remote policy server
# Example: just tf-exec-lerobot --host 10.208.2.251 --port 8001 --episode 0
# Example: just tf-exec-lerobot --host 10.208.2.251 --port 8001 --episode 0 --dry-run
tf-exec-lerobot *args:
    #!/usr/bin/env bash
    set -euo pipefail
    export PYTHONPATH="$(echo "${PYTHONPATH:-}" | tr ':' '\n' | grep -v '/opt/ros/humble' | tr '\n' ':')"
    python3 {{repo}}/scripts/inference/teacher_forcing_exec.py --lerobot-root "{{repo}}/data/processed/pick_up_the_banana_and_place_it_into_the_red_plate" --prompt "pick up the banana and place it into the red plate" {{args}}

# Open-loop rollout eval on processed data → per-demo trajectory.json + trajectory.html
# Example: just openloop-rollout --policy-dir checkpoints/my_run/4999
# Example: just openloop-rollout --policy-dir checkpoints/my_run/4999 --max-demos 1 --max-steps-per-demo 100
openloop-rollout policy_dir="" *args:
    if [ -z "{{policy_dir}}" ]; then echo "Usage: just openloop-rollout <policy_dir> [extra args...]"; exit 1; fi
    PYTHONPATH="{{openpi}}/src:{{repo}}:${PYTHONPATH:-}" {{uv}} run --project {{repo}} python {{repo}}/scripts/inference/openloop_rollout.py --policy-dir "{{policy_dir}}" {{args}}


# ── Debug ────────────────────────────────────────────────────────────────────

debug target="camera" output_dir="{{repo}}/data/debug/model_input_frames" every_n="20" max_per_cam="300" duration_s="30" *args:
    case "{{target}}" in \
        camera) {{uv}} run --project {{repo}} python {{repo}}/scripts/inference/dump_model_input_images.py --output-dir "{{output_dir}}" --every-n "{{every_n}}" --max-per-cam "{{max_per_cam}}" --duration-s "{{duration_s}}" {{args}} ;; \
        *) echo "Usage: just debug <camera> [output_dir] [every_n] [max_per_cam] [duration_s] [extra args...]"; exit 1 ;; \
    esac
