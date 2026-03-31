set shell := ["bash", "-eu", "-o", "pipefail", "-c"]
set quiet := true

# ── Configuration ────────────────────────────────────────────────────────────
# Paths
repo := justfile_directory()
uv   := env_var_or_default("UV_BIN", "uv")
a1_ros_backend := repo + "/scripts/collect_data/a1_ros_backend.sh"
a1_runtime_pythonpath := repo + "/third_party/A1_SDK_runtime/install/lib/python3/dist-packages:/usr/lib/python3/dist-packages"

# Model
checkpoint   := env_var_or_default("CHECKPOINT_DIR", repo + "/checkpoints/19999")
openpi       := env_var_or_default("OPENPI_ROOT", repo + "/../TFP")
model_config := "pi05_a1_v21_lora_5k"
# Teleoperation
teleop_leader_port := "/dev/ttyACM2"

default:
    @just --list

# ── Environment ──────────────────────────────────────────────────────────────

doctor:
    scripts/collect_data/dragdatacoach.sh doctor

which-python:
    scripts/collect_data/dragdatacoach.sh which-python

which-camera-python:
    scripts/collect_data/dragdatacoach.sh which-camera-python

setup-main:
    {{uv}} venv --clear --python 3.12 {{repo}}/.venv
    {{uv}} pip install --python {{repo}}/.venv/bin/python -e {{repo}} --no-deps
    {{uv}} pip install --python {{repo}}/.venv/bin/python hydra-core omegaconf pyzmq opencv-python-headless numpy scipy pandas rospkg catkin-pkg pyparsing pyyaml pyserial pillow
    @echo "main env ready: {{repo}}/.venv"

setup-camera:
    {{uv}} venv --clear --python 3.10 {{repo}}/.venv-camera
    {{uv}} pip install --python {{repo}}/.venv-camera/bin/python hydra-core omegaconf pyzmq opencv-python-headless numpy scipy pandas rospkg catkin-pkg pyparsing pyyaml pyserial pillow "pyrealsense2==2.56.5.9235"
    @echo "camera env ready: {{repo}}/.venv-camera"

setup-all:
    just setup-main
    just setup-camera
    just setup-teleop

# ── Commands ─────────────────────────────────────────────────────────────────

launch target="driver" serial="/dev/a1":
    case "{{target}}" in \
        roscore) \
            {{a1_ros_backend}} roscore ;; \
        camera-server) \
            DATACOACH_PYTHON=$(scripts/collect_data/dragdatacoach.sh which-camera-python) \
            DATACOACH_ROS_PYTHONPATH=$(scripts/collect_data/dragdatacoach.sh which-ros-pythonpath "$DATACOACH_PYTHON") \
            PYTHONPATH="$DATACOACH_ROS_PYTHONPATH:${PYTHONPATH:-}" \
            "$DATACOACH_PYTHON" scripts/collect_data/run_data_services.py service_mode=live "a1_server.components=[]" ;; \
        joint-tracker) \
            {{a1_ros_backend}} tracker ;; \
        ee-tracker) \
            {{a1_ros_backend}} ee-tracker ;; \
        a1-server) \
            DATACOACH_PYTHON=$(scripts/collect_data/dragdatacoach.sh which-python) \
            DATACOACH_ROS_PYTHONPATH=$(scripts/collect_data/dragdatacoach.sh which-ros-pythonpath "$DATACOACH_PYTHON") \
            PYTHONPATH="$DATACOACH_ROS_PYTHONPATH:${PYTHONPATH:-}" \
            "$DATACOACH_PYTHON" scripts/collect_data/run_a1_server.py "a1_server.components=[ros_subscriber,policy_action_subscriber]" ;; \
        driver) \
            {{a1_ros_backend}} driver "{{serial}}" ;; \
        ee-record) \
            {{a1_ros_backend}} ee-record "{{serial}}" ;; \
        tracker) \
            {{a1_ros_backend}} tracker ;; \
        *) echo "Usage: just launch <roscore|camera-server|joint-tracker|ee-tracker|a1-server|driver|ee-record|tracker> [serial]"; exit 1 ;; \
    esac

joint-tracker:
    just launch joint-tracker

# Relay /arm_joint_target_position (JointState) → /arm_joint_command_host (arm_control)
# Required for inference when using /arm_joint_target_position as the control topic.
joint-relay:
    DATACOACH_PYTHON=$(scripts/collect_data/dragdatacoach.sh which-python) \
    DATACOACH_ROS_PYTHONPATH=$(scripts/collect_data/dragdatacoach.sh which-ros-pythonpath "$DATACOACH_PYTHON") \
    PYTHONPATH="$DATACOACH_ROS_PYTHONPATH:${PYTHONPATH:-}" \
    "$DATACOACH_PYTHON" {{repo}}/scripts/inference/joint_target_relay.py

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
            DATACOACH_PYTHON=$(scripts/collect_data/dragdatacoach.sh which-camera-python)
            DATACOACH_ROS_PYTHONPATH=$(scripts/collect_data/dragdatacoach.sh which-ros-pythonpath "$DATACOACH_PYTHON")
            trap '' INT; set +e
            PYTHONPATH="$DATACOACH_ROS_PYTHONPATH:${PYTHONPATH:-}" \
            "$DATACOACH_PYTHON" \
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

drag-collect *args:
    scripts/collect_data/dragdatacoach_all_in_one.sh {{args}}

teleop-stop:
    just teleop stop

test target="camera" *args:
    #!/usr/bin/env bash
    set -euo pipefail
    PY=$(scripts/collect_data/dragdatacoach.sh which-camera-python)
    case "{{target}}" in
        camera)     "$PY" scripts/collect_data/test_camera_connections.py --config configs/drag_replay.yaml --timeout-s 6.0 {{args}} ;;
        camera-raw) "$PY" scripts/collect_data/test_camera_connections.py {{args}} ;;
        *) echo "Usage: just test <camera|camera-raw> [args...]"; exit 1 ;;
    esac

print target="joints" count="0" unit="deg":
    case "{{target}}" in \
        joints) \
            DATACOACH_PYTHON=$(scripts/collect_data/dragdatacoach.sh which-python) \
            DATACOACH_ROS_PYTHONPATH=$(scripts/collect_data/dragdatacoach.sh which-ros-pythonpath "$DATACOACH_PYTHON") \
            PYTHONPATH="$DATACOACH_ROS_PYTHONPATH:${PYTHONPATH:-}" \
            "$DATACOACH_PYTHON" scripts/collect_data/print_joint_angles.py --count "{{count}}" --unit "{{unit}}" ;; \
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
    #!/usr/bin/env bash
    set -euo pipefail
    port="${LEADER_PORT:-{{port}}}"
    resolve_dev() { readlink -f "$1" 2>/dev/null || echo "$1"; }
    arm_real="$(resolve_dev /dev/a1)"
    if [[ ! -e "$port" ]]; then
        for candidate in /dev/ttyACM* /dev/ttyUSB*; do
            [[ -e "$candidate" ]] || continue
            candidate_real="$(resolve_dev "$candidate")"
            [[ "$candidate_real" == "$arm_real" ]] && continue
            port="$candidate"
            break
        done
    fi
    if [[ ! -e "$port" ]]; then
        echo "leader port not found: ${LEADER_PORT:-{{port}}}"
        ls -l /dev/ttyACM* /dev/ttyUSB* /dev/a1 2>/dev/null || true
        exit 1
    fi
    echo "using leader port: $port"
    {{repo}}/third_party/lerobot/.venv/bin/lerobot-calibrate \
        --teleop.type=so100_leader \
        --teleop.port="$port" \
        --teleop.id="{{id}}"

# One-time setup: create third_party/lerobot/.venv (Python 3.12) and install lerobot[feetech].
# Run this once before the first `just teleop`.
setup-teleop:
    {{uv}} venv --clear --python 3.12 {{repo}}/third_party/lerobot/.venv
    {{uv}} pip install --python {{repo}}/third_party/lerobot/.venv/bin/python -e "{{repo}}/third_party/lerobot[feetech]"
    {{uv}} pip install --python {{repo}}/third_party/lerobot/.venv/bin/python pyparsing pytz
    @echo "teleop env ready: {{repo}}/third_party/lerobot/.venv"

# SO leader → A1 jointTracker teleoperation.
# Starts: roscore → single_arm_node → jointTrackerdemo → SO leader bridge.
# Backend selection:
#   auto   -> host ROS Noetic on Ubuntu 20.04 when available, otherwise Docker
#   docker -> force Docker Noetic
#   host   -> force host ROS Noetic
# Run `just setup-teleop` once before first use.
# Example: just teleop
# Example: just teleop stop
# Example: LEADER_PORT=/dev/ttyACM1 just teleop
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
    resolve_dev() { readlink -f "$1" 2>/dev/null || echo "$1"; }

    if [[ ! -x "$bridge_bin" ]]; then
        echo "teleop env not set up. run: just setup-teleop"
        exit 1
    fi

    if [[ ! -e "$single_arm_serial" ]]; then
        echo "single arm serial not found: $single_arm_serial"
        ls -l /dev/ttyACM* /dev/a1 2>/dev/null || true
        exit 1
    fi

    single_arm_real="$(resolve_dev "$single_arm_serial")"
    leader_real="$(resolve_dev "$leader_port")"
    if [[ ! -e "$leader_port" || "$leader_real" == "$single_arm_real" ]]; then
        for candidate in /dev/ttyACM* /dev/ttyUSB*; do
            [[ -e "$candidate" ]] || continue
            candidate_real="$(resolve_dev "$candidate")"
            [[ "$candidate_real" == "$single_arm_real" ]] && continue
            leader_port="$candidate"
            leader_real="$candidate_real"
            break
        done
    fi

    if [[ ! -e "$leader_port" ]]; then
        echo "leader port not found: $leader_port"
        ls -l /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || true
        exit 1
    fi

    echo "using single arm serial: $single_arm_serial"
    echo "using leader port: $leader_port"

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
        "{{a1_ros_backend}} roscore" \
        "roscore ready, press enter to proceed" \
        "" "another roscore/master is already running"

    start_service "single_arm_node" \
        "{{a1_ros_backend}} driver $single_arm_serial" \
        "single arm node ready, press enter to proceed"

    start_service "joint_tracker" \
        "{{a1_ros_backend}} tracker" \
        "joint tracker ready, press enter to proceed" \
        "model->njoints|start tracking|get target position"

    start_service "bridge" \
        "cd {{repo}} && BRIDGE_PYTHONPATH=\$(scripts/collect_data/dragdatacoach.sh which-ros-pythonpath {{repo}}/third_party/lerobot/.venv/bin/python) && PYTHONPATH=\"\$BRIDGE_PYTHONPATH:\${PYTHONPATH:-}\" $bridge_bin --leader-port $leader_port --leader-id my_leader --gripper-min-stroke-mm 0 --gripper-max-stroke-mm 200" \
        "bridge running, press enter to finish"

    echo "teleop services running in background."
    echo "logs: $log_dir"
    echo "stop with: just teleop stop"

# ── Inference ────────────────────────────────────────────────────────────────

# Live inference: real cameras + current joint state → policy → robot.
# Starts: roscore → single_arm_node → jointTrackerdemo → camera_server → policy server → policy_ros_bridge.
# Example: just infer
# Example: just infer "pick up the marker and place it into the red plate"
# Example: just infer "pick up the marker" -- --exec-rate 50
infer prompt="pick up the marker and place it into the red plate" *args:
    #!/usr/bin/env bash
    set -euo pipefail

    eric_sdk="/home/eric/A1_SDK/install"
    eric_python="/home/eric/openpi/.venv/bin/python"
    checkpoint_dir="/home/eric/19999"
    port="8001"
    serial="${SINGLE_ARM_SERIAL:-/dev/a1}"
    run_dir="/tmp/lerobot-infer"
    log_dir="$run_dir/logs"
    tail_pid_file="$run_dir/tail.pids"
    mkdir -p "$log_dir"
    : > "$tail_pid_file"

    if [[ ! -e "$serial" ]]; then
        echo "[infer] serial not found: $serial"
        ls -l /dev/ttyACM* /dev/a1 2>/dev/null || true
        exit 1
    fi

    cleanup() {
        echo "[infer] shutting down..."
        while IFS= read -r pid; do kill "$pid" 2>/dev/null || true; done < "$tail_pid_file"
        for f in "$run_dir"/*.pid; do
            [[ -f "$f" ]] || continue; kill "$(cat "$f")" 2>/dev/null || true; rm -f "$f"
        done
    }
    trap cleanup EXIT

    start_service() {
        local name="$1" cmd="$2" prompt="$3"
        local suppress="${4:-}" ok_regex="${5:-}"
        local log="$log_dir/$name.log" pid_file="$run_dir/$name.pid"
        echo "[$name] starting..."; : > "$log"
        nohup bash -lc "$cmd" >> "$log" 2>&1 &
        local pid=$!; echo "$pid" > "$pid_file"
        if [[ -n "$suppress" ]]; then
            tail -n 0 -F "$log" | grep -Eav --line-buffered "$suppress" | sed -u "s/^/[$name] /" &
        else
            tail -n 0 -F "$log" | sed -u "s/^/[$name] /" &
        fi
        echo "$!" >> "$tail_pid_file"
        sleep 1
        if ! kill -0 "$pid" 2>/dev/null; then
            if [[ -n "$ok_regex" ]] && grep -Eq "$ok_regex" "$log"; then
                echo "[$name] existing instance detected, reusing."; rm -f "$pid_file"
            else
                echo "[$name] failed to start. log: $log"; exit 1
            fi
        fi
        read -r -p "$prompt"
    }

    start_service "roscore" \
        "cd $eric_sdk && source setup.bash && roscore" \
        "[roscore] ready — press enter: " \
        "" "another roscore/master is already running"

    start_service "single_arm_node" \
        "cd $eric_sdk && source setup.bash && roslaunch signal_arm single_arm_node.launch single_arm_serial_port_path:=$serial" \
        "[single_arm_node] ready — press enter: "

    start_service "joint_tracker" \
        "cd $eric_sdk && source setup.bash && roslaunch mobiman jointTrackerdemo.launch" \
        "[joint_tracker] ready — press enter: " \
        "model->njoints|start tracking|get target position"

    start_service "camera_server" \
        "cd {{repo}} && {{uv}} run --project {{repo}} python scripts/collect_data/run_data_services.py service_mode=live 'a1_server.components=[]'" \
        "[camera_server] ready — press enter: "

    echo "[infer] starting policy server (checkpoint=$checkpoint_dir) ..."
    policy_log="$log_dir/policy.log"
    PYTHONPATH="{{openpi}}/src:${PYTHONPATH:-}" \
        "$eric_python" {{repo}}/scripts/inference/serve_policy_a1.py \
        --port "$port" policy:checkpoint \
        --policy.config {{model_config}} \
        --policy.dir "$checkpoint_dir" \
        > "$policy_log" 2>&1 &
    policy_pid=$!
    echo "$policy_pid" > "$run_dir/policy.pid"
    tail -n 0 -F "$policy_log" | sed -u "s/^/[policy] /" &
    echo "$!" >> "$tail_pid_file"

    echo "[infer] waiting for policy server on port $port ..."
    for i in $(seq 1 90); do
        ss -tlnp 2>/dev/null | grep -q ":$port " && { echo "[infer] policy ready."; break; }
        kill -0 "$policy_pid" 2>/dev/null || { echo "[infer] policy crashed. log: $policy_log"; exit 1; }
        sleep 2
    done

    echo "[infer] running live bridge ..."
    PYTHONPATH="$eric_sdk/lib/python3/dist-packages:/opt/ros/noetic/lib/python3/dist-packages:{{openpi}}/packages/openpi-client/src:${PYTHONPATH:-}" \
        "$eric_python" {{repo}}/scripts/inference/policy_ros_bridge.py \
        --host 127.0.0.1 --port "$port" --prompt "{{prompt}}" {{args}}

# Teacher-forcing inference: GT observations from dataset → policy → robot.
# Starts: roscore → single_arm_node → jointTrackerdemo → policy server → lerobot_policy_bridge.
# Example: just teacher-infer -- --episode 0 --num-chunks 10
# Example: just teacher-infer -- --episode 0 --chunk-size 10 --no-step-mode
teacher-infer *args:
    #!/usr/bin/env bash
    set -euo pipefail

    eric_sdk="/home/eric/A1_SDK/install"
    eric_python="/home/eric/openpi/.venv/bin/python"
    checkpoint_dir="/home/eric/19999"
    port="8001"
    serial="${SINGLE_ARM_SERIAL:-/dev/a1}"
    run_dir="/tmp/lerobot-infer"
    log_dir="$run_dir/logs"
    tail_pid_file="$run_dir/tail.pids"
    mkdir -p "$log_dir"
    : > "$tail_pid_file"

    if [[ ! -e "$serial" ]]; then
        echo "[teacher-infer] serial not found: $serial"
        ls -l /dev/ttyACM* /dev/a1 2>/dev/null || true
        exit 1
    fi

    cleanup() {
        echo "[teacher-infer] shutting down..."
        while IFS= read -r pid; do kill "$pid" 2>/dev/null || true; done < "$tail_pid_file"
        for f in "$run_dir"/*.pid; do
            [[ -f "$f" ]] || continue; kill "$(cat "$f")" 2>/dev/null || true; rm -f "$f"
        done
    }
    trap cleanup EXIT

    start_service() {
        local name="$1" cmd="$2" prompt="$3"
        local suppress="${4:-}" ok_regex="${5:-}"
        local log="$log_dir/$name.log" pid_file="$run_dir/$name.pid"
        echo "[$name] starting..."; : > "$log"
        nohup bash -lc "$cmd" >> "$log" 2>&1 &
        local pid=$!; echo "$pid" > "$pid_file"
        if [[ -n "$suppress" ]]; then
            tail -n 0 -F "$log" | grep -Eav --line-buffered "$suppress" | sed -u "s/^/[$name] /" &
        else
            tail -n 0 -F "$log" | sed -u "s/^/[$name] /" &
        fi
        echo "$!" >> "$tail_pid_file"
        sleep 1
        if ! kill -0 "$pid" 2>/dev/null; then
            if [[ -n "$ok_regex" ]] && grep -Eq "$ok_regex" "$log"; then
                echo "[$name] existing instance detected, reusing."; rm -f "$pid_file"
            else
                echo "[$name] failed to start. log: $log"; exit 1
            fi
        fi
        read -r -p "$prompt"
    }

    start_service "roscore" \
        "cd $eric_sdk && source setup.bash && roscore" \
        "[roscore] ready — press enter: " \
        "" "another roscore/master is already running"

    start_service "single_arm_node" \
        "cd $eric_sdk && source setup.bash && roslaunch signal_arm single_arm_node.launch single_arm_serial_port_path:=$serial" \
        "[single_arm_node] ready — press enter: "

    start_service "joint_tracker" \
        "cd $eric_sdk && source setup.bash && roslaunch mobiman jointTrackerdemo.launch" \
        "[joint_tracker] ready — press enter: " \
        "model->njoints|start tracking|get target position"

    echo "[teacher-infer] starting policy server (checkpoint=$checkpoint_dir) ..."
    policy_log="$log_dir/policy.log"
    PYTHONPATH="{{openpi}}/src:${PYTHONPATH:-}" \
        "$eric_python" {{repo}}/scripts/inference/serve_policy_a1.py \
        --port "$port" policy:checkpoint \
        --policy.config {{model_config}} \
        --policy.dir "$checkpoint_dir" \
        > "$policy_log" 2>&1 &
    policy_pid=$!
    echo "$policy_pid" > "$run_dir/policy.pid"
    tail -n 0 -F "$policy_log" | sed -u "s/^/[policy] /" &
    echo "$!" >> "$tail_pid_file"

    echo "[teacher-infer] waiting for policy server on port $port ..."
    for i in $(seq 1 90); do
        ss -tlnp 2>/dev/null | grep -q ":$port " && { echo "[teacher-infer] policy ready."; break; }
        kill -0 "$policy_pid" 2>/dev/null || { echo "[teacher-infer] policy crashed. log: $policy_log"; exit 1; }
        sleep 2
    done

    echo "[teacher-infer] running bridge ..."
    "$eric_python" {{repo}}/scripts/inference/lerobot_policy_bridge.py {{args}}

# Teacher-forcing policy bridge only (policy server must already be running).
# Example: just lerobot-policy-bridge -- --episode 0
# Example: just lerobot-policy-bridge -- --episode 0 --num-chunks 1
lerobot-policy-bridge *args:
    /home/eric/openpi/.venv/bin/python {{repo}}/scripts/inference/lerobot_policy_bridge.py {{args}}

policy policy_dir=checkpoint port="8001":
    PYTHONPATH="{{openpi}}/src:${PYTHONPATH:-}" {{uv}} run --project {{repo}} python {{repo}}/scripts/inference/serve_policy_a1.py --port {{port}} policy:checkpoint --policy.config {{model_config}} --policy.dir "{{policy_dir}}"

# Policy → ROS bridge: same pipeline as just teleop but policy replaces the SO leader arm.
# Reads live cameras (ZMQ) + /joint_states_host (ROS), calls policy, publishes to /arm_joint_target_position.
# Requires: just launch roscore, just launch driver, just joint-tracker, just launch camera-server, just policy
# Example: just policy-ros-bridge
# Example: just policy-ros-bridge 127.0.0.1 8001 "pick up the marker"
policy-ros-bridge host="127.0.0.1" port="8001" prompt="pick up the marker and place it into the red plate" *args:
    PYTHONPATH="{{openpi}}/packages/openpi-client/src:${PYTHONPATH:-}" {{uv}} run --project {{repo}} python {{repo}}/scripts/inference/policy_ros_bridge.py --host "{{host}}" --port "{{port}}" --prompt "{{prompt}}" {{args}}

# ZMQ↔WebSocket bridge: reads state+cameras from ZMQ, calls WebSocket policy server, publishes actions to ZMQ
# Run this alongside `just policy` so the A1 server receives joint actions over ZMQ.
# Example: just zmq-bridge
# Example: just zmq-bridge nyushrobo5090 8001 "pick up the cup"
zmq-bridge host="127.0.0.1" port="8001" prompt="pick up the marker and place it into the red plate" *args:
    PYTHONPATH="{{openpi}}/packages/openpi-client/src:${PYTHONPATH:-}" {{uv}} run --project {{repo}} python {{repo}}/scripts/inference/zmq_ws_bridge.py --host "{{host}}" --port "{{port}}" --prompt "{{prompt}}" {{args}}

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
# Example: just tf-lerobot /home/eric/lerobot/data/a1_v21_old
# Example: just tf-lerobot /home/eric/lerobot/data/a1_v21_old "pick up the marker and place it into the red plate" -- --episode 3
# Example: just tf-lerobot /home/eric/lerobot/data/a1_v21_old "pick up the marker and place it into the red plate" -- --episode 3 --max-steps 200
tf-lerobot lerobot_root prompt="pick up the marker and place it into the red plate" *args:
    PYTHONPATH="{{openpi}}/packages/openpi-client/src:${PYTHONPATH:-}" {{uv}} run --project {{repo}} python {{repo}}/scripts/inference/teacher_forcing_infer.py --lerobot-root "{{lerobot_root}}" --prompt "{{prompt}}" {{args}}

# Teacher-forcing execution: run policy on recorded demo and send joint targets to A1 arm via ZMQ.
# Requires: just policy + just launch a1-server + just joint-relay
# Add --dry-run to infer without sending to robot.
# Example: just tf-exec demo_0
# Example: just tf-exec demo_0 -- --exec-rate 5 --dry-run
# Example: just tf-exec demo_0 -- --max-steps 50
tf-exec demo="" processed_root="{{repo}}/data/processed_data/pick_twice" *args:
    PYTHONPATH="{{openpi}}/packages/openpi-client/src:${PYTHONPATH:-}" {{uv}} run --project {{repo}} python {{repo}}/scripts/inference/teacher_forcing_exec.py --processed-root "{{processed_root}}" $([ -n "{{demo}}" ] && echo "--demo {{demo}}") {{args}}

# Teacher-forcing execution on a LeRobot v2.1 dataset — sends joint targets to A1 arm via ZMQ.
# Requires: just policy + just launch a1-server + just joint-relay
# Example: just tf-exec-lerobot /path/to/dataset "swap the marker and block" -- --episode 0
# Example: just tf-exec-lerobot /path/to/dataset "swap the marker and block" -- --episode 0 --dry-run
tf-exec-lerobot lerobot_root prompt="pick up the marker and place it into the red plate" *args:
    PYTHONPATH="{{openpi}}/packages/openpi-client/src:${PYTHONPATH:-}" {{uv}} run --project {{repo}} python {{repo}}/scripts/inference/teacher_forcing_exec.py --lerobot-root "{{lerobot_root}}" --prompt "{{prompt}}" {{args}}

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
