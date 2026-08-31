set shell := ["bash", "-e", "-o", "pipefail", "-c"]
set quiet := true

uv   := env("UV_BIN", "uv")
repo := justfile_directory()
vpy  := repo + "/.venv/bin/python"

default:
    @just --list

# ── Setup ────────────────────────────────────────────────────────────────────

setup:
    #!/usr/bin/env bash
    set -euo pipefail
    export UV_DEFAULT_INDEX="https://pypi.org/simple"
    export UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-cu128}"
    {{uv}} python install 3.12
    {{uv}} sync --frozen --python 3.12
    source {{repo}}/scripts/runtime/a1_console.sh
    a1_success "Main environment ready: {{repo}}/.venv"

udev:
    scripts/runtime/install_a1_udev.sh

# ── Local Checks ─────────────────────────────────────────────────────────────

check:
    {{vpy}} -m galaxea_a1_runtime.cli doctor --repo-root "{{repo}}"
    find {{repo}}/scripts -type f -name '*.sh' -print0 | xargs -0 -r -n1 bash -n
    {{vpy}} -m ruff check {{repo}}/galaxea_a1_runtime {{repo}}/scripts {{repo}}/tests
    {{vpy}} -m ruff format --check {{repo}}/galaxea_a1_runtime {{repo}}/scripts {{repo}}/tests
    just test

# List tracked operator configurations.
operator-configs:
    {{vpy}} -m galaxea_a1_runtime.cli configs --repo-root "{{repo}}"

# Open the Operator Panel.
panel:
    {{vpy}} -m galaxea_a1_runtime.cli panel --repo-root "{{repo}}"

# Render the tracked read-only Foxglove layout from System config.
foxglove-layout:
    {{vpy}} {{repo}}/scripts/runtime/render_foxglove_layout.py

ros-python-check:
    #!/usr/bin/env bash
    set -euo pipefail
    source {{repo}}/scripts/runtime/a1_config.sh
    source {{repo}}/scripts/runtime/a1_services.sh
    a1_load_shell_config env \
      PYTHONPATH="{{repo}}:${PYTHONPATH:-}" \
      {{vpy}} -m galaxea_a1_runtime.configuration.system \
      --repo-root "{{repo}}" --shell
    docker image inspect "${IMAGE}" >/dev/null
    docker run --rm --network none \
      -e A1_SDK_ROOT=/workspace/third_party/A1_SDK \
      -e "PYTHONPATH=${A1_CONTAINER_PYTHONPATH}" \
      -v "{{repo}}:/workspace:ro" \
      "${IMAGE}" \
      /workspace/scripts/runtime/a1_ros_python_check.py \
      --config /workspace/configs/system/a1.toml

test:
    {{vpy}} -m pytest -q {{repo}}/tests

models:
    {{vpy}} {{repo}}/scripts/models/model_store.py doctor --repo-root "{{repo}}"

model-fetch config:
    {{vpy}} {{repo}}/scripts/models/model_store.py fetch \
        --repo-root "{{repo}}" "{{config}}"

model-verify config:
    {{vpy}} {{repo}}/scripts/models/model_store.py verify \
        --repo-root "{{repo}}" "{{config}}"

# ── Hardware Workflow ────────────────────────────────────────────────────────

# Check configured serial devices and cameras without moving the robot.
hardware *args:
    {{vpy}} -m galaxea_a1_runtime.cli hardware --repo-root "{{repo}}" {{args}}

rosbag *args:
    {{repo}}/scripts/apps/recording/a1_rosbag.sh {{args}}

camera-check *args:
    #!/usr/bin/env bash
    set -euo pipefail
    scripts/apps/cameras/a1_camera_web_runtime.sh stop
    trap 'scripts/apps/cameras/a1_camera_web_runtime.sh start' EXIT
    {{vpy}} scripts/apps/cameras/a1_camera_diagnostics.py {{args}}

# Start, stop, inspect, or read logs from the persistent cameras.
cameras action="start":
    scripts/apps/cameras/a1_camera_observability_runtime.sh "{{action}}"

# Start a read-only Foxglove endpoint without starting the A1 driver or relay.
foxglove action="start":
    scripts/runtime/a1_observability_runtime.sh "{{action}}"

eef-test:
    scripts/runtime/a1_joint_runtime.sh services
    scripts/runtime/a1_joint_runtime.sh eef-nudge --execute

# Reset and collect episodes into one experiment.
collect experiment task:
    scripts/apps/teleop/a1_teleop_runtime.sh \
        --task "{{task}}" \
        collect "{{experiment}}"

teleop-test:
    #!/usr/bin/env bash
    set -euo pipefail
    scripts/apps/teleop/a1_teleop_runtime.sh start
    source {{repo}}/scripts/runtime/a1_console.sh
    a1_info "Teleop is live. Check leader keys with: just logs"

# Move the robot to its tracked collection reset state.
reset:
    scripts/apps/teleop/a1_teleop_runtime.sh reset

lingbot *args:
    scripts/apps/lingbot/a1_lingbot_runtime.sh run {{args}}

lingbot-batch *args:
    scripts/apps/lingbot/a1_lingbot_runtime.sh batch {{args}}

lingbot-batch-resume *args:
    scripts/apps/lingbot/a1_lingbot_runtime.sh batch --resume {{args}}

lingbot-batch-report scene_note *args:
    {{vpy}} -m galaxea_a1_runtime.apps.lingbot.batch_export report --repo-root "{{repo}}" --scene-note "{{scene_note}}" {{args}}

lingbot-batch-export scene_note *args:
    {{vpy}} -m galaxea_a1_runtime.apps.lingbot.batch_export export --repo-root "{{repo}}" --scene-note "{{scene_note}}" {{args}}

lingbot-setup *args:
    scripts/apps/lingbot/a1_lingbot_runtime.sh setup {{args}}

lingbot-verify *args:
    scripts/apps/lingbot/a1_lingbot_runtime.sh verify {{args}}

lingbot-smoke *args:
    scripts/apps/lingbot/a1_lingbot_runtime.sh smoke {{args}}

lingbot-attention *args:
    scripts/apps/lingbot/a1_lingbot_runtime.sh attention {{args}}

pi05-setup:
    scripts/apps/pi05/a1_pi05_runtime.sh setup

pi05-verify:
    scripts/apps/pi05/a1_pi05_runtime.sh verify

pi05-smoke:
    scripts/apps/pi05/a1_pi05_runtime.sh smoke

pi05:
    scripts/apps/pi05/a1_pi05_runtime.sh start

offline-eval run_id="":
    scripts/apps/eef_policy_offline_eval.sh {{run_id}}

teacher-force run_id="":
    scripts/apps/eef_policy_teacher_forcing.sh {{run_id}}

# Stop repository-owned motion and inference runtimes.
stop:
    scripts/apps/teleop/a1_teleop_runtime.sh stop >/dev/null 2>&1 || true
    scripts/apps/lingbot/a1_lingbot_runtime.sh stop >/dev/null 2>&1 || true
    scripts/apps/pi05/a1_pi05_runtime.sh stop >/dev/null 2>&1 || true
    scripts/runtime/a1_joint_runtime.sh stop >/dev/null 2>&1 || true
    scripts/runtime/a1_runtime.sh stop >/dev/null 2>&1 || true
    scripts/runtime/a1_stop_managed.sh --keep-camera-monitor

logs:
    scripts/apps/teleop/a1_teleop_runtime.sh logs || true
    scripts/runtime/a1_runtime.sh logs || true
    scripts/apps/cameras/a1_camera_observability_runtime.sh logs || true

# ── Dataset ─────────────────────────────────────────────────────────────────

# Validate a canonical dataset.
dataset-doctor experiment *args:
    {{vpy}} -m galaxea_a1_runtime.cli dataset doctor \
        --repo-root "{{repo}}" \
        "{{experiment}}" {{args}}

derive config target="all":
    {{vpy}} -m galaxea_a1_runtime.lerobot.derive \
        --config "{{config}}" \
        --target "{{target}}"

# Export a canonical dataset to joint-action LeRobot v2.1.
export-v21 experiment *args:
    {{vpy}} -m galaxea_a1_runtime.cli dataset export-v21 \
        --repo-root "{{repo}}" \
        "{{experiment}}" {{args}}
