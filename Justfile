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
    {{vpy}} -m ruff check {{repo}}/galaxea_a1_runtime {{repo}}/scripts
    {{vpy}} -m ruff format --check {{repo}}/galaxea_a1_runtime {{repo}}/scripts
    just foxglove-extension-check

# List tracked operator configurations.
configs:
    {{vpy}} -m galaxea_a1_runtime.cli configs --repo-root "{{repo}}"

# List every validated task Prompt.
prompts:
    {{vpy}} -m galaxea_a1_runtime.cli prompt list --repo-root "{{repo}}"

# Atomically register one Prompt in a tracked catalog.
prompt-register catalog task_id prompt distribution:
    {{vpy}} -m galaxea_a1_runtime.cli prompt register \
        "{{catalog}}" "{{task_id}}" "{{prompt}}" \
        --distribution "{{distribution}}" --repo-root "{{repo}}"

# Atomically create a task catalog with its initial Prompt.
prompt-catalog-create catalog catalog_id task_id prompt distribution:
    {{vpy}} -m galaxea_a1_runtime.cli prompt create-catalog \
        "{{catalog}}" "{{catalog_id}}" "{{task_id}}" "{{prompt}}" \
        --distribution "{{distribution}}" --repo-root "{{repo}}"

# Open the Operator Panel.
panel:
    {{vpy}} -m galaxea_a1_runtime.cli panel --repo-root "{{repo}}"

# Render the tracked A1 Foxglove layout from System config.
foxglove-layout:
    {{vpy}} {{repo}}/scripts/runtime/render_foxglove_layout.py

# Compile and lint the pinned shared Foxglove collection console.
foxglove-extension-check:
    npm --prefix {{repo}}/external/embodied-ops/foxglove/collection-console ci
    npm --prefix {{repo}}/external/embodied-ops/foxglove/collection-console run build
    npm --prefix {{repo}}/external/embodied-ops/foxglove/collection-console run lint

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

models:
    {{vpy}} {{repo}}/scripts/models/model_store.py doctor --repo-root "{{repo}}"

# Build the official ROS 2 camera, synchronization and recording environment.
ros2-setup:
    docker build -t galaxea-a1-runtime/ros2-jazzy:local -f {{repo}}/docker/ros2/Dockerfile {{repo}}
    docker build -t galaxea-a1-runtime/ros1-bridge:local -f {{repo}}/docker/ros1-bridge/Dockerfile {{repo}}

# Inspect native ROS 2 camera topics or record a bounded raw MCAP capture.
ros2 *args:
    {{vpy}} -m galaxea_a1_runtime.apps.cameras.ros2_tools {{args}}

# Build the isolated upstream TRAC-IK adapter without hardware access.
trac-ik-setup:
    {{vpy}} -m galaxea_a1_runtime.apps.trac_ik_setup

# Hardware-free runtime readiness check, separate from static configuration validation.
trac-ik-check:
    {{vpy}} -m galaxea_a1_runtime.apps.trac_ik_setup --check

model-fetch config:
    {{vpy}} {{repo}}/scripts/models/model_store.py fetch \
        --repo-root "{{repo}}" "{{config}}"

model-verify config:
    {{vpy}} {{repo}}/scripts/models/model_store.py verify \
        --repo-root "{{repo}}" "{{config}}"

# Register every model of a tracked release plan from verified content.
model-register-release plan *args:
    {{vpy}} {{repo}}/scripts/models/register_release.py \
        --repo-root "{{repo}}" "{{plan}}" {{args}}

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

# Start Foxglove telemetry and scoped operator services without the A1 driver/relay.
foxglove action="start":
    scripts/runtime/a1_observability_runtime.sh "{{action}}"

eef-test: trac-ik-check
    scripts/runtime/a1_joint_runtime.sh services
    scripts/runtime/a1_joint_runtime.sh eef-nudge --execute

# Reset and collect episodes into one experiment. Foxglove controls the gates.
collect experiment task:
    {{vpy}} -m galaxea_a1_runtime.cli collect \
        --repo-root "{{repo}}" --task "{{task}}" "{{experiment}}"

# Reset and collect episodes with terminal Enter/d/q controls.
collect-cli experiment task:
    {{vpy}} -m galaxea_a1_runtime.cli collect --cli \
        --repo-root "{{repo}}" --task "{{task}}" "{{experiment}}"

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

tfp-setup:
    scripts/apps/tfp/a1_tfp_runtime.sh setup

diffusion2one-setup:
    scripts/apps/diffusion2one/a1_diffusion2one_runtime.sh setup

diffusion2one-env-setup:
    scripts/apps/diffusion2one/a1_diffusion2one_runtime.sh environment

diffusion2one-verify:
    scripts/apps/diffusion2one/a1_diffusion2one_runtime.sh verify

diffusion2one-smoke:
    scripts/apps/diffusion2one/a1_diffusion2one_runtime.sh smoke

diffusion2one *args:
    scripts/apps/diffusion2one/a1_diffusion2one_runtime.sh run {{args}}

# Teacher defaults to the model server only; smoke uses synthetic observations.
diffusion2one-teacher action="server" *args:
    scripts/apps/diffusion2one/a1_diffusion2one_teacher_runtime.sh {{action}} {{args}}

# MOVES HARDWARE: select and launch a registered LingBot-family model.
inference *args:
    {{vpy}} -m galaxea_a1_runtime.cli inference --repo-root "{{repo}}" {{args}}

tfp-verify:
    scripts/apps/tfp/a1_tfp_runtime.sh verify

tfp-smoke:
    scripts/apps/tfp/a1_tfp_runtime.sh smoke

tfp:
    scripts/apps/tfp/a1_tfp_runtime.sh run

offline-eval run_id="":
    scripts/apps/eef_policy_offline_eval.sh {{run_id}}

teacher-force run_id="":
    scripts/apps/eef_policy_teacher_forcing.sh {{run_id}}

# Stop repository-owned motion and inference runtimes.
stop:
    scripts/apps/teleop/a1_teleop_runtime.sh stop >/dev/null 2>&1 || true
    scripts/apps/lingbot/a1_lingbot_runtime.sh stop >/dev/null 2>&1 || true
    scripts/apps/pi05/a1_pi05_runtime.sh stop >/dev/null 2>&1 || true
    scripts/apps/tfp/a1_tfp_runtime.sh stop >/dev/null 2>&1 || true
    scripts/runtime/a1_joint_runtime.sh stop >/dev/null 2>&1 || true
    scripts/runtime/a1_runtime.sh stop >/dev/null 2>&1 || true
    scripts/runtime/a1_stop_managed.sh --keep-camera-monitor

logs:
    scripts/apps/teleop/a1_teleop_runtime.sh logs || true
    scripts/runtime/a1_runtime.sh logs || true
    scripts/apps/cameras/a1_camera_observability_runtime.sh logs || true

# ── Dataset ─────────────────────────────────────────────────────────────────

# Export a finalized raw collection episode without opening hardware.
bag-export episode *args:
    {{vpy}} -m galaxea_a1_runtime.apps.teleop.bag_export "{{episode}}" {{args}}

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
