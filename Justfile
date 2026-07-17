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
    {{vpy}} {{repo}}/scripts/apps/cameras/a1_camera_diagnostics.py --help >/dev/null
    {{vpy}} {{repo}}/scripts/apps/lingbot/a1_lingbot_doctor.py --help >/dev/null
    {{vpy}} {{repo}}/scripts/apps/lingbot/lingbot_va_ee_bridge.py --help >/dev/null
    {{vpy}} {{repo}}/scripts/apps/lingbot/probe_lingbot_server.py --help >/dev/null
    {{vpy}} {{repo}}/scripts/apps/lingbot/setup_lingbot_inference.py --help >/dev/null
    {{vpy}} {{repo}}/scripts/apps/lingbot/smoke_lingbot_inference.py --help >/dev/null
    {{vpy}} {{repo}}/scripts/apps/lingbot/verify_lingbot_inference.py --help >/dev/null
    {{vpy}} {{repo}}/scripts/apps/pi05/probe_pi05_server.py --help >/dev/null
    {{vpy}} {{repo}}/scripts/apps/pi05/pi05_ee_bridge.py --help >/dev/null
    {{vpy}} {{repo}}/scripts/apps/pi05/setup_pi05_inference.py --help >/dev/null
    {{vpy}} {{repo}}/scripts/apps/pi05/smoke_pi05_inference.py --help >/dev/null
    {{vpy}} {{repo}}/scripts/apps/pi05/verify_pi05_inference.py --help >/dev/null
    {{vpy}} {{repo}}/scripts/apps/teleop/so100_joint_bridge.py --help >/dev/null
    {{vpy}} {{repo}}/scripts/apps/teleop/teleop_collect.py --help >/dev/null
    {{repo}}/scripts/apps/recording/a1_rosbag.sh --help >/dev/null
    {{vpy}} {{repo}}/scripts/runtime/a1_ros_python_check.py --help >/dev/null
    {{vpy}} -m ruff check {{repo}}/galaxea_a1_runtime {{repo}}/scripts {{repo}}/tests
    {{vpy}} -m ruff format --check {{repo}}/galaxea_a1_runtime {{repo}}/scripts {{repo}}/tests
    just test

ros-python-check:
    #!/usr/bin/env bash
    set -euo pipefail
    source {{repo}}/scripts/runtime/a1_config.sh
    a1_load_shell_config env \
      PYTHONPATH="{{repo}}:${PYTHONPATH:-}" \
      {{vpy}} -m galaxea_a1_runtime.configuration.system \
      --repo-root "{{repo}}" --shell
    docker image inspect "${IMAGE}" >/dev/null
    docker run --rm --network none \
      --entrypoint /workspace/scripts/runtime/a1_ros_python_check.py \
      -v "{{repo}}:/workspace:ro" \
      "${IMAGE}" \
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

hardware *args:
    {{vpy}} {{repo}}/scripts/runtime/a1_hardware_check.py {{args}}

rosbag *args:
    {{repo}}/scripts/apps/recording/a1_rosbag.sh {{args}}

cameras *args:
    {{vpy}} scripts/apps/cameras/a1_camera_diagnostics.py {{args}}

camera-web *args:
    scripts/apps/cameras/a1_camera_web_runtime.sh {{args}}

eef-test:
    scripts/runtime/a1_runtime.sh services
    scripts/runtime/a1_runtime.sh eef-nudge --execute

teleop experiment:
    scripts/apps/teleop/a1_teleop_runtime.sh collect "{{experiment}}"

teleop-test:
    #!/usr/bin/env bash
    set -euo pipefail
    scripts/apps/teleop/a1_teleop_runtime.sh start
    source {{repo}}/scripts/runtime/a1_console.sh
    a1_info "Teleop is live. Check leader keys with: just logs"

reset:
    scripts/apps/teleop/a1_teleop_runtime.sh reset

lingbot:
    scripts/apps/lingbot/a1_lingbot_runtime.sh start

lingbot-setup:
    scripts/apps/lingbot/a1_lingbot_runtime.sh setup

lingbot-verify:
    scripts/apps/lingbot/a1_lingbot_runtime.sh verify

lingbot-smoke:
    scripts/apps/lingbot/a1_lingbot_runtime.sh smoke

pi05-setup:
    scripts/apps/pi05/a1_pi05_runtime.sh setup

pi05-verify:
    scripts/apps/pi05/a1_pi05_runtime.sh verify

pi05-smoke:
    scripts/apps/pi05/a1_pi05_runtime.sh smoke

pi05:
    scripts/apps/pi05/a1_pi05_runtime.sh start

stop:
    scripts/apps/cameras/a1_camera_web_runtime.sh stop >/dev/null 2>&1 || true
    scripts/apps/teleop/a1_teleop_runtime.sh stop >/dev/null 2>&1 || true
    scripts/apps/lingbot/a1_lingbot_runtime.sh stop >/dev/null 2>&1 || true
    scripts/apps/pi05/a1_pi05_runtime.sh stop >/dev/null 2>&1 || true
    scripts/runtime/a1_joint_runtime.sh stop >/dev/null 2>&1 || true
    scripts/runtime/a1_runtime.sh stop >/dev/null 2>&1 || true
    scripts/runtime/a1_stop_managed.sh

logs:
    scripts/apps/teleop/a1_teleop_runtime.sh logs || true
    scripts/runtime/a1_runtime.sh logs || true

# ── Dataset ─────────────────────────────────────────────────────────────────

convert experiment target="all":
    {{vpy}} -m galaxea_a1_runtime.lerobot.pipeline \
        --config "{{repo}}/configs/datasets/{{experiment}}.toml" \
        --target "{{target}}"
