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
    {{repo}}/scripts/apps/cameras/a1_camera_web_runtime.sh --help >/dev/null
    {{vpy}} {{repo}}/scripts/apps/lingbot/a1_lingbot_doctor.py --help >/dev/null
    {{vpy}} {{repo}}/scripts/apps/lingbot/lingbot_va_ee_bridge.py --help >/dev/null
    {{vpy}} -m galaxea_a1_runtime.apps.lingbot.batch_config --help >/dev/null
    {{vpy}} -m galaxea_a1_runtime.apps.lingbot.batch_export --help >/dev/null
    {{vpy}} -m galaxea_a1_runtime.apps.lingbot.batch_progress --help >/dev/null
    {{vpy}} -m galaxea_a1_runtime.apps.lingbot.operator_input --help >/dev/null
    {{vpy}} -m galaxea_a1_runtime.apps.lingbot.run_artifacts --help >/dev/null
    {{vpy}} {{repo}}/scripts/apps/lingbot/probe_lingbot_server.py --help >/dev/null
    {{vpy}} {{repo}}/scripts/apps/lingbot/setup_lingbot_inference.py --help >/dev/null
    {{vpy}} {{repo}}/scripts/apps/lingbot/smoke_lingbot_inference.py --help >/dev/null
    {{vpy}} {{repo}}/scripts/apps/lingbot/verify_lingbot_inference.py --help >/dev/null
    {{vpy}} {{repo}}/scripts/apps/pi05/probe_pi05_server.py --help >/dev/null
    {{vpy}} {{repo}}/scripts/apps/pi05/pi05_ee_bridge.py --help >/dev/null
    {{vpy}} {{repo}}/scripts/apps/pi05/setup_pi05_inference.py --help >/dev/null
    {{vpy}} {{repo}}/scripts/apps/pi05/smoke_pi05_inference.py --help >/dev/null
    {{vpy}} {{repo}}/scripts/apps/pi05/verify_pi05_inference.py --help >/dev/null
    {{vpy}} {{repo}}/scripts/runtime/a1_reset.py --help >/dev/null
    {{vpy}} -m galaxea_a1_runtime.apps.task_selection --help >/dev/null
    {{vpy}} -m galaxea_a1_runtime.apps.eef_policy_offline --help >/dev/null
    {{vpy}} -m galaxea_a1_runtime.apps.eef_policy_teacher_forcing --help >/dev/null
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
    #!/usr/bin/env bash
    set -euo pipefail
    scripts/apps/cameras/a1_camera_web_runtime.sh stop
    trap 'scripts/apps/cameras/a1_camera_web_runtime.sh' EXIT
    {{vpy}} scripts/apps/cameras/a1_camera_diagnostics.py {{args}}

camera-web *args:
    scripts/apps/cameras/a1_camera_web_runtime.sh {{args}}

eef-test:
    scripts/runtime/a1_joint_runtime.sh services
    scripts/runtime/a1_joint_runtime.sh eef-nudge --execute

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
    scripts/apps/cameras/a1_camera_web_runtime.sh logs || true

# ── Dataset ─────────────────────────────────────────────────────────────────

convert experiment target="all":
    {{vpy}} -m galaxea_a1_runtime.lerobot.pipeline \
        --config "{{repo}}/configs/datasets/{{experiment}}.toml" \
        --target "{{target}}"
