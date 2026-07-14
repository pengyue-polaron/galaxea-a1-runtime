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
    export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
    export UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-cu128}"
    {{uv}} python install 3.12
    {{uv}} sync --frozen --python 3.12
    echo "main env ready: {{repo}}/.venv"

udev:
    scripts/runtime/install_a1_udev.sh

# ── Local Checks ─────────────────────────────────────────────────────────────

check:
    {{vpy}} -m galaxea_a1_runtime.cli doctor --repo-root "{{repo}}"
    {{vpy}} -m ruff check {{repo}}/galaxea_a1_runtime {{repo}}/scripts {{repo}}/tests
    just test

test:
    {{vpy}} -m pytest -q \
        {{repo}}/tests/test_galaxea_a1_runtime_safety.py \
        {{repo}}/tests/test_galaxea_a1_cli.py \
        {{repo}}/tests/test_galaxea_a1_runtime_schema.py \
        {{repo}}/tests/test_galaxea_a1_runtime_actions.py \
        {{repo}}/tests/test_galaxea_a1_collection_schema.py \
        {{repo}}/tests/test_galaxea_a1_teleop_mapping.py \
        {{repo}}/tests/test_galaxea_a1_teleop_config.py \
        {{repo}}/tests/test_galaxea_a1_teleop_static.py \
        {{repo}}/tests/test_galaxea_a1_runtime_safety_report.py \
        {{repo}}/tests/test_galaxea_a1_runtime_lerobot_dataset.py \
        {{repo}}/tests/test_galaxea_a1_lerobot_robot.py \
        {{repo}}/tests/test_galaxea_a1_lerobot_recorder.py \
        {{repo}}/tests/test_galaxea_a1_eef.py \
        {{repo}}/tests/test_galaxea_a1_eef_bridge.py \
        {{repo}}/tests/test_galaxea_a1_ros1_adapter.py \
        {{repo}}/tests/test_galaxea_a1_lingbot_actions.py \
        {{repo}}/tests/test_galaxea_a1_lingbot_pack.py \
        {{repo}}/tests/test_galaxea_a1_lingbot_static.py \
        {{repo}}/tests/test_galaxea_a1_act_config.py \
        {{repo}}/tests/test_galaxea_a1_act_static.py \
        {{repo}}/tests/test_galaxea_a1_runtime_doctor.py \
        {{repo}}/tests/test_galaxea_a1_policy_profiles.py \
        {{repo}}/tests/test_galaxea_a1_lerobot_writer.py \
        {{repo}}/tests/test_galaxea_a1_migration.py \
        {{repo}}/tests/test_galaxea_a1_convert_raw.py \
        {{repo}}/tests/test_galaxea_a1_relay_safety.py

models:
    {{vpy}} {{repo}}/scripts/models/model_store.py doctor --repo-root "{{repo}}"

model-link slot source:
    {{vpy}} {{repo}}/scripts/models/model_store.py register \
        --repo-root "{{repo}}" "{{slot}}" "{{source}}"

# ── Hardware Workflow ────────────────────────────────────────────────────────

hardware *args:
    {{vpy}} {{repo}}/scripts/runtime/a1_hardware_check.py {{args}}

cameras *args:
    scripts/apps/teleop/a1_teleop_runtime.sh cameras {{args}}

eef-test:
    scripts/runtime/a1_runtime.sh services
    scripts/runtime/a1_runtime.sh eef-nudge --execute

teleop experiment:
    scripts/apps/teleop/a1_teleop_runtime.sh collect "{{experiment}}"

teleop-test:
    scripts/apps/teleop/a1_teleop_runtime.sh start
    @echo "Teleop is live. Check leader keys with: just logs"

reset:
    scripts/apps/teleop/a1_teleop_runtime.sh reset

lingbot:
    scripts/apps/lingbot/a1_lingbot_runtime.sh start

act:
    scripts/apps/act/a1_act_joint_runtime.sh start

stop:
    scripts/apps/act/a1_act_joint_runtime.sh stop >/dev/null 2>&1 || true
    scripts/apps/teleop/a1_teleop_runtime.sh stop >/dev/null 2>&1 || true
    scripts/apps/lingbot/a1_lingbot_runtime.sh stop >/dev/null 2>&1 || true
    scripts/runtime/a1_joint_runtime.sh stop >/dev/null 2>&1 || true
    scripts/runtime/a1_runtime.sh stop >/dev/null 2>&1 || true
    @echo "A1 runtime stopped."

logs:
    scripts/apps/act/a1_act_joint_runtime.sh logs || true
    scripts/apps/teleop/a1_teleop_runtime.sh logs || true
    scripts/runtime/a1_runtime.sh logs || true

# ── Dataset ─────────────────────────────────────────────────────────────────

convert experiment:
    {{vpy}} -m galaxea_a1_runtime.lerobot.lingbot_pack \
        --config "{{repo}}/configs/datasets/{{experiment}}.toml" --overwrite
