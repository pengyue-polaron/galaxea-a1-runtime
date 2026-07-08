set shell := ["bash", "-eu", "-o", "pipefail", "-c"]
set quiet := true

uv   := env("UV_BIN", "uv")
repo := justfile_directory()
vpy  := repo + "/.venv/bin/python"

default:
    @just --list

# ── Environment ──────────────────────────────────────────────────────────────

setup-main:
    #!/usr/bin/env bash
    set -euo pipefail
    export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
    export UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-cu128}"
    {{uv}} python install 3.12
    {{uv}} sync --frozen --python 3.12
    echo "main env ready: {{repo}}/.venv"

udev-install:
    scripts/runtime/install_a1_udev.sh

# ── New Runtime ──────────────────────────────────────────────────────────────

runtime action="doctor" *args:
    #!/usr/bin/env bash
    set -euo pipefail
    case "{{action}}" in
        plan) {{vpy}} -m galaxea_a1_runtime.cli plan --repo-root "{{repo}}" {{args}} ;;
        dry-run|profile-plan) {{vpy}} -m galaxea_a1_runtime.cli runtime-plan --repo-root "{{repo}}" {{args}} ;;
        doctor|static-doctor) {{vpy}} -m galaxea_a1_runtime.cli doctor --repo-root "{{repo}}" {{args}} ;;
        profiles|policy-profiles) {{vpy}} -m galaxea_a1_runtime.cli policy-profiles {{args}} ;;
        safety|safety-report) {{vpy}} -m galaxea_a1_runtime.cli safety-report {{args}} ;;
        test) just runtime-test ;;
        *) echo "Usage: just runtime <plan|dry-run|doctor|profiles|safety|test> [args...]"; exit 1 ;;
    esac

runtime-plan:
    @just runtime plan

runtime-test:
    {{vpy}} -m pytest \
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
        {{repo}}/tests/test_galaxea_a1_lingbot_static.py \
        {{repo}}/tests/test_galaxea_a1_runtime_doctor.py \
        {{repo}}/tests/test_galaxea_a1_runtime_supervisor.py \
        {{repo}}/tests/test_galaxea_a1_policy_profiles.py \
        {{repo}}/tests/test_galaxea_a1_lerobot_writer.py \
        {{repo}}/tests/test_galaxea_a1_migration.py \
        {{repo}}/tests/test_galaxea_a1_convert_raw.py \
        {{repo}}/tests/test_a1_relay_core.py

# ── Dataset ──────────────────────────────────────────────────────────────────

dataset action="migration-plan" *args:
    #!/usr/bin/env bash
    set -euo pipefail
    case "{{action}}" in
        migration-plan|migrate-plan) {{vpy}} -m galaxea_a1_runtime.cli migration-plan {{args}} ;;
        convert-raw) {{vpy}} -m galaxea_a1_runtime.lerobot.convert_raw {{args}} ;;
        *) echo "Usage: just dataset <migration-plan|convert-raw> [args...]"; exit 1 ;;
    esac

# ── Safe Hardware Runtime ────────────────────────────────────────────────────

a1-runtime action="status" *args:
    scripts/runtime/a1_runtime.sh "{{action}}" {{args}}

a1-teleop action="status" *args:
    scripts/apps/teleop/a1_teleop_runtime.sh "{{action}}" {{args}}

a1-lingbot action="status" *args:
    scripts/apps/lingbot/a1_lingbot_runtime.sh "{{action}}" {{args}}

collect mode="teleop" experiment="":
    #!/usr/bin/env bash
    set -euo pipefail
    case "{{mode}}" in
        teleop)
            if [[ -z "{{experiment}}" ]]; then
                echo "Usage: just collect teleop <experiment>"
                exit 2
            fi
            scripts/apps/teleop/a1_teleop_runtime.sh collect "{{experiment}}"
            ;;
        *)
            echo "Usage: just collect teleop <experiment>"
            exit 2
            ;;
    esac
