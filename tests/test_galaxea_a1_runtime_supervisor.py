from pathlib import Path

from galaxea_a1_runtime.config import RuntimeProfile
from galaxea_a1_runtime.runtime.supervisor import build_runtime_plan, format_runtime_plan


def test_static_runtime_plan_does_not_touch_hardware():
    plan = build_runtime_plan(repo_root=Path("/repo"), profile=RuntimeProfile.STATIC)

    assert plan.touches_hardware is False
    assert "doctor" in format_runtime_plan(plan)


def test_safe_runtime_plan_is_explicitly_hardware_touching():
    plan = build_runtime_plan(repo_root=Path("/repo"), profile=RuntimeProfile.SAFE)

    assert plan.touches_hardware is True
    assert [step.name for step in plan.steps] == [
        "serial-present",
        "safe-runtime",
        "execution-doctor",
    ]


def test_collect_plan_uses_teleop_collection_entrypoint():
    plan = build_runtime_plan(repo_root=Path("/repo"), profile=RuntimeProfile.COLLECT)
    text = format_runtime_plan(plan)

    assert plan.touches_hardware is True
    assert "[start] teleop-collection" in text
    assert "just collect teleop <experiment>" in text
    assert "pending" not in text


def test_infer_plan_marks_generic_runner_selection_as_pending():
    plan = build_runtime_plan(repo_root=Path("/repo"), profile=RuntimeProfile.INFER)
    text = format_runtime_plan(plan)

    assert plan.touches_hardware is True
    assert "[pending] policy-runner-selection" in text
    assert "a1-lingbot start" in text


def test_direct_debug_plan_stops_safe_runtime_first():
    plan = build_runtime_plan(repo_root=Path("/repo"), profile=RuntimeProfile.DIRECT_DEBUG)

    assert plan.steps[0].name == "stop-safe-runtime"
    assert "arm_joint_command_host" in plan.steps[1].shell_command()
