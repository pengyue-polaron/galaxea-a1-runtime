from pathlib import Path

from galaxea_a1_runtime.runtime.doctor import checks_exit_code, run_static_doctor


def test_static_doctor_has_no_failures_for_repo_root():
    repo = Path(__file__).resolve().parents[1]
    checks = run_static_doctor(repo)

    assert checks_exit_code(checks) == 0
    assert {check.name for check in checks} >= {
        "architecture_doc",
        "runbook_doc",
        "safety_doc",
        "runtime_package",
        "pure_imports",
        "third_party_lerobot",
        "relay_core_shim",
        "joint_tracker_staged_launch",
        "teleop_runtime_script",
        "teleop_config",
        "teleop_bridge_script",
        "teleop_collect_script",
        "legacy_mainline_removed",
        "base_runtime_lingbot_free",
    }
