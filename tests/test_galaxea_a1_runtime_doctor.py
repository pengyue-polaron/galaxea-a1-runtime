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
        "third_party_policy_doc",
        "third_party_vendor_manifest",
        "third_party_vendor_manifest_entries",
        "runtime_package",
        "pure_imports",
        "third_party_lerobot",
        "third_party_nested_git_dirs",
        "vendor_A1_SDK_path",
        "vendor_A1_SDK_no_nested_git",
        "vendor_A1_SDK_local_overrides",
        "vendor_A1_SDK_runtime_path",
        "vendor_lerobot_path",
        "vendor_lerobot_no_nested_git",
        "vendor_lerobot_rev",
        "a1_so_leader_adapter",
        "vendored_so_leader_unpatched",
        "relay_core_shim",
        "joint_tracker_staged_launch",
        "teleop_runtime_script",
        "teleop_config",
        "teleop_bridge_script",
        "teleop_collect_script",
        "legacy_mainline_removed",
        "base_runtime_lingbot_free",
    }
