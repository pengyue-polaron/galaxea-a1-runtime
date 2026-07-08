"""Static runtime doctor checks.

The static doctor is intentionally hardware-free. It may inspect files and
import pure modules, but it must not start ROS, Docker, cameras, or serial IO.
"""

from __future__ import annotations

import json
import sys
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

EXPECTED_LEROBOT_V060_COMMIT = "30da8e687a6dfc617fcd94afc367ac7071c376ce"
EXPECTED_VENDOR_NAMES = ("A1_SDK", "A1_SDK_runtime", "lerobot")
REMOVED_MAINLINE_PATHS = (
    "a1",
    "CLAUDE.md",
    "scripts/collect_data",
    "scripts/inference",
    "scripts/train",
    "scripts/process_data/convert_episodes_to_lerobot_v21.py",
    "third_party/TFP_pro",
    "troubleshooting.md",
)


@dataclass(frozen=True)
class Check:
    name: str
    level: str
    detail: str


def run_static_doctor(repo_root: Path) -> list[Check]:
    checks: list[Check] = []

    def add(name: str, ok: bool, detail: str, *, required: bool = True) -> None:
        checks.append(Check(name, "PASS" if ok else ("FAIL" if required else "WARN"), detail))

    architecture = repo_root / "docs" / "ARCHITECTURE.md"
    add("architecture_doc", architecture.is_file(), str(architecture))
    runbook_doc = repo_root / "docs" / "RUNBOOK.md"
    add("runbook_doc", runbook_doc.is_file(), str(runbook_doc))
    safety_doc = repo_root / "docs" / "SAFETY.md"
    add("safety_doc", safety_doc.is_file(), str(safety_doc))
    third_party_doc = repo_root / "third_party" / "README.md"
    add("third_party_policy_doc", third_party_doc.is_file(), str(third_party_doc))
    vendor_manifest = repo_root / "third_party" / "vendors.toml"
    add("third_party_vendor_manifest", vendor_manifest.is_file(), str(vendor_manifest))

    package_dir = repo_root / "galaxea_a1_runtime"
    add("runtime_package", package_dir.is_dir(), str(package_dir))

    try:
        from galaxea_a1_runtime.schema import default_dataset_contract
        from galaxea_a1_runtime.safety import RelayInputs, validate_relay_inputs
        from galaxea_a1_runtime.runtime.safety_report import build_safety_settings
        from galaxea_a1_runtime.collection import state_names_for_mode
        from galaxea_a1_runtime.teleop import JointMappingConfig
        from galaxea_a1_runtime.teleop.config import load_teleop_config

        contract = default_dataset_contract()
        settings = build_safety_settings()
        teleop_state_names = state_names_for_mode("eef_joint")
        teleop_mapping = JointMappingConfig()
        teleop_config = load_teleop_config(repo_root / "configs" / "teleop" / "a1_so100.toml", repo_root=repo_root)
        decision = validate_relay_inputs(
            RelayInputs(
                enabled=False,
                joint_age=0.0,
                source_age=0.0,
                status_age=0.0,
                joint_count=0,
                source_count=0,
                motor_error_codes=(),
            )
        )
        add(
            "pure_imports",
            contract.dataset_format == "v3.0"
            and not decision.allowed
            and len(settings) > 0
            and len(teleop_state_names) == 14
            and len(teleop_mapping.sign) == 6,
            "schema, safety, collection, teleop, and safety report imported without ROS",
        )
        add(
            "teleop_config",
            teleop_config.collection.state_mode.value == "joint"
            and teleop_config.gripper.max_stroke_mm == 200.0,
            str(teleop_config.path),
        )
    except Exception as exc:
        add("pure_imports", False, repr(exc))

    pyproject = repo_root / "pyproject.toml"
    try:
        data = tomllib.loads(pyproject.read_text())
        rev = data.get("tool", {}).get("uv", {}).get("sources", {}).get("lerobot", {}).get("rev")
        add(
            "lerobot_v060_pin",
            rev == EXPECTED_LEROBOT_V060_COMMIT,
            f"pyproject lerobot rev={rev!r}; target={EXPECTED_LEROBOT_V060_COMMIT}",
            required=False,
        )
    except Exception as exc:
        add("lerobot_v060_pin", False, repr(exc), required=False)

    third_party_lerobot = repo_root / "third_party" / "lerobot"
    add("third_party_lerobot", third_party_lerobot.is_dir(), str(third_party_lerobot))
    try:
        vendor_data = tomllib.loads(vendor_manifest.read_text())
        vendors = _vendor_entries(vendor_data)
        vendor_names = tuple(vendor["name"] for vendor in vendors)
        add(
            "third_party_vendor_manifest_entries",
            set(EXPECTED_VENDOR_NAMES) <= set(vendor_names) and len(vendor_names) == len(set(vendor_names)),
            ", ".join(vendor_names),
        )
        for vendor in vendors:
            name = str(vendor["name"])
            rel_path = Path(str(vendor["path"]))
            vendor_path = repo_root / rel_path
            add(f"vendor_{name}_path", vendor_path.exists(), str(rel_path))
            nested = sorted(
                str(path.relative_to(repo_root))
                for path in vendor_path.glob("**/.git")
                if path.is_dir()
            )
            add(
                f"vendor_{name}_no_nested_git",
                not nested,
                "none" if not nested else ", ".join(nested),
            )
            policy = str(vendor.get("patch_policy", "")).strip()
            add(f"vendor_{name}_patch_policy", bool(policy), policy or "missing")
            overrides = tuple(str(item) for item in vendor.get("local_overrides", ()))
            missing_overrides = [item for item in overrides if not (vendor_path / item).exists()]
            add(
                f"vendor_{name}_local_overrides",
                not missing_overrides,
                "none" if not overrides else "tracked: " + ", ".join(overrides),
                required=False,
            )
        lerobot_vendor = next((vendor for vendor in vendors if vendor["name"] == "lerobot"), None)
        add(
            "vendor_lerobot_rev",
            lerobot_vendor is not None and lerobot_vendor.get("upstream_rev") == EXPECTED_LEROBOT_V060_COMMIT,
            "missing" if lerobot_vendor is None else str(lerobot_vendor.get("upstream_rev")),
            required=False,
        )
    except Exception as exc:
        add("third_party_vendor_manifest_entries", False, repr(exc))
    nested_git_dirs = sorted(
        str(path.relative_to(repo_root))
        for path in (repo_root / "third_party").glob("**/.git")
        if path.is_dir()
    )
    add(
        "third_party_nested_git_dirs",
        not nested_git_dirs,
        "none" if not nested_git_dirs else "local artifact(s): " + ", ".join(nested_git_dirs),
        required=False,
    )
    vendored_pyproject = third_party_lerobot / "pyproject.toml"
    try:
        vendored = tomllib.loads(vendored_pyproject.read_text())
        version = vendored.get("project", {}).get("version")
        add(
            "vendored_lerobot_v060",
            version == "0.6.0",
            f"third_party/lerobot version={version!r}; target='0.6.0'",
            required=False,
        )
    except Exception as exc:
        add("vendored_lerobot_v060", False, repr(exc), required=False)
    vendored_so_leader = third_party_lerobot / "src" / "lerobot" / "teleoperators" / "so_leader" / "so_leader.py"
    a1_so_leader = repo_root / "galaxea_a1_runtime" / "teleop" / "a1_so_leader.py"
    add("a1_so_leader_adapter", a1_so_leader.is_file(), str(a1_so_leader))
    try:
        text = vendored_so_leader.read_text()
        add(
            "vendored_so_leader_unpatched",
            '"shoulder_pan": Motor(' in text and '"joint0": Motor(' not in text,
            str(vendored_so_leader),
            required=False,
        )
    except Exception as exc:
        add("vendored_so_leader_unpatched", False, repr(exc), required=False)

    relay_core = repo_root / "scripts" / "runtime" / "a1_relay_core.py"
    add("relay_core_shim", relay_core.is_file(), str(relay_core))
    joint_tracker_launch = repo_root / "scripts" / "runtime" / "joint_tracker_staged.launch"
    add("joint_tracker_staged_launch", joint_tracker_launch.is_file(), str(joint_tracker_launch))
    teleop_runtime = repo_root / "scripts" / "apps" / "teleop" / "a1_teleop_runtime.sh"
    add("teleop_runtime_script", teleop_runtime.is_file(), str(teleop_runtime))
    teleop_bridge = repo_root / "scripts" / "apps" / "teleop" / "so100_joint_bridge.py"
    add("teleop_bridge_script", teleop_bridge.is_file(), str(teleop_bridge))
    teleop_collect = repo_root / "scripts" / "apps" / "teleop" / "teleop_collect.py"
    add("teleop_collect_script", teleop_collect.is_file(), str(teleop_collect))

    existing_removed_paths = [
        path for path in REMOVED_MAINLINE_PATHS if (repo_root / path).exists()
    ]
    add(
        "legacy_mainline_removed",
        not existing_removed_paths,
        "removed paths absent"
        if not existing_removed_paths
        else "still present: " + ", ".join(existing_removed_paths),
    )

    base_runtime = repo_root / "scripts" / "runtime" / "a1_runtime.sh"
    try:
        text = base_runtime.read_text()
        add(
            "base_runtime_lingbot_free",
            "lingbot" not in text.lower(),
            str(base_runtime),
        )
    except Exception as exc:
        add("base_runtime_lingbot_free", False, repr(exc))

    return checks


def checks_to_json(checks: list[Check]) -> str:
    return json.dumps([asdict(check) for check in checks], indent=2, sort_keys=True)


def _vendor_entries(data: dict) -> tuple[dict, ...]:
    if int(data.get("schema_version", 0)) != 1:
        raise ValueError("third_party/vendors.toml schema_version must be 1")
    vendors = data.get("vendor")
    if not isinstance(vendors, list) or not vendors:
        raise ValueError("third_party/vendors.toml must contain [[vendor]] entries")
    required = {"name", "path", "mode", "upstream", "patch_policy"}
    out: list[dict] = []
    for index, vendor in enumerate(vendors):
        if not isinstance(vendor, dict):
            raise ValueError(f"vendor entry {index} must be a table")
        missing = sorted(required - set(vendor))
        if missing:
            raise ValueError(f"vendor entry {index} missing fields: {missing}")
        if vendor["mode"] != "snapshot":
            raise ValueError(f"vendor {vendor['name']!r} has unsupported mode {vendor['mode']!r}")
        out.append(vendor)
    return tuple(out)


def checks_exit_code(checks: list[Check]) -> int:
    return 1 if any(check.level == "FAIL" for check in checks) else 0


def print_checks(checks: list[Check]) -> None:
    width = max((len(check.name) for check in checks), default=0)
    for check in checks:
        print(f"[{check.level:4}] {check.name:<{width}}  {check.detail}")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    checks = run_static_doctor(args.repo_root)
    if args.json:
        print(checks_to_json(checks))
    else:
        print_checks(checks)
    return checks_exit_code(checks)


if __name__ == "__main__":
    sys.exit(main())
