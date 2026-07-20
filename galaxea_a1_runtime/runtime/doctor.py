"""Static runtime doctor checks.

The static doctor is intentionally hardware-free. It may inspect files and
import pure modules, but it must not start ROS, Docker, cameras, or serial IO.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from galaxea_a1_runtime.configuration.paths import (
    LINGBOT_CONFIG,
    PI05_CONFIG,
    TELEOP_CONFIG,
)
from galaxea_a1_runtime.constants import SAFE_RELAY_SCRIPT
from galaxea_a1_runtime.runtime.health_checks import (
    Check,
)

GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def run_static_doctor(repo_root: Path) -> list[Check]:
    checks: list[Check] = []

    def add(name: str, ok: bool, detail: str, *, required: bool = True) -> None:
        checks.append(
            Check(name, "PASS" if ok else ("FAIL" if required else "WARN"), detail)
        )

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
        from galaxea_a1_runtime.apps.lingbot.config import load_lingbot_config
        from galaxea_a1_runtime.apps.pi05.config import load_pi05_config
        from galaxea_a1_runtime.teleop.config import load_teleop_config

        teleop_config = load_teleop_config(
            repo_root / TELEOP_CONFIG, repo_root=repo_root
        )
        lingbot_config = load_lingbot_config(
            repo_root / LINGBOT_CONFIG,
            repo_root=repo_root,
        )
        pi05_config = load_pi05_config(
            repo_root / PI05_CONFIG,
            repo_root=repo_root,
        )
        system_paths = {
            teleop_config.system.path,
            lingbot_config.system.path,
            pi05_config.system.path,
        }
        add(
            "tracked_config_graph",
            len(system_paths) == 1,
            "Teleop, LingBot, and pi0.5 configs parsed; System config(s): "
            + ", ".join(str(path) for path in sorted(system_paths)),
        )
    except Exception as exc:
        add("tracked_config_graph", False, repr(exc))

    pyproject = repo_root / "pyproject.toml"
    try:
        data = tomllib.loads(pyproject.read_text())
        sources = data.get("tool", {}).get("uv", {}).get("sources", {})
        expected_sources = {
            "embodied-ops": "external/embodied-ops",
            "lerobot": "third_party/lerobot",
            "lerobot-robot-galaxea-a1": "external/lerobot-robot-galaxea-a1",
            "lerobot-teleoperator-galaxea-a1-so-leader": (
                "external/lerobot-teleoperator-galaxea-a1-so-leader"
            ),
        }
        for package, expected_path in expected_sources.items():
            source = sources.get(package, {})
            add(
                f"{package.replace('-', '_')}_source",
                source.get("path") == expected_path and source.get("editable") is True,
                f"pyproject {package} source={source!r}",
            )
    except Exception as exc:
        add("first_party_plugin_sources", False, repr(exc))

    third_party_lerobot = repo_root / "third_party" / "lerobot"
    add("third_party_lerobot", third_party_lerobot.is_dir(), str(third_party_lerobot))
    lerobot_vendor: dict | None = None
    try:
        vendor_data = tomllib.loads(vendor_manifest.read_text())
        vendors = _vendor_entries(vendor_data)
        vendor_names = tuple(vendor["name"] for vendor in vendors)
        vendor_paths = {Path(str(vendor["path"])) for vendor in vendors}
        tracked_vendor_paths = {
            path.relative_to(repo_root)
            for path in (repo_root / "third_party").iterdir()
            if path.is_dir() and not path.name.startswith(".")
        }
        add(
            "third_party_vendor_manifest_entries",
            len(vendor_names) == len(set(vendor_names))
            and len(vendor_paths) == len(vendors)
            and vendor_paths == tracked_vendor_paths,
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
            missing_overrides = [
                item for item in overrides if not (vendor_path / item).exists()
            ]
            add(
                f"vendor_{name}_local_overrides",
                not missing_overrides,
                "none" if not overrides else "tracked: " + ", ".join(overrides),
                required=False,
            )
        lerobot_vendor = next(
            (vendor for vendor in vendors if vendor["name"] == "lerobot"), None
        )
        add(
            "vendor_lerobot_rev",
            lerobot_vendor is not None
            and GIT_COMMIT.fullmatch(str(lerobot_vendor.get("upstream_rev", "")))
            is not None,
            "missing"
            if lerobot_vendor is None
            else str(lerobot_vendor.get("upstream_rev")),
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
        "none"
        if not nested_git_dirs
        else "local artifact(s): " + ", ".join(nested_git_dirs),
        required=False,
    )
    vendored_pyproject = third_party_lerobot / "pyproject.toml"
    try:
        vendored = tomllib.loads(vendored_pyproject.read_text())
        version = vendored.get("project", {}).get("version")
        expected_version = (
            None if lerobot_vendor is None else lerobot_vendor.get("version")
        )
        add(
            "vendored_lerobot_version",
            expected_version is not None and version == expected_version,
            f"third_party/lerobot version={version!r}; manifest={expected_version!r}",
            required=False,
        )
    except Exception as exc:
        add("vendored_lerobot_version", False, repr(exc), required=False)
    vendored_so_leader = (
        third_party_lerobot
        / "src"
        / "lerobot"
        / "teleoperators"
        / "so_leader"
        / "so_leader.py"
    )
    a1_so_leader = (
        repo_root
        / "external"
        / "lerobot-teleoperator-galaxea-a1-so-leader"
        / "lerobot_teleoperator_galaxea_a1_so_leader"
        / "galaxea_a1_so_leader.py"
    )
    add("a1_so_leader_plugin", a1_so_leader.is_file(), str(a1_so_leader))
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

    relay_script = repo_root / "scripts" / "runtime" / SAFE_RELAY_SCRIPT
    add("safe_relay_script", relay_script.is_file(), str(relay_script))
    runtime_services = repo_root / "scripts" / "runtime" / "a1_services.sh"
    add("runtime_services_lib", runtime_services.is_file(), str(runtime_services))
    runtime_processes = repo_root / "scripts" / "runtime" / "a1_processes.sh"
    add("runtime_processes_lib", runtime_processes.is_file(), str(runtime_processes))
    runtime_tmux = repo_root / "scripts" / "runtime" / "a1_tmux.sh"
    add("runtime_tmux_lib", runtime_tmux.is_file(), str(runtime_tmux))
    joint_tracker_launch = (
        repo_root / "scripts" / "runtime" / "joint_tracker_staged.launch"
    )
    add(
        "joint_tracker_staged_launch",
        joint_tracker_launch.is_file(),
        str(joint_tracker_launch),
    )
    teleop_runtime = repo_root / "scripts" / "apps" / "teleop" / "a1_teleop_runtime.sh"
    add("teleop_runtime_script", teleop_runtime.is_file(), str(teleop_runtime))
    teleop_bridge = repo_root / "scripts" / "apps" / "teleop" / "so100_joint_bridge.py"
    add("teleop_bridge_script", teleop_bridge.is_file(), str(teleop_bridge))
    teleop_collect = repo_root / "scripts" / "apps" / "teleop" / "teleop_collect.py"
    add("teleop_collect_script", teleop_collect.is_file(), str(teleop_collect))
    camera_web_runtime = (
        repo_root / "scripts" / "apps" / "cameras" / "a1_camera_web_runtime.sh"
    )
    add(
        "camera_web_runtime_script",
        camera_web_runtime.is_file(),
        str(camera_web_runtime),
    )
    lingbot_runtime = (
        repo_root / "scripts" / "apps" / "lingbot" / "a1_lingbot_runtime.sh"
    )
    add("lingbot_runtime_script", lingbot_runtime.is_file(), str(lingbot_runtime))
    lingbot_run_artifacts = (
        repo_root / "galaxea_a1_runtime" / "apps" / "lingbot" / "run_artifacts.py"
    )
    add(
        "lingbot_run_artifacts",
        lingbot_run_artifacts.is_file(),
        str(lingbot_run_artifacts),
    )
    lingbot_batch_config = (
        repo_root / "configs" / "runs" / "lingbot" / "fruit_placement.toml"
    )
    add(
        "lingbot_batch_config",
        lingbot_batch_config.is_file(),
        str(lingbot_batch_config),
    )
    a1_reset_script = repo_root / "scripts" / "runtime" / "a1_reset.py"
    add("a1_reset_script", a1_reset_script.is_file(), str(a1_reset_script))

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
            raise ValueError(
                f"vendor {vendor['name']!r} has unsupported mode {vendor['mode']!r}"
            )
        out.append(vendor)
    return tuple(out)
