"""Command line entry point for hardware-free runtime operations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from galaxea_a1_runtime.config import RuntimeProfile
from galaxea_a1_runtime.lerobot.migration import plan_raw_episodes_to_v30, plan_v21_to_v30
from galaxea_a1_runtime.policies.profiles import POLICY_PROFILES
from galaxea_a1_runtime.runtime.doctor import (
    checks_exit_code,
    checks_to_json,
    print_checks,
    run_static_doctor,
)
from galaxea_a1_runtime.runtime.safety_report import format_safety_report, safety_report_as_dict
from galaxea_a1_runtime.runtime.supervisor import build_runtime_plan, format_runtime_plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="galaxea-a1-runtime")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor")
    doctor.add_argument("--repo-root", type=Path, default=Path.cwd())
    doctor.add_argument("--json", action="store_true")

    profiles = subparsers.add_parser("policy-profiles")
    profiles.add_argument("--json", action="store_true")

    safety = subparsers.add_parser("safety-report")
    safety.add_argument("--json", action="store_true")

    migrate = subparsers.add_parser("migration-plan")
    migrate.add_argument("--kind", choices=["raw-episodes", "lerobot-v2.1"], required=True)
    migrate.add_argument("--repo-id", required=True)
    migrate.add_argument("--source-root", type=Path)
    migrate.add_argument("--target-root", type=Path)
    migrate.add_argument("--json", action="store_true")

    plan = subparsers.add_parser("plan")
    plan.add_argument("--repo-root", type=Path, default=Path.cwd())

    runtime_plan = subparsers.add_parser("runtime-plan")
    runtime_plan.add_argument("--repo-root", type=Path, default=Path.cwd())
    runtime_plan.add_argument(
        "--profile",
        choices=[item.value for item in RuntimeProfile],
        default=RuntimeProfile.STATIC.value,
    )
    runtime_plan.add_argument("--serial", default="/dev/a1")

    args = parser.parse_args(argv)
    if args.command == "doctor":
        checks = run_static_doctor(args.repo_root)
        if args.json:
            print(checks_to_json(checks))
        else:
            print_checks(checks)
        return checks_exit_code(checks)

    if args.command == "policy-profiles":
        if args.json:
            print(json.dumps({k: vars(v) for k, v in POLICY_PROFILES.items()}, indent=2, sort_keys=True))
        else:
            for profile in POLICY_PROFILES.values():
                print(
                    f"{profile.name}: policy.type={profile.lerobot_policy_type} "
                    f"action_mode={profile.action_mode} checkpoint={profile.default_checkpoint}"
                )
        return 0

    if args.command == "safety-report":
        if args.json:
            print(json.dumps(safety_report_as_dict(), indent=2, sort_keys=True))
        else:
            print(format_safety_report())
        return 0

    if args.command == "migration-plan":
        if args.kind == "lerobot-v2.1":
            migration_plan = plan_v21_to_v30(repo_id=args.repo_id)
        else:
            if args.source_root is None or args.target_root is None:
                parser.error("--source-root and --target-root are required for raw-episodes")
            migration_plan = plan_raw_episodes_to_v30(
                source_root=args.source_root,
                target_repo_id=args.repo_id,
                target_root=args.target_root,
            )
        if args.json:
            print(json.dumps(vars(migration_plan), indent=2, sort_keys=True, default=str))
        else:
            print(migration_plan.shell_command())
            for note in migration_plan.notes:
                print(f"- {note}")
        return 0

    if args.command == "plan":
        print((args.repo_root / "docs" / "ARCHITECTURE.md").read_text())
        return 0

    if args.command == "runtime-plan":
        plan = build_runtime_plan(
            repo_root=args.repo_root,
            profile=RuntimeProfile(args.profile),
            serial=args.serial,
        )
        print(format_runtime_plan(plan))
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
