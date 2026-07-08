"""Command line entry point for hardware-free runtime operations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from galaxea_a1_runtime.runtime.doctor import (
    checks_exit_code,
    checks_to_json,
    print_checks,
    run_static_doctor,
)
from galaxea_a1_runtime.runtime.safety_report import format_safety_report, safety_report_as_dict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="galaxea-a1-runtime")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor")
    doctor.add_argument("--repo-root", type=Path, default=Path.cwd())
    doctor.add_argument("--json", action="store_true")

    safety = subparsers.add_parser("safety-report")
    safety.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "doctor":
        checks = run_static_doctor(args.repo_root)
        if args.json:
            print(checks_to_json(checks))
        else:
            print_checks(checks)
        return checks_exit_code(checks)

    if args.command == "safety-report":
        if args.json:
            print(json.dumps(safety_report_as_dict(), indent=2, sort_keys=True))
        else:
            print(format_safety_report())
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
