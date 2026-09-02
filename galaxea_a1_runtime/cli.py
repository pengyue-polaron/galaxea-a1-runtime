"""Unified command line entry point for A1 checks and operator workflows."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from embodied_ops import (
    checks_exit_code,
    checks_to_json,
    print_dataset_report,
    print_export_report,
    print_checks,
    standard_export_report,
)

from galaxea_a1_runtime.configuration.paths import (
    A1_RESET_POSE,
    LINGBOT_BATCH_CONFIG,
    LINGBOT_CONFIG,
    TELEOP_CONFIG,
)
from galaxea_a1_runtime.console import ArgumentParser, failure, success
from galaxea_a1_runtime.doctor import run_static_doctor
from galaxea_a1_runtime.safety_report import (
    format_safety_report,
    safety_report_as_dict,
)


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(
        prog="galaxea-a1-runtime",
        description="Checks, reports, and tracked A1 operator workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="run hardware-free repository checks")
    doctor.add_argument("--repo-root", type=Path, default=Path.cwd())
    doctor.add_argument("--json", action="store_true")

    safety = subparsers.add_parser(
        "safety-report", help="show the effective tracked safety contract"
    )
    safety.add_argument("--json", action="store_true")

    configs = subparsers.add_parser(
        "configs", help="list tracked operator configurations"
    )
    configs.add_argument("--repo-root", type=Path, default=Path.cwd())
    configs.add_argument("--json", action="store_true")

    config = subparsers.add_parser(
        "config", help="template, validate, or create a repository configuration"
    )
    config_commands = config.add_subparsers(dest="config_command", required=True)
    config_template = config_commands.add_parser(
        "template", help="print one validated same-kind template"
    )
    config_template.add_argument("kind")
    config_template.add_argument("source")
    config_template.add_argument("--repo-root", type=Path, default=Path.cwd())
    for name in ("validate", "create"):
        config_write = config_commands.add_parser(
            name,
            help=f"{name} a create-only configuration candidate",
        )
        config_write.add_argument("kind")
        config_write.add_argument("filename")
        config_write.add_argument("candidate", type=Path)
        config_write.add_argument("--repo-root", type=Path, default=Path.cwd())

    panel = subparsers.add_parser("panel", help="serve the tracked Web operator panel")
    panel.add_argument("--repo-root", type=Path, default=Path.cwd())

    hardware = subparsers.add_parser(
        "hardware", help="passively inspect configured serial and camera hardware"
    )
    hardware.add_argument("--config", type=Path, default=TELEOP_CONFIG)
    hardware.add_argument("--repo-root", type=Path, default=Path.cwd())
    hardware.add_argument("--json", action="store_true")

    dataset = subparsers.add_parser(
        "dataset", help="inspect or export canonical datasets"
    )
    dataset_commands = dataset.add_subparsers(dest="dataset_command", required=True)
    dataset_doctor = dataset_commands.add_parser(
        "doctor", help="validate one canonical LeRobot v3 dataset"
    )
    dataset_doctor.add_argument("experiment")
    dataset_doctor.add_argument("--config", type=Path, default=TELEOP_CONFIG)
    dataset_doctor.add_argument("--repo-root", type=Path, default=Path.cwd())
    dataset_doctor.add_argument("--json", action="store_true")
    dataset_export = dataset_commands.add_parser(
        "export-v21", help="export one canonical dataset to joint-action LeRobot v2.1"
    )
    dataset_export.add_argument("experiment")
    dataset_export.add_argument("--repo-root", type=Path, default=Path.cwd())
    dataset_export.add_argument("--json", action="store_true")

    collect = subparsers.add_parser(
        "collect", help="MOVES HARDWARE: reset, then run tracked Teleop collection"
    )
    collect.add_argument("experiment")
    collect.add_argument("--task", required=True)
    collect.add_argument("--config", type=Path, default=TELEOP_CONFIG)
    collect.add_argument("--repo-root", type=Path, default=Path.cwd())

    evaluate = subparsers.add_parser(
        "evaluate", help="MOVES HARDWARE: run one tracked LingBot evaluation"
    )
    evaluate.add_argument("task")
    evaluate.add_argument("--scene-note", required=True)
    evaluate.add_argument("--config", type=Path, default=LINGBOT_CONFIG)
    evaluate.add_argument("--model")
    evaluate.add_argument("--repo-root", type=Path, default=Path.cwd())

    batch = subparsers.add_parser(
        "batch", help="MOVES HARDWARE: run a tracked LingBot batch plan"
    )
    batch.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=LINGBOT_BATCH_CONFIG,
    )
    batch.add_argument("--scene-note", required=True)
    batch.add_argument("--model")
    batch.add_argument("--resume", action="store_true")
    batch.add_argument("--repo-root", type=Path, default=Path.cwd())

    reset = subparsers.add_parser(
        "reset", help="MOVES HARDWARE: run one tracked staged A1 reset"
    )
    reset.add_argument(
        "pose",
        nargs="?",
        type=Path,
        default=A1_RESET_POSE,
    )
    reset.add_argument("--repo-root", type=Path, default=Path.cwd())

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

    if args.command == "panel":
        from galaxea_a1_runtime.apps.operator_panel import serve_a1_operator_panel

        return serve_a1_operator_panel(args.repo_root)

    if args.command == "hardware":
        from galaxea_a1_runtime.apps.teleop.hardware_check import (
            inspect_hardware,
            print_hardware_report,
        )
        from galaxea_a1_runtime.teleop.config import load_teleop_config

        try:
            root = args.repo_root.resolve()
            config = load_teleop_config(args.config, repo_root=root)
            return print_hardware_report(
                inspect_hardware(config),
                json_output=args.json,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            failure(str(exc))
            return 2

    if args.command == "dataset":
        return _run_dataset_command(args)

    if args.command == "collect":
        from galaxea_a1_runtime.apps.operator_panel import run_collection_session

        try:
            return run_collection_session(
                args.repo_root,
                config=args.config,
                experiment=args.experiment,
                task=args.task,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            failure(str(exc))
            return 2

    from galaxea_a1_runtime.apps.operator_panel import A1OperatorPanelAdapter

    adapter = A1OperatorPanelAdapter(args.repo_root)

    if args.command == "config":
        try:
            if args.config_command == "template":
                result = adapter.config_template(
                    {"kind": args.kind, "source": args.source}
                )
                print(result["content"], end="")
                return 0
            payload = {
                "kind": args.kind,
                "filename": args.filename,
                "content": args.candidate.read_text(),
            }
            if args.config_command == "validate":
                result = adapter.validate_config(payload)
                success(f"Valid configuration: {result['path']}")
            else:
                result = adapter.create_config(payload)
                success(f"Created configuration: {result['created']}")
            return 0
        except (OSError, ValueError) as exc:
            failure(str(exc))
            return 2

    if args.command == "configs":
        catalog = adapter.catalog()
        if args.json:
            print(json.dumps(catalog, indent=2, ensure_ascii=False))
        else:
            _print_config_catalog(catalog)
        return 0

    values: dict[str, object]
    if args.command == "evaluate":
        values = {
            "config": str(args.config),
            "model": args.model,
            "task": args.task,
            "scene_note": args.scene_note,
        }
        workflow = "evaluate"
    elif args.command == "batch":
        values = {
            "config": str(args.config),
            "model": args.model,
            "scene_note": args.scene_note,
            "resume": args.resume,
        }
        workflow = "batch"
    elif args.command == "reset":
        values = {"pose": str(args.pose)}
        workflow = "reset"
    else:
        return 2

    try:
        launch = adapter.build_launch(workflow, values)
    except (ValueError, FileNotFoundError) as exc:
        failure(str(exc))
        return 2
    return subprocess.run(
        launch.command,
        cwd=args.repo_root.resolve(),
        check=False,
    ).returncode


def _print_config_catalog(catalog: dict[str, object]) -> None:
    groups = catalog["configuration_groups"]
    assert isinstance(groups, list)
    for group in groups:
        assert isinstance(group, dict)
        print(f"{group['label']}:")
        items = group["items"]
        assert isinstance(items, list)
        for item in items:
            assert isinstance(item, dict)
            print(f"  {item['label']}: {item['value']}")


def _run_dataset_command(args) -> int:
    from galaxea_a1_runtime.collection import validate_experiment_name

    try:
        root = args.repo_root.resolve()
        experiment = validate_experiment_name(args.experiment)
        if args.dataset_command == "doctor":
            from galaxea_a1_runtime.apps.teleop.dataset_doctor import dataset_report
            from galaxea_a1_runtime.teleop.config import load_teleop_config

            config = load_teleop_config(args.config, repo_root=root)
            result = dataset_report(config, experiment=experiment)
        else:
            from galaxea_a1_runtime.lerobot.derivation_config import (
                load_derivation_config,
            )
            from galaxea_a1_runtime.lerobot.derive import (
                JOINT_V21,
                build_derivatives,
            )

            config_path = root / "configs/datasets" / f"{experiment}_derivatives.toml"
            result = build_derivatives(
                load_derivation_config(config_path),
                targets=(JOINT_V21,),
            )[JOINT_V21]
            result = standard_export_report(
                robot="galaxea-a1",
                experiment=experiment,
                result=result,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        failure(str(exc))
        return 2
    if args.dataset_command == "doctor":
        print_dataset_report(result, json_output=args.json)
    else:
        print_export_report(result, json_output=args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
