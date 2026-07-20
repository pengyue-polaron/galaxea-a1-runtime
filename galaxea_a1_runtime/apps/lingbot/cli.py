"""Configuration-driven LingBot end-effector bridge entrypoint."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from galaxea_a1_runtime.apps.eef_policy_cli import run_eef_policy_bridge
from galaxea_a1_runtime.apps.lingbot.config import (
    default_config_path,
    load_lingbot_config,
)
from galaxea_a1_runtime.console import ArgumentParser, info, warning


REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> Namespace:
    parser = ArgumentParser(description="LingBot-VA EE-pose bridge for Galaxea A1")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path(REPO_ROOT),
        help="Tracked LingBot deployment TOML",
    )
    parser.add_argument("--model")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--video-filename", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_lingbot_config(
        args.config,
        repo_root=REPO_ROOT,
        model_selector=args.model,
    )
    task = config.task_catalog.task(args.task_id)
    from galaxea_a1_runtime.apps.lingbot.bridge import A1LingBotEEBridge
    from galaxea_a1_runtime.apps.lingbot.run_artifacts import (
        record_lingbot_run_outcome,
    )

    info(f"LingBot config: {config.path}")
    info(
        f"LingBot task: {task.task_id} distribution={task.distribution} "
        f"prompt={task.prompt!r}"
    )
    if not config.execution.execute:
        warning("Dry run: execution.execute=false in the deployment config.")
    bridge = A1LingBotEEBridge(
        config,
        task,
        run_id=args.run_id,
        video_filename=args.video_filename,
    )
    return run_eef_policy_bridge(
        bridge,
        break_status_line=bridge.live_status.break_line,
        record_outcome=lambda kind, message: record_lingbot_run_outcome(
            config.recording.output_root,
            args.run_id,
            kind=kind,
            message=message,
        ),
    )
