"""Configuration-driven LingBot end-effector bridge entrypoint."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Callable, Protocol

from galaxea_a1_runtime.apps.lingbot.config import (
    default_config_path,
    load_lingbot_config,
)
from galaxea_a1_runtime.console import ArgumentParser, info, warning


REPO_ROOT = Path(__file__).resolve().parents[3]


class _LiveStatus(Protocol):
    def break_line(self) -> None: ...


class _BridgeLifecycle(Protocol):
    live_status: _LiveStatus

    def run(self) -> None: ...

    def close(self) -> None: ...


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
    return run_bridge(
        A1LingBotEEBridge(
            config,
            task,
            run_id=args.run_id,
            video_filename=args.video_filename,
        ),
        record_outcome=lambda kind, message: record_lingbot_run_outcome(
            config.recording.output_root,
            args.run_id,
            kind=kind,
            message=message,
        ),
    )


def run_bridge(
    bridge: _BridgeLifecycle,
    *,
    record_outcome: Callable[[str, str], None] | None = None,
) -> int:
    """Run one bridge and cleanly end expected operator/safety stops."""

    outcome: tuple[str, str] | None = None
    try:
        bridge.run()
    except KeyboardInterrupt:
        bridge.live_status.break_line()
        message = "Ctrl+C received; locking the arm and finalizing AgentView video."
        info(message)
        outcome = ("operator_interrupted", message)
    except Exception as exc:
        # Keep the optional numeric/URDF dependencies out of static CLI startup.
        from galaxea_a1_runtime.apps.eef_policy_actions import (
            EefPolicyWorkspaceRejected,
        )
        from galaxea_a1_runtime.hardware.eef_ik import A1EefIkTargetRejected

        if isinstance(exc, A1EefIkTargetRejected):
            kind = "ik_target_rejected"
            boundary = "IK"
        elif isinstance(exc, EefPolicyWorkspaceRejected):
            kind = "workspace_target_rejected"
            boundary = "workspace"
        else:
            raise
        bridge.live_status.break_line()
        message = (
            f"Policy target was rejected by the tracked {boundary} bounds; this attempt "
            f"ended safely. {exc}"
        )
        warning(message)
        outcome = (kind, message)
    finally:
        bridge.close()
    if outcome is not None and record_outcome is not None:
        record_outcome(*outcome)
    return 0
