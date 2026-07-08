"""Dry-run runtime profile planner for Galaxea A1."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from galaxea_a1_runtime.config import RuntimeProfile
from galaxea_a1_runtime.teleop.config import default_config_path, load_teleop_config


class RuntimeStepKind(StrEnum):
    CHECK = "check"
    PENDING = "pending"
    START = "start"
    STOP = "stop"
    DOCTOR = "doctor"


@dataclass(frozen=True)
class RuntimeStep:
    kind: RuntimeStepKind
    name: str
    command: tuple[str, ...]
    touches_hardware: bool
    note: str = ""

    def shell_command(self) -> str:
        return " ".join(self.command)


@dataclass(frozen=True)
class RuntimePlan:
    profile: RuntimeProfile
    steps: tuple[RuntimeStep, ...]

    @property
    def touches_hardware(self) -> bool:
        return any(step.touches_hardware for step in self.steps)


def build_runtime_plan(
    *,
    repo_root: Path,
    profile: RuntimeProfile,
    serial: str = "/dev/a1",
) -> RuntimePlan:
    repo = str(repo_root)
    if profile == RuntimeProfile.STATIC:
        return RuntimePlan(
            profile=profile,
            steps=(
                RuntimeStep(
                    kind=RuntimeStepKind.DOCTOR,
                    name="static-doctor",
                    command=(".venv/bin/python", "-m", "galaxea_a1_runtime.cli", "doctor", "--repo-root", repo),
                    touches_hardware=False,
                ),
            ),
        )
    if profile == RuntimeProfile.SAFE:
        return RuntimePlan(
            profile=profile,
            steps=(
                RuntimeStep(
                    kind=RuntimeStepKind.CHECK,
                    name="serial-present",
                    command=("test", "-e", serial),
                    touches_hardware=False,
                    note="Fails closed if the A1 serial device is absent.",
                ),
                RuntimeStep(
                    kind=RuntimeStepKind.START,
                    name="safe-runtime",
                    command=("just", "a1-runtime", "services"),
                    touches_hardware=True,
                    note="Starts ROS master, driver, staged tracker, and locked relay.",
                ),
                RuntimeStep(
                    kind=RuntimeStepKind.DOCTOR,
                    name="execution-doctor",
                    command=("just", "a1-runtime", "doctor", "--require-execution"),
                    touches_hardware=True,
                    note="Use only after the arm is powered and positioned safely.",
                ),
            ),
        )
    if profile == RuntimeProfile.COLLECT:
        teleop_serial, leader_port, config_path = _teleop_plan_inputs(repo_root, fallback_serial=serial)
        return RuntimePlan(
            profile=profile,
            steps=(
                RuntimeStep(
                    kind=RuntimeStepKind.CHECK,
                    name="serial-present",
                    command=("test", "-e", teleop_serial),
                    touches_hardware=False,
                    note="Fails closed if the A1 serial device is absent.",
                ),
                RuntimeStep(
                    kind=RuntimeStepKind.CHECK,
                    name="leader-present",
                    command=("test", "-e", leader_port),
                    touches_hardware=False,
                    note=f"Configured in {config_path}.",
                ),
                RuntimeStep(
                    kind=RuntimeStepKind.START,
                    name="teleop-collection",
                    command=("just", "collect", "teleop", "<experiment>"),
                    touches_hardware=True,
                    note=f"Starts staged joint teleop using {config_path}.",
                ),
                RuntimeStep(
                    kind=RuntimeStepKind.DOCTOR,
                    name="teleop-doctor",
                    command=("just", "a1-teleop", "doctor"),
                    touches_hardware=False,
                    note="Static/import checks; use hardware checks only after power-on.",
                ),
            ),
        )
    if profile == RuntimeProfile.INFER:
        safe = build_runtime_plan(repo_root=repo_root, profile=RuntimeProfile.SAFE, serial=serial)
        return RuntimePlan(
            profile=profile,
            steps=safe.steps
            + (
                RuntimeStep(
                    kind=RuntimeStepKind.PENDING,
                    name="policy-runner-selection",
                    command=("just", "runtime", "profiles"),
                    touches_hardware=False,
                    note="Use a dedicated app runner such as just a1-lingbot start before enabling motion.",
                ),
            ),
        )
    if profile == RuntimeProfile.DIRECT_DEBUG:
        return RuntimePlan(
            profile=profile,
            steps=(
                RuntimeStep(
                    kind=RuntimeStepKind.STOP,
                    name="stop-safe-runtime",
                    command=("just", "a1-runtime", "stop"),
                    touches_hardware=True,
                    note="Avoid two trackers or relays fighting over command topics.",
                ),
                RuntimeStep(
                    kind=RuntimeStepKind.START,
                    name="direct-debug-tracker",
                    command=(
                        "roslaunch",
                        "/workspace/scripts/runtime/ee_tracker_staged.launch",
                        "staged_command_topic:=/arm_joint_command_host",
                    ),
                    touches_hardware=True,
                    note="Explicit hardware debug only; bypasses the relay.",
                ),
            ),
        )
    raise ValueError(f"unsupported runtime profile: {profile}")


def _teleop_plan_inputs(repo_root: Path, *, fallback_serial: str) -> tuple[str, str, str]:
    path = default_config_path(repo_root)
    try:
        config = load_teleop_config(path, repo_root=repo_root)
    except Exception:
        return fallback_serial, "/dev/ttyACM0", str(path)
    return config.host.a1_serial, config.leader.port, str(config.path)


def format_runtime_plan(plan: RuntimePlan) -> str:
    lines = [f"profile={plan.profile}", f"touches_hardware={str(plan.touches_hardware).lower()}"]
    for index, step in enumerate(plan.steps, start=1):
        suffix = f"  # {step.note}" if step.note else ""
        lines.append(f"{index}. [{step.kind}] {step.name}: {step.shell_command()}{suffix}")
    return "\n".join(lines)
