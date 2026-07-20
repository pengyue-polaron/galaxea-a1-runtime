"""Shared presentation and pure validation helpers for runtime doctors."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from galaxea_a1_runtime.console import padded_label
from galaxea_a1_runtime.constants import ARM_JOINT_COUNT, IDLE_TIMEOUT_CODE
from galaxea_a1_runtime.runtime.relay import decode_relay_status
from galaxea_a1_runtime.safety import validate_arm_control_command


@dataclass(frozen=True)
class Check:
    name: str
    level: str
    detail: str


def arm_control_result(
    message: Any, *, arm_joints: int, allowed_modes: tuple[int, ...]
) -> tuple[bool, str]:
    """Apply the relay's complete staged-command contract in a doctor."""

    validate_arm_control_command(
        p_des=message.p_des,
        v_des=message.v_des,
        kp=message.kp,
        kd=message.kd,
        t_ff=message.t_ff,
        mode=message.mode,
        arm_joints=arm_joints,
        allowed_modes=allowed_modes,
    )
    return True, f"joints={arm_joints} mode={message.mode}"


class RosDoctorContext:
    """Shared ROS topic/node probes used by app-specific doctors."""

    def __init__(
        self,
        *,
        rospy: Any,
        rosnode: Any,
        checks: list[Check],
        timeout_s: float,
        required: bool,
    ) -> None:
        self.rospy = rospy
        self.rosnode = rosnode
        self.checks = checks
        self.timeout_s = timeout_s
        self.required = required
        self.published = dict(rospy.get_published_topics())
        self.nodes = set(rosnode.get_node_names())

    def message(
        self,
        name: str,
        topic: str,
        message_type: Any,
        validator: Callable[[Any], tuple[bool, str]],
        *,
        required: bool | None = None,
    ) -> Any | None:
        is_required = self.required if required is None else required
        message = self._wait_for_message(
            name, topic, message_type, required=is_required
        )
        if message is None:
            return None
        try:
            ok, detail = validator(message)
            add_check(self.checks, name, ok, detail, required=is_required)
            return message
        except Exception as exc:  # Message validation errors are checks.
            add_check(
                self.checks,
                name,
                False,
                f"{topic}: {exc}",
                required=is_required,
            )
            return None

    def motor_status(
        self, topic: str, message_type: Any, *, required: bool | None = None
    ) -> Any | None:
        is_required = self.required if required is None else required
        message = self._wait_for_message(
            "motor_status", topic, message_type, required=is_required
        )
        if message is not None:
            try:
                level, detail = motor_status_level(message)
            except Exception as exc:
                add_check(
                    self.checks,
                    "motor_status",
                    False,
                    f"{topic}: {exc}",
                    required=is_required,
                )
                return None
            if level == "FAIL" and not is_required:
                level = "WARN"
            add_level(self.checks, "motor_status", level, detail)
        return message

    def _wait_for_message(
        self,
        name: str,
        topic: str,
        message_type: Any,
        *,
        required: bool,
    ) -> Any | None:
        if topic not in self.published:
            add_check(
                self.checks,
                name,
                False,
                f"{topic} not published",
                required=required,
            )
            return None
        try:
            return self.rospy.wait_for_message(
                topic, message_type, timeout=self.timeout_s
            )
        except Exception as exc:  # ROS transport and decode errors are checks.
            add_check(
                self.checks,
                name,
                False,
                f"{topic}: {exc}",
                required=required,
            )
            return None

    def node(self, check_name: str, node_name: str) -> bool:
        try:
            alive = node_name in self.nodes and bool(
                self.rosnode.rosnode_ping(node_name, max_count=1, verbose=False)
            )
        except Exception:
            alive = False
        add_check(
            self.checks,
            check_name,
            alive,
            f"{node_name} responds to XML-RPC"
            if alive
            else f"{node_name} missing or stale",
            required=self.required,
        )
        return alive


def add_check(
    checks: list[Check],
    name: str,
    ok: bool,
    detail: str,
    *,
    required: bool = True,
) -> None:
    level = "PASS" if ok else ("FAIL" if required else "WARN")
    checks.append(Check(name, level, detail))


def add_level(checks: list[Check], name: str, level: str, detail: str) -> None:
    checks.append(Check(name, level, detail))


def checks_to_json(checks: list[Check]) -> str:
    return json.dumps([asdict(item) for item in checks], indent=2, sort_keys=True)


def checks_exit_code(checks: list[Check]) -> int:
    return 1 if any(item.level == "FAIL" for item in checks) else 0


def print_checks(checks: list[Check]) -> None:
    width = max((len(item.name) for item in checks), default=0)
    for item in checks:
        print(f"{padded_label(item.level)} {item.name:<{width}}  {item.detail}")


def finish_checks(checks: list[Check], *, json_output: bool) -> int:
    if json_output:
        print(checks_to_json(checks))
    else:
        print_checks(checks)
    return checks_exit_code(checks)


def motor_status_level(msg: Any) -> tuple[str, str]:
    names = list(msg.data.name)
    errors = list(msg.data.motor_errors)
    actuator_count = ARM_JOINT_COUNT + 1
    if len(errors) < actuator_count:
        return (
            "FAIL",
            f"motor status has {len(errors)} entries, need {actuator_count}",
        )
    rows: list[str] = []
    bad: list[tuple[str, int]] = []
    idle_timeout: list[str] = []
    for index, item in enumerate(errors):
        name = names[index] if index < len(names) else f"motor{index + 1}"
        code = int(item.error_code)
        desc = "; ".join(str(part) for part in item.error_description) or "OK"
        rows.append(f"{index + 1}:{name}=code{code}({desc})")
        if index < actuator_count and code == IDLE_TIMEOUT_CODE:
            idle_timeout.append(name)
        elif index < actuator_count and code != 0:
            bad.append((name, code))
    detail = "; ".join(rows)
    if bad:
        return "FAIL", detail
    if idle_timeout:
        return "WARN", "idle ECU->ACU timeout treated as non-blocking; " + detail
    return "PASS", detail


def relay_status_result(msg: Any, *, require_execution: bool) -> tuple[bool, str]:
    status = decode_relay_status(str(getattr(msg, "data", "")))
    if not status.valid or status.state == "FAULT":
        return False, f"{status.state}: {status.reason}"
    allowed = (
        {"LOCKED", "ACTIVE"} if require_execution else {"LOCKED", "ARMING", "ACTIVE"}
    )
    return status.state in allowed, f"{status.state}: {status.reason}"
