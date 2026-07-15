"""Pure validation of ACT joint-policy output before execution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ActActionValidator:
    joint_names: tuple[str, ...]
    lower_limits: np.ndarray
    upper_limits: np.ndarray
    execute_steps: int
    step_guard_enabled: bool
    max_first_delta_rad: float
    max_step_rad: float

    def validate(
        self, chunk: np.ndarray, current_joints: tuple[float, ...]
    ) -> np.ndarray:
        dof = len(self.joint_names)
        if chunk.ndim != 2 or chunk.shape[1] != dof + 1:
            raise RuntimeError(f"invalid chunk shape: {chunk.shape}")
        if not np.all(np.isfinite(chunk)):
            raise RuntimeError("ACT chunk contains non-finite values")
        if len(current_joints) != dof or not np.all(np.isfinite(current_joints)):
            raise RuntimeError(
                f"current joint feedback must contain {dof} finite values"
            )
        steps = chunk[: min(self.execute_steps, len(chunk))].copy()
        if not len(steps):
            raise RuntimeError("ACT chunk contains no executable steps")
        previous = np.asarray(current_joints, dtype=np.float64)
        for index, row in enumerate(steps):
            joints = row[:dof]
            if row[dof] < 0.0 or row[dof] > 1.0:
                raise RuntimeError(
                    f"ACT target {index} gripper={row[dof]:.4f} is outside [0, 1]"
                )
            violations = self._joint_limit_violations(joints)
            if violations:
                raise RuntimeError(
                    f"ACT target {index} violates joint limits: "
                    + "; ".join(violations)
                )
            if self.step_guard_enabled:
                step = float(np.max(np.abs(joints - previous)))
                limit = self.max_first_delta_rad if index == 0 else self.max_step_rad
                if step > limit:
                    raise RuntimeError(
                        f"ACT target {index} step={step:.4f} rad exceeds limit={limit:.4f}"
                    )
            previous = joints
        return steps

    def _joint_limit_violations(self, joints: np.ndarray) -> list[str]:
        violations: list[str] = []
        for name, value, lower, upper in zip(
            self.joint_names,
            joints,
            self.lower_limits,
            self.upper_limits,
            strict=True,
        ):
            value = float(value)
            if value < float(lower) or value > float(upper):
                violations.append(
                    f"{name}={value:.4f} outside "
                    f"[{float(lower):.4f}, {float(upper):.4f}] "
                    f"(target={np.round(joints, 4).tolist()})"
                )
        return violations
