"""Bounded numerical IK with hard pose tolerances and a joint-clearance cost.

This computes one endpoint, not a time-parameterized trajectory. SciPy's
L-BFGS-B initializes a feasible pose region; SLSQP then prefers a nearby posture
with clearance from joint limits. All output is independently checked with FK.
"""

from __future__ import annotations

import time

import numpy as np

from galaxea_a1_runtime.configuration.system import ConstrainedIkConfig
from galaxea_a1_runtime.hardware.eef_ik import (
    A1EefIkSolver,
    A1EefIkTargetRejected,
    IkSolution,
    _finite_vector,
    _quat_to_matrix,
    _rotation_vector,
)


class ConstrainedIkSolver(A1EefIkSolver):
    def __init__(self, *, config: ConstrainedIkConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        from scipy.optimize import minimize

        self.config = config
        self._minimize = minimize

    def solve(
        self,
        current_joint_positions,
        target_xyz,
        target_quat_xyzw,
        *,
        max_joint_delta_rad=None,
        previous_joint_target=None,
    ) -> IkSolution:
        start = _finite_vector(current_joint_positions, len(self.joints), "IK feedback")
        xyz = _finite_vector(target_xyz, 3, "target xyz")
        rotation = _quat_to_matrix(target_quat_xyzw)
        if np.any(start < self.lower_limits) or np.any(start > self.upper_limits):
            raise ValueError("current joint positions violate tracked limits")
        delta = self.max_solution_delta_rad
        if max_joint_delta_rad is not None:
            if not np.isfinite(max_joint_delta_rad) or max_joint_delta_rad <= 0:
                raise ValueError(
                    "IK joint delta restriction must be finite and positive"
                )
            delta = min(delta, max_joint_delta_rad)
        lower = np.maximum(self.lower_limits, start - delta)
        upper = np.minimum(self.upper_limits, start + delta)
        # Projection applies only to a numerical initial guess. Feedback and
        # solved output are never clipped; no projected guess is published.
        initial = start.copy()
        seed_source = "feedback"
        if previous_joint_target is not None:
            initial = np.clip(
                _finite_vector(
                    previous_joint_target, len(self.joints), "previous IK target"
                ),
                lower,
                upper,
            )
            seed_source = "previous_target"
        began = time.monotonic()
        deadline = began + self.config.timeout_s
        position_scale = self.position_tolerance_m * self.config.tolerance_scale
        rotation_scale = self.orientation_tolerance_rad * self.config.tolerance_scale
        cached_q = None
        cached_pose = None

        def check_deadline():
            if time.monotonic() >= deadline:
                raise A1EefIkTargetRejected("Constrained IK exceeded its solve timeout")

        def pose(q):
            nonlocal cached_q, cached_pose
            check_deadline()
            if cached_q is not None and np.array_equal(q, cached_q):
                return cached_pose
            transform, origins, axes = self._kinematics(q)
            dp = xyz - transform[:3, 3]
            dr = _rotation_vector(rotation @ transform[:3, :3].T)
            jacobian = np.array(
                [
                    np.r_[np.cross(axis, transform[:3, 3] - origin), axis]
                    for origin, axis in zip(origins, axes, strict=True)
                ]
            ).T
            # d ||log(R_target R(q)^T)||² / dq = -2 log(R)^T J_angular.
            values = np.array(
                [
                    1 - np.dot(dp, dp) / position_scale**2,
                    1 - np.dot(dr, dr) / rotation_scale**2,
                ]
            )
            gradients = np.array(
                [
                    2 * dp @ jacobian[:3] / position_scale**2,
                    2 * dr @ jacobian[3:] / rotation_scale**2,
                ]
            )
            cached_q = q.copy()
            cached_pose = values, gradients
            return cached_pose

        def feasibility(q):
            values, gradients = pose(q)
            residual = 1 - values
            return float(residual @ residual), -2 * residual @ gradients

        def posture(q):
            check_deadline()
            margin = self.config.limit_margin_rad
            low = np.maximum(margin - (q - self.lower_limits), 0)
            high = np.maximum(margin - (self.upper_limits - q), 0)
            weight = self.config.limit_weight
            displacement = q - start
            cost = (
                displacement @ displacement
                + weight * (low @ low + high @ high) / margin**2
            )
            gradient = 2 * displacement + 2 * weight * (high - low) / margin**2
            return float(cost), gradient

        bounds = list(zip(lower, upper, strict=True))
        warm = self._minimize(
            feasibility,
            initial,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": self.max_iterations,
                "ftol": self.config.optimizer_tolerance,
                "gtol": self.config.optimizer_tolerance,
            },
        )
        initial = _finite_vector(warm.x, len(self.joints), "IK numerical initializer")
        result = self._minimize(
            posture,
            initial,
            jac=True,
            method="SLSQP",
            bounds=bounds,
            constraints=[
                {
                    "type": "ineq",
                    "fun": lambda q: pose(q)[0],
                    "jac": lambda q: pose(q)[1],
                }
            ],
            options={
                "maxiter": self.max_iterations,
                "ftol": self.config.optimizer_tolerance,
            },
        )
        check_deadline()
        q = _finite_vector(result.x, len(self.joints), "constrained IK solution")
        transform, _, _ = self._kinematics(q)
        pe = float(np.linalg.norm(xyz - transform[:3, 3]))
        re = float(np.linalg.norm(_rotation_vector(rotation @ transform[:3, :3].T)))
        max_delta = float(np.max(abs(q - start)))
        # The posture optimum is a preference, not an extra pose acceptance
        # requirement. An iteration/line-search stop may leave a valid endpoint;
        # independently verified constraints remain authoritative. Record status.
        if (
            np.any(q < lower)
            or np.any(q > upper)
            or max_delta > delta
            or pe > self.position_tolerance_m
            or re > self.orientation_tolerance_rad
        ):
            raise A1EefIkTargetRejected(
                "Constrained IK rejected: "
                f"status={result.message}, position_error_m={pe:.6f}, "
                f"orientation_error_rad={re:.6f}, max_joint_delta_rad={max_delta:.6f}"
            )
        return IkSolution(
            joint_positions=tuple(float(x) for x in q),
            iterations=int(warm.nit + result.nit),
            position_error_m=pe,
            orientation_error_rad=re,
            max_joint_delta_rad=max_delta,
            backend="constrained",
            solve_time_s=time.monotonic() - began,
            minimum_joint_margin_rad=float(
                np.min(np.minimum(q - self.lower_limits, self.upper_limits - q))
            ),
            seed_source=seed_source,
            optimization_status=str(result.message),
        )
