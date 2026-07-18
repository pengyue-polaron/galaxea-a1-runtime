from pathlib import Path

import numpy as np
import pytest

from galaxea_a1_runtime.configuration.system import load_system_config
from galaxea_a1_runtime.hardware.eef_ik import A1EefIkSolver, build_eef_ik_solver


REPO = Path(__file__).resolve().parents[1]
SYSTEM = REPO / "configs/system/a1.toml"
RESET_JOINTS = np.asarray(
    [
        0.012999534606933594,
        0.0010004043579101562,
        -0.07400035858154297,
        -1.569000244140625,
        0.33699989318847656,
        -0.048999786376953125,
    ]
)


def test_tracked_a1_ik_reaches_cartesian_target_with_named_joint_limits():
    system = load_system_config(SYSTEM, repo_root=REPO)
    solver = build_eef_ik_solver(system)
    start_xyz, start_quat = solver.forward(RESET_JOINTS)

    assert system.eef_ik.position_tolerance_m == pytest.approx(0.002)
    assert system.eef_ik.orientation_tolerance_rad == pytest.approx(0.02)
    assert system.eef_ik.max_solution_delta_rad == pytest.approx(1.50)
    solution = solver.solve(
        RESET_JOINTS,
        start_xyz + np.asarray([0.03, 0.0, 0.0]),
        start_quat,
    )
    achieved_xyz, achieved_quat = solver.forward(solution.joint_positions)

    assert solver.joint_names == system.joint_safety.names
    assert achieved_xyz == pytest.approx(
        start_xyz + [0.03, 0.0, 0.0], abs=system.eef_ik.position_tolerance_m
    )
    assert abs(float(np.dot(achieved_quat, start_quat))) == pytest.approx(1.0, abs=1e-4)
    joints = np.asarray(solution.joint_positions)
    assert np.all(joints >= np.asarray(system.joint_safety.lower_limits))
    assert np.all(joints <= np.asarray(system.joint_safety.upper_limits))
    assert solution.max_joint_delta_rad <= system.eef_ik.max_solution_delta_rad


def test_a1_ik_rejects_solution_beyond_tracked_delta_limit():
    system = load_system_config(SYSTEM, repo_root=REPO)
    settings = system.eef_ik
    solver = A1EefIkSolver(
        urdf_path=settings.urdf,
        joint_names=system.joint_safety.names,
        lower_limits=system.joint_safety.lower_limits,
        upper_limits=system.joint_safety.upper_limits,
        max_iterations=settings.max_iterations,
        damping=settings.damping,
        orientation_weight=settings.orientation_weight,
        max_iteration_step_rad=settings.max_iteration_step_rad,
        position_tolerance_m=settings.position_tolerance_m,
        orientation_tolerance_rad=settings.orientation_tolerance_rad,
        max_solution_delta_rad=0.01,
    )
    start_xyz, start_quat = solver.forward(RESET_JOINTS)

    with pytest.raises(RuntimeError, match="exceeds the configured joint delta"):
        solver.solve(RESET_JOINTS, start_xyz + [0.03, 0.0, 0.0], start_quat)
