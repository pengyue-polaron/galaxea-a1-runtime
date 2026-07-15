from dataclasses import replace

import numpy as np

from galaxea_a1_runtime.lerobot.boundary_trim import decide_episode_bounds
from galaxea_a1_runtime.lerobot.boundary_trim_config import BoundaryTrimConfig
from galaxea_a1_runtime.schema import DEFAULT_STATE_NAMES, JOINT_ACTION_NAMES


def trim_config(**overrides) -> BoundaryTrimConfig:
    config = BoundaryTrimConfig(
        enabled=True,
        anchor_window_s=0.5,
        joint_deadband_rad=0.01,
        gripper_deadband=0.01,
        confirm_frames=5,
        pre_roll_s=0.5,
        post_roll_s=0.75,
        max_trim_fraction=0.5,
        min_kept_duration_s=5.0,
    )
    return replace(config, **overrides)


def vectors(frame_count: int = 300) -> tuple[np.ndarray, np.ndarray]:
    actions = np.zeros((frame_count, len(JOINT_ACTION_NAMES)))
    states = np.zeros((frame_count, len(DEFAULT_STATE_NAMES)))
    states[:, 6] = 1.0
    return actions, states


def decide(
    actions: np.ndarray,
    states: np.ndarray,
    *,
    config: BoundaryTrimConfig | None = None,
):
    return decide_episode_bounds(
        actions=actions,
        states=states,
        action_names=JOINT_ACTION_NAMES,
        state_names=DEFAULT_STATE_NAMES,
        fps=10,
        config=config or trim_config(),
    )


def test_trims_only_stable_boundaries_with_rolls():
    actions, states = vectors()
    actions[60:, 0] = 0.2
    actions[220:, 0] = 0.4
    states[65:, 7] = 0.2
    states[230:, 7] = 0.4

    result = decide(actions, states)

    assert (result.start, result.end) == (55, 238)
    assert result.start_anchor_stable
    assert result.end_action_anchor_stable
    assert result.end_state_anchor_stable
    assert result.guard_reason is None


def test_gripper_only_motion_is_not_mistaken_for_idle():
    actions, states = vectors()
    actions[60:, -1] = 0.3
    actions[220:, -1] = 0.8
    states[65:, -1] = 0.3
    states[230:, -1] = 0.8

    result = decide(actions, states)

    assert (result.start, result.end) == (55, 238)


def test_internal_pause_is_preserved():
    actions, states = vectors()
    actions[60:120, 0] = 0.2
    actions[120:170, 0] = 0.0
    actions[170:220, 0] = 0.3
    actions[220:, 0] = 0.4
    states[:, 7] = actions[:, 0]

    result = decide(actions, states)

    assert result.start < 120
    assert result.end > 170


def test_unstable_start_anchor_disables_prefix_trim():
    actions, states = vectors()
    actions[:5, 0] = np.linspace(0.0, 0.02, 5)
    actions[60:, 0] = 0.2
    actions[220:, 0] = 0.4
    states[65:, 7] = 0.2
    states[230:, 7] = 0.4

    result = decide(actions, states)

    assert result.start == 0
    assert not result.start_anchor_stable
    assert result.guard_reason == "no_confirmed_task_motion"


def test_unstable_final_action_anchor_disables_suffix_trim():
    actions, states = vectors()
    actions[60:, 0] = 0.2
    actions[-5:, 0] = np.linspace(0.2, 0.22, 5)
    states[65:, 7] = 0.2

    result = decide(actions, states)

    assert result.start == 55
    assert result.end == len(actions)
    assert not result.end_action_anchor_stable


def test_unstable_final_feedback_anchor_disables_suffix_trim():
    actions, states = vectors()
    actions[60:, 0] = 0.2
    states[65:, 7] = 0.2
    states[-5:, 7] = np.linspace(0.2, 0.22, 5)

    result = decide(actions, states)

    assert result.start == 55
    assert result.end == len(actions)
    assert not result.end_state_anchor_stable
    assert result.end_reason == "unstable_feedback_anchor"


def test_guard_preserves_episode_when_trim_is_too_large():
    actions, states = vectors()
    actions[100:, 0] = 0.2
    actions[200:, 0] = 0.4
    states[:, 7] = actions[:, 0]

    result = decide(
        actions,
        states,
        config=trim_config(max_trim_fraction=0.1),
    )

    assert (result.start, result.end) == (0, len(actions))
    assert result.guard_reason == "maximum_trim_fraction"


def test_no_motion_and_disabled_policy_preserve_the_episode():
    actions, states = vectors()

    no_motion = decide(actions, states)
    disabled = decide(actions, states, config=trim_config(enabled=False))

    assert (no_motion.start, no_motion.end) == (0, len(actions))
    assert no_motion.guard_reason == "no_confirmed_task_motion"
    assert (disabled.start, disabled.end) == (0, len(actions))
    assert disabled.start_reason == "disabled"
