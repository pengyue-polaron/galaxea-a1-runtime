import numpy as np
import pytest

from galaxea_a1_runtime.apps.lingbot.rollout import LingBotActionChunk


def test_first_lingbot_chunk_skips_execution_frame_zero_but_keeps_it_in_cache():
    values = np.arange(8 * 3 * 8, dtype=np.float32).reshape(8, 3, 8)

    chunk = LingBotActionChunk.from_response(
        values, first=True, execute_frames=1, observations_per_frame=4
    )

    steps = list(chunk.steps())
    assert chunk.start_frame == 1
    assert chunk.end_frame == 2
    assert chunk.cache_state.shape == (8, 2, 8)
    assert steps[0][:3] == (1, 0, 1)
    assert chunk.total_steps == 8
    assert chunk.needs_observation_after(1)


def test_subsequent_lingbot_chunk_cache_starts_at_execution_frame_zero():
    values = np.zeros((8, 3, 8), dtype=np.float32)

    chunk = LingBotActionChunk.from_response(
        values, first=False, execute_frames=2, observations_per_frame=4
    )

    steps = list(chunk.steps())
    assert chunk.cache_state.shape == (8, 2, 8)
    assert steps[0][:3] == (0, 0, 0)
    assert steps[-1][:3] == (1, 7, 1)


def test_lingbot_chunk_rejects_incompatible_action_shape_and_horizon():
    with pytest.raises(RuntimeError, match="shape"):
        LingBotActionChunk.from_response(
            np.zeros((7, 3, 8)),
            first=False,
            execute_frames=1,
            observations_per_frame=4,
        )
    with pytest.raises(RuntimeError, match="divisible"):
        LingBotActionChunk.from_response(
            np.zeros((8, 3, 7)),
            first=False,
            execute_frames=1,
            observations_per_frame=4,
        )


def test_two_frame_lingbot_execution_skips_only_the_conditioned_first_frame():
    values = np.zeros((8, 4, 4), dtype=np.float32)

    first = LingBotActionChunk.from_response(
        values, first=True, execute_frames=2, observations_per_frame=4
    )
    subsequent = LingBotActionChunk.from_response(
        values, first=False, execute_frames=2, observations_per_frame=4
    )

    assert first.cache_state.shape == (8, 3, 4)
    assert first.total_steps == 8
    assert subsequent.cache_state.shape == (8, 2, 4)
    assert subsequent.total_steps == 8
