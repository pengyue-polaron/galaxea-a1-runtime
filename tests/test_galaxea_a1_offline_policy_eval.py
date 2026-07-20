from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from galaxea_a1_runtime.evaluation.metrics import case_metrics, even_indices
from galaxea_a1_runtime.evaluation.eef_policy_io import (
    pi05_observation,
    validated_pi05_actions,
)
from galaxea_a1_runtime.evaluation.offline_config import (
    load_offline_eval_config,
)
from galaxea_a1_runtime.evaluation.types import EpisodeRecord
from galaxea_a1_runtime.evaluation.teacher_forcing import (
    action_step_records,
    complete_block_starts,
    summarize_teacher_forcing,
)
from galaxea_a1_runtime.apps.eef_policy_offline import (
    _lingbot_teacher_forced_cases,
)


REPO = Path(__file__).resolve().parents[1]


def test_offline_evaluation_config_is_strict_and_hardware_free():
    config = load_offline_eval_config(
        REPO / "configs/evaluation/fruit_placement_offline.toml", repo_root=REPO
    )

    assert config.dataset_root == REPO / "data/processed/fruit_placement_eef_v21"
    assert config.coverage.lingbot_first_frame_episodes_per_task == 26
    assert config.coverage.pi05_frames_per_episode == 3
    assert config.teacher_forcing_episode_index == 0


def test_offline_evaluation_requires_tracked_teacher_forcing_selection(tmp_path):
    text = (REPO / "configs/evaluation/fruit_placement_offline.toml").read_text()
    text = text.replace("\n[teacher_forcing]\nepisode_index = 0\n", "\n")
    path = tmp_path / "offline.toml"
    path.write_text(text)

    with pytest.raises(ValueError, match="teacher_forcing"):
        load_offline_eval_config(path, repo_root=REPO)


def test_even_indices_include_episode_boundaries():
    assert even_indices(99, 3) == [0, 49, 99]


def test_offline_action_metrics_handle_quaternion_sign_and_workspace():
    state = np.asarray(
        [[0.1, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0, 0, 0, 0, 0, 0, 0, 0.5]],
        dtype=np.float32,
    )
    target = np.asarray([[0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5]])
    prediction = target.copy()
    prediction[0, 3:7] *= -1.0
    episode = EpisodeRecord(0, 0, "task", state, target.astype(np.float32), np.zeros(1))

    result = case_metrics(
        model="test",
        scope="first",
        episode=episode,
        frame_index=0,
        prediction=prediction,
        target=target,
        latency_s=0.1,
        xyz_min=(0.06, -0.27, 0.06),
        xyz_max=(0.44, 0.14, 0.50),
        min_quat_norm=0.25,
    )

    assert result["xyz_error_m"]["max"] == 0.0
    assert result["quaternion_angle_deg"]["max"] == 0.0
    assert result["runtime_rejected_steps"] == 0


def test_offline_action_metrics_count_all_runtime_rejections():
    state = np.asarray(
        [[0.1, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0, 0, 0, 0, 0, 0, 0, 0.5]],
        dtype=np.float32,
    )
    target = np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]])
    prediction = target.copy()
    prediction[0, 0] = 1.0
    prediction[0, 7] = 1.01
    episode = EpisodeRecord(0, 0, "task", state, target.astype(np.float32), np.zeros(1))

    bounds_rejected = case_metrics(
        model="test",
        scope="first",
        episode=episode,
        frame_index=0,
        prediction=prediction,
        target=target,
        latency_s=0.1,
        xyz_min=(0.06, -0.27, 0.06),
        xyz_max=(0.44, 0.14, 0.50),
        min_quat_norm=0.25,
    )
    prediction[0, 3:7] = 0.0
    rejected = case_metrics(
        model="test",
        scope="first",
        episode=episode,
        frame_index=0,
        prediction=prediction,
        target=target,
        latency_s=0.1,
        xyz_min=(0.06, -0.27, 0.06),
        xyz_max=(0.44, 0.14, 0.50),
        min_quat_norm=0.25,
    )

    assert bounds_rejected["raw_workspace_violation_steps"] == 1
    assert bounds_rejected["raw_gripper_violation_steps"] == 1
    assert bounds_rejected["runtime_rejected_steps"] == 1
    assert rejected["runtime_rejected_steps"] == 1


def test_lingbot_teacher_forcing_does_not_count_the_first_inference_twice():
    states = np.zeros((9, 14), dtype=np.float32)
    states[:, 6] = 1.0
    actions = np.zeros((9, 8), dtype=np.float32)
    actions[:, 6] = 1.0
    episode = EpisodeRecord(0, 0, "task", states, actions, np.arange(9))
    first = np.zeros((8, 4, 4), dtype=np.float32)
    first[6] = 1.0
    replay = first.copy()
    images = {
        index: {
            "observation.images.front": np.zeros((2, 2, 3), dtype=np.uint8),
            "observation.images.wrist": np.zeros((2, 2, 3), dtype=np.uint8),
        }
        for index in range(9)
    }
    deployment = SimpleNamespace(
        observations=SimpleNamespace(front_key="front", wrist_key="wrist"),
        policy_server=SimpleNamespace(
            model_action_dim=30,
            action_channel_ids=tuple(range(8)),
            frame_chunk_size=4,
            action_per_frame=4,
        ),
        system=SimpleNamespace(
            eef=SimpleNamespace(
                xyz_min=(0.0, -1.0, 0.0),
                xyz_max=(1.0, 1.0, 1.0),
                min_quat_norm=0.25,
            )
        ),
    )

    class Client:
        def __init__(self) -> None:
            self.requests = []

        def infer(self, request):
            self.requests.append(request)
            return {} if request.get("compute_kv_cache") else {"action": replay}

    client = Client()
    cases = _lingbot_teacher_forced_cases(
        client=client,
        deployment=deployment,
        episode=episode,
        images=images,
        first=first,
        chunks=2,
    )

    assert len(cases) == 1
    assert cases[0]["scope"] == "teacher_forced_cache_replay"
    assert cases[0]["frame_index"] == 4
    assert len(client.requests) == 3


def test_sequential_teacher_forcing_records_each_next_action_distance():
    states = np.zeros((4, 14), dtype=np.float32)
    states[:, 6] = 1.0
    states[:, 0] = [0.1, 0.11, 0.12, 0.13]
    actions = np.zeros((4, 8), dtype=np.float32)
    actions[:, 6] = 1.0
    actions[:, 0] = [0.01, 0.02, 0.03, 0.04]
    episode = EpisodeRecord(0, 0, "task", states, actions, np.arange(4) / 30.0)
    prediction = actions.copy()
    prediction[:, 0] += 0.001
    prediction[:, 3:7] *= -1.0

    records = action_step_records(
        model="test",
        episode=episode,
        action_start=0,
        prediction=prediction,
        target=actions,
        inference_index=0,
    )
    result = summarize_teacher_forcing(
        episode=episode,
        steps=records,
        inferences=[{"latency_s": 0.1}],
        unique_ground_truth_observation_steps=4,
        ground_truth_history_observation_inputs=4,
        ground_truth_action_input_steps=4,
        xyz_min=(0.0, -1.0, 0.0),
        xyz_max=(1.0, 1.0, 1.0),
        min_quat_norm=0.25,
    )

    assert complete_block_starts(4, 4) == (0,)
    assert [item["action_index"] for item in records] == [0, 1, 2, 3]
    assert records[0]["ground_truth_next_action"] == pytest.approx(actions[0])
    assert records[0]["predicted_to_ground_truth_xyz_distance_m"] == pytest.approx(
        0.001
    )
    assert records[0]["quaternion_angle_deg"] == pytest.approx(0.0)
    assert result["predicted_action_steps"] == 4
    assert result["ground_truth_action_input_steps"] == 4
    assert result["raw_workspace_violation_steps"] == 0
    assert result["raw_gripper_violation_steps"] == 0
    assert result["quaternion_below_min_norm_steps"] == 0


def test_sequential_teacher_forcing_rejects_partial_model_blocks():
    with pytest.raises(ValueError, match="divisible"):
        complete_block_starts(5, 4)


def test_shared_pi05_offline_adapter_uses_exact_ground_truth_state_and_shape():
    states = np.zeros((1, 14), dtype=np.float32)
    states[0, 6] = 1.0
    actions = np.zeros((1, 8), dtype=np.float32)
    actions[0, 6] = 1.0
    episode = EpisodeRecord(0, 0, "task", states, actions, np.zeros(1))
    deployment = SimpleNamespace(
        observations=SimpleNamespace(front_key="front", wrist_key="wrist"),
        model_contract=SimpleNamespace(action_horizon=10, source_action_dim=8),
    )
    images = {
        "observation.images.front": np.zeros((2, 2, 3), dtype=np.uint8),
        "observation.images.wrist": np.ones((2, 2, 3), dtype=np.uint8),
    }

    packet = pi05_observation(deployment, episode, images, 0)
    predicted = validated_pi05_actions(
        {"actions": np.zeros((10, 8), dtype=np.float32)}, deployment
    )

    assert packet["observation/state"] == pytest.approx(states[0])
    assert packet["prompt"] == "task"
    assert predicted.shape == (10, 8)
    with pytest.raises(RuntimeError, match="expected finite"):
        validated_pi05_actions({"actions": np.zeros((9, 8))}, deployment)
