from pathlib import Path
from types import SimpleNamespace

import numpy as np

from galaxea_a1_runtime.evaluation.metrics import case_metrics, even_indices
from galaxea_a1_runtime.evaluation.offline_config import (
    load_offline_eval_config,
)
from galaxea_a1_runtime.evaluation.types import EpisodeRecord
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
    assert result["raw_output_rewrite_steps"] == 0


def test_offline_action_metrics_distinguish_rewrites_from_rejections():
    state = np.asarray(
        [[0.1, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0, 0, 0, 0, 0, 0, 0, 0.5]],
        dtype=np.float32,
    )
    target = np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]])
    prediction = target.copy()
    prediction[0, 0] = 1.0
    prediction[0, 7] = 1.01
    episode = EpisodeRecord(0, 0, "task", state, target.astype(np.float32), np.zeros(1))

    rewritten = case_metrics(
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

    assert rewritten["raw_output_rewrite_steps"] == 1
    assert rewritten["runtime_rejected_steps"] == 0
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
        policy_server=SimpleNamespace(frame_chunk_size=4, action_per_frame=4),
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
