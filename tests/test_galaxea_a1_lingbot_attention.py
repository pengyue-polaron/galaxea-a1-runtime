from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from galaxea_a1_runtime.apps.lingbot.attention import (
    LingBotAttentionCapture,
    attention_overlay,
    cache_aware_attention_rollout,
    decoded_temporal_alignment,
    split_paired_camera_token_map,
    visual_token_frames,
    write_attention_audit,
)


def test_paired_camera_tokens_are_split_by_latent_width_not_vector_halves():
    front, wrist = split_paired_camera_token_map(
        np.arange(16),
        grid_height=2,
        grid_width_per_camera=4,
    )

    assert front.tolist() == [[0, 1, 2, 3], [8, 9, 10, 11]]
    assert wrist.tolist() == [[4, 5, 6, 7], [12, 13, 14, 15]]


def test_causal_vae_latent_frames_align_to_decoded_rgb_anchors():
    alignment = decoded_temporal_alignment(latent_frames=4, temporal_scale=4)

    assert alignment.decoded_frame_count == 13
    assert alignment.anchor_indices == (0, 4, 8, 12)


def test_visual_cache_groups_are_split_by_time_and_classified():
    ids = np.array([0] * 8 + [1] * 3 + [2] * 4 + [3] * 6)
    predicted = np.array([False] * 8 + [False] * 3 + [True] * 4 + [False] * 6)

    frames = visual_token_frames(
        ids,
        predicted,
        tokens_per_frame=4,
        excluded_cache_ids=frozenset({3}),
    )

    assert [
        (frame.cache_id, frame.frame_in_group, frame.is_predicted) for frame in frames
    ] == [(0, 0, False), (0, 1, False), (2, 0, True)]
    assert frames[1].positions.tolist() == [4, 5, 6, 7]


def test_cache_aware_rollout_propagates_later_query_relevance_to_earlier_target():
    identity = np.eye(2, dtype=np.float32)

    result = cache_aware_attention_rollout(
        [0.5 * identity, 0.5 * identity],
        [0.1 * identity, 0.2 * identity],
        selected_query_positions=np.array([0, 1]),
    )

    np.testing.assert_allclose(result, [0.125, 0.125])


def test_attention_capture_exposes_direct_and_prediction_mediated_wam_paths():
    attention = SimpleNamespace()
    attention.attn_op = lambda query, key, value: query
    transformer = SimpleNamespace(
        blocks=[SimpleNamespace(attn1=attention)],
    )
    capture = LingBotAttentionCapture(
        layers=(0,),
        frame_chunk_size=2,
        action_query_tokens=6,
        selected_action_query_tokens=3,
        grid_height=1,
        grid_width_per_camera=1,
        video_forward_calls=2,
        action_forward_calls=2,
    )
    capture.install(transformer)

    attention.attn_caches = {
        "pos": {
            "mask": torch.ones(8, dtype=torch.bool),
            "id": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
            "is_pred": torch.zeros(8, dtype=torch.bool),
        }
    }
    capture.begin()
    video_query = torch.ones((1, 4, 2, 3))
    video_key = torch.ones((1, 8, 2, 3))
    attention.attn_op(video_query, video_key, torch.zeros_like(video_key))
    attention.attn_op(video_query, video_key, torch.zeros_like(video_key))

    attention.attn_caches = {
        "pos": {
            "mask": torch.ones(14, dtype=torch.bool),
            "id": torch.tensor([0] * 4 + [1] * 4 + [2] * 6),
            "is_pred": torch.tensor([False] * 4 + [True] * 4 + [False] * 6),
        }
    }
    action_query = torch.ones((1, 6, 2, 3))
    action_key = torch.ones((1, 14, 2, 3))
    attention.attn_op(action_query, action_key, torch.zeros_like(action_key))
    attention.attn_op(action_query, action_key, torch.zeros_like(action_key))
    result = capture.finish()

    assert result["schema_version"] == 3
    assert result["kind"] == "wam_multistage_cache_aware_attention_rollout"
    assert result["token_layout"]["actual_history_frames"] == 2
    assert result["token_layout"]["predicted_future_frames"] == 2
    assert set(result["paths"]) == {
        "action_to_predicted_future",
        "predicted_future_to_actual_history",
        "action_to_actual_history",
        "action_via_predicted_future_to_actual_history",
    }
    predicted = result["paths"]["action_to_predicted_future"]["frames"]
    actual = result["paths"]["action_to_actual_history"]["frames"]
    future_to_history = result["paths"]["predicted_future_to_actual_history"][
        "source_frames"
    ]
    assert len(predicted) == 2
    assert len(actual) == 2
    assert len(future_to_history) == 2
    assert len(future_to_history[0]["history_frames"]) == 2
    assert actual[-1]["relative_to_latest"] == 0
    assert predicted[0]["maps"]["front"].shape == (1, 1)
    assert predicted[0]["maps"]["wrist"].shape == (1, 1)
    assert predicted[0]["attention_mass"]["front"] > 0


def test_attention_audit_atomically_writes_wam_sources_and_overlays(tmp_path: Path):
    actual = [
        {
            "front": np.full((24, 32, 3), 80 + index, dtype=np.uint8),
            "wrist": np.full((24, 32, 3), 120 + index, dtype=np.uint8),
        }
        for index in range(2)
    ]
    predicted = [
        {
            "front": np.full((24, 32, 3), 160 + index, dtype=np.uint8),
            "wrist": np.full((24, 32, 3), 200 + index, dtype=np.uint8),
        }
        for index in range(2)
    ]

    def frame(index_name: str, index: int) -> dict[str, object]:
        return {
            index_name: index,
            "maps": {
                "front": np.arange(4, dtype=np.float32).reshape(2, 2),
                "wrist": np.flipud(np.arange(4, dtype=np.float32).reshape(2, 2)),
            },
        }

    attention = {
        "schema_version": 3,
        "kind": "wam_multistage_cache_aware_attention_rollout",
        "paths": {
            "action_to_predicted_future": {
                "frames": [frame("future_index", index) for index in range(2)]
            },
            "predicted_future_to_actual_history": {
                "source_frames": [
                    {
                        "future_index": future,
                        "history_frames": [
                            frame("history_index", history) for history in range(2)
                        ],
                    }
                    for future in range(2)
                ]
            },
            "action_to_actual_history": {
                "frames": [frame("history_index", index) for index in range(2)]
            },
            "action_via_predicted_future_to_actual_history": {
                "frames": [frame("history_index", index) for index in range(2)]
            },
        },
    }

    output = write_attention_audit(
        tmp_path,
        attention=attention,
        actual_observations=actual,
        predicted_observations=predicted,
        front_key="front",
        wrist_key="wrist",
        prompt="put fruit into bowl",
    )

    assert (output / "attention.json").is_file()
    assert (output / "actual_history_00_front_input.png").is_file()
    assert (output / "predicted_future_01_wrist_input.png").is_file()
    assert (output / "action_to_actual_history_01_front_overlay.png").is_file()
    assert (
        output
        / "predicted_future_to_actual_history_future_01_history_00_wrist_overlay.png"
    ).is_file()
    assert not any(tmp_path.glob(".*.staging"))
    overlay = attention_overlay(
        actual[0]["front"],
        attention["paths"]["action_to_actual_history"]["frames"][0]["maps"]["front"],
    )
    assert overlay.shape == actual[0]["front"].shape
    assert overlay.dtype == np.uint8
