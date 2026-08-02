"""Reduced LingBot WAM attention capture and visualization."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class VisualTokenFrame:
    positions: np.ndarray
    cache_id: int
    frame_in_group: int
    is_predicted: bool


@dataclass(frozen=True)
class DecodedTemporalAlignment:
    decoded_frame_count: int
    anchor_indices: tuple[int, ...]


def decoded_temporal_alignment(
    *,
    latent_frames: int,
    temporal_scale: int,
) -> DecodedTemporalAlignment:
    """Map causal VAE latent frames to their decoded RGB anchor indices."""

    if latent_frames <= 0 or temporal_scale <= 0:
        raise ValueError("VAE temporal alignment dimensions must be positive")
    return DecodedTemporalAlignment(
        decoded_frame_count=1 + (latent_frames - 1) * temporal_scale,
        anchor_indices=tuple(index * temporal_scale for index in range(latent_frames)),
    )


def split_paired_camera_token_map(
    values: np.ndarray,
    *,
    grid_height: int,
    grid_width_per_camera: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Undo the VAE's width-wise front/wrist latent concatenation."""

    vector = np.asarray(values)
    expected = 2 * grid_height * grid_width_per_camera
    if vector.ndim != 1 or vector.size != expected:
        raise ValueError(
            f"paired camera token map must contain {expected} values, "
            f"got {vector.shape}"
        )
    combined = vector.reshape(grid_height, 2 * grid_width_per_camera)
    return (
        combined[:, :grid_width_per_camera],
        combined[:, grid_width_per_camera:],
    )


def visual_token_frames(
    cache_ids: np.ndarray,
    is_predicted: np.ndarray,
    *,
    tokens_per_frame: int,
    excluded_cache_ids: frozenset[int] = frozenset(),
) -> tuple[VisualTokenFrame, ...]:
    """Decode visual cache groups into chronological frame token positions."""

    ids = np.asarray(cache_ids)
    predicted = np.asarray(is_predicted)
    if (
        ids.ndim != 1
        or predicted.ndim != 1
        or ids.shape != predicted.shape
        or tokens_per_frame <= 0
    ):
        raise ValueError("attention cache identity arrays are invalid")
    frames: list[VisualTokenFrame] = []
    for cache_id in sorted(int(value) for value in np.unique(ids) if value >= 0):
        if cache_id in excluded_cache_ids:
            continue
        positions = np.flatnonzero(ids == cache_id)
        if positions.size < tokens_per_frame or positions.size % tokens_per_frame:
            continue
        flags = predicted[positions]
        if not np.all(flags == flags[0]):
            raise ValueError(
                "one attention cache group mixes predicted and actual keys"
            )
        for frame_in_group, start in enumerate(
            range(0, positions.size, tokens_per_frame)
        ):
            frames.append(
                VisualTokenFrame(
                    positions=positions[start : start + tokens_per_frame].astype(
                        np.int64,
                        copy=False,
                    ),
                    cache_id=cache_id,
                    frame_in_group=frame_in_group,
                    is_predicted=bool(flags[0]),
                )
            )
    return tuple(frames)


def select_current_query_token_positions(
    cache_ids: np.ndarray,
    *,
    query_tokens: int,
) -> tuple[np.ndarray, int]:
    """Locate the temporary current-query keys added for this forward call."""

    ids = np.asarray(cache_ids)
    if ids.ndim != 1 or query_tokens <= 0 or ids.size < query_tokens:
        raise ValueError("attention cache ids are invalid")
    current_id = int(ids.max(initial=-1))
    positions = np.flatnonzero(ids == current_id)
    if positions.size != query_tokens:
        raise RuntimeError(
            "latest attention cache group is not the current query: "
            f"expected {query_tokens}, got {positions.size}"
        )
    return positions.astype(np.int64, copy=False), current_id


def cache_aware_attention_rollout_matrix(
    query_transitions: list[np.ndarray],
    direct_target_attention: list[np.ndarray],
    *,
    selected_query_positions: np.ndarray,
) -> np.ndarray:
    """Roll selected final-query relevance backward to cached target keys."""

    if not query_transitions or len(query_transitions) != len(direct_target_attention):
        raise ValueError("attention rollout layers are incomplete")
    query_tokens = query_transitions[0].shape[0]
    target_tokens = direct_target_attention[0].shape[1]
    selected = np.asarray(selected_query_positions)
    if (
        selected.ndim != 1
        or selected.size == 0
        or not np.issubdtype(selected.dtype, np.integer)
        or np.any(selected < 0)
        or np.any(selected >= query_tokens)
        or np.unique(selected).size != selected.size
    ):
        raise ValueError("selected query positions are invalid")
    for transition, direct in zip(
        query_transitions, direct_target_attention, strict=True
    ):
        if transition.shape != (query_tokens, query_tokens) or direct.shape != (
            query_tokens,
            target_tokens,
        ):
            raise ValueError("attention rollout layer shapes are inconsistent")
    relevance_to_query = np.eye(query_tokens, dtype=np.float32)[selected]
    return _weighted_attention_rollout(
        query_transitions,
        direct_target_attention,
        initial_query_relevance=relevance_to_query,
    )


def _weighted_attention_rollout(
    query_transitions: list[np.ndarray],
    direct_target_attention: list[np.ndarray],
    *,
    initial_query_relevance: np.ndarray,
) -> np.ndarray:
    if not query_transitions or len(query_transitions) != len(direct_target_attention):
        raise ValueError("attention rollout layers are incomplete")
    query_tokens = query_transitions[0].shape[0]
    target_tokens = direct_target_attention[0].shape[1]
    for transition, direct in zip(
        query_transitions, direct_target_attention, strict=True
    ):
        if transition.shape != (query_tokens, query_tokens) or direct.shape != (
            query_tokens,
            target_tokens,
        ):
            raise ValueError("attention rollout layer shapes are inconsistent")
    relevance_to_query = np.asarray(initial_query_relevance, dtype=np.float32)
    if (
        relevance_to_query.ndim != 2
        or relevance_to_query.shape[1] != query_tokens
        or not np.isfinite(relevance_to_query).all()
    ):
        raise ValueError("initial query relevance is invalid")
    relevance_to_target = np.zeros(
        (relevance_to_query.shape[0], target_tokens),
        dtype=np.float32,
    )
    for transition, direct in reversed(
        list(zip(query_transitions, direct_target_attention, strict=True))
    ):
        relevance_to_target += relevance_to_query @ direct
        relevance_to_query = relevance_to_query @ transition
    return relevance_to_target


def cache_aware_attention_rollout(
    query_transitions: list[np.ndarray],
    direct_target_attention: list[np.ndarray],
    *,
    selected_query_positions: np.ndarray,
) -> np.ndarray:
    """Return the mean selected-query rollout to cached target keys."""

    return cache_aware_attention_rollout_matrix(
        query_transitions,
        direct_target_attention,
        selected_query_positions=selected_query_positions,
    ).mean(axis=0)


class LingBotAttentionCapture:
    """Capture video and action self-attention only on an explicit audit."""

    def __init__(
        self,
        *,
        layers: tuple[int, ...],
        frame_chunk_size: int,
        action_query_tokens: int,
        selected_action_query_tokens: int,
        grid_height: int,
        grid_width_per_camera: int,
        video_forward_calls: int,
        action_forward_calls: int,
        cache_name: str = "pos",
    ) -> None:
        if (
            not layers
            or frame_chunk_size <= 0
            or action_query_tokens <= 0
            or not 0 < selected_action_query_tokens <= action_query_tokens
            or grid_height <= 0
            or grid_width_per_camera <= 0
            or video_forward_calls <= 0
            or action_forward_calls <= 0
        ):
            raise ValueError("LingBot attention capture dimensions must be positive")
        self.layers = layers
        self.frame_chunk_size = frame_chunk_size
        self.action_query_tokens = action_query_tokens
        self.selected_action_query_tokens = selected_action_query_tokens
        self.grid_height = grid_height
        self.grid_width_per_camera = grid_width_per_camera
        self.video_forward_calls = video_forward_calls
        self.action_forward_calls = action_forward_calls
        self.cache_name = cache_name
        self._active = False
        self._calls: dict[str, dict[int, int]] = {}
        self._captures: dict[str, dict[int, dict[str, Any]]] = {}

    @property
    def tokens_per_camera(self) -> int:
        return self.grid_height * self.grid_width_per_camera

    @property
    def tokens_per_frame(self) -> int:
        return 2 * self.tokens_per_camera

    @property
    def video_query_tokens(self) -> int:
        return self.frame_chunk_size * self.tokens_per_frame

    def install(self, transformer: Any) -> None:
        blocks = transformer.blocks
        expected_layers = tuple(range(len(blocks)))
        if self.layers != expected_layers:
            raise ValueError(
                "attention.capture_layers must contain every transformer layer "
                f"for rollout: configured={self.layers}, expected={expected_layers}"
            )
        for layer in self.layers:
            attention = blocks[layer].attn1
            original = attention.attn_op

            def captured_attention(
                query,
                key,
                value,
                *,
                _layer=layer,
                _attention=attention,
                _original=original,
            ):
                self._observe(_layer, _attention, query, key)
                return _original(query, key, value)

            attention.attn_op = captured_attention
        transformer._galaxea_attention_capture = self

    def begin(self) -> None:
        if self._active:
            raise RuntimeError("LingBot attention capture is already active")
        self._active = True
        self._calls = {
            stage: {layer: 0 for layer in self.layers} for stage in ("video", "action")
        }
        self._captures = {"video": {}, "action": {}}

    def finish(self) -> dict[str, Any]:
        self._active = False
        for stage in ("video", "action"):
            missing = [
                layer for layer in self.layers if layer not in self._captures[stage]
            ]
            if missing:
                raise RuntimeError(
                    f"LingBot {stage} attention capture missed layers: {missing}"
                )
        video = [self._captures["video"][layer] for layer in self.layers]
        action = [self._captures["action"][layer] for layer in self.layers]
        actual_signature = self._consistent_signature(
            video,
            name="video actual-history",
        )
        action_signature = self._consistent_signature(
            action,
            name="action visual",
        )
        action_actual_signature = tuple(
            item for item in action_signature if not item[2]
        )
        predicted_signature = tuple(item for item in action_signature if item[2])
        if actual_signature != action_actual_signature:
            raise RuntimeError(
                "video and action stages selected different actual-history frames"
            )
        if len(predicted_signature) != self.frame_chunk_size:
            raise RuntimeError(
                "action attention did not expose one predicted visual cache frame "
                f"per chunk frame: expected {self.frame_chunk_size}, "
                f"got {len(predicted_signature)}"
            )

        actual_count = len(actual_signature)
        if actual_count == 0:
            raise RuntimeError("attention capture found no actual visual history")
        actual_tokens = actual_count * self.tokens_per_frame
        predicted_tokens = self.frame_chunk_size * self.tokens_per_frame
        action_direct = [capture["direct_visual"] for capture in action]
        action_transitions = [capture["query_transition"] for capture in action]
        action_visual = cache_aware_attention_rollout(
            action_transitions,
            action_direct,
            selected_query_positions=np.arange(
                self.selected_action_query_tokens,
                dtype=np.int64,
            ),
        )
        if action_visual.size != actual_tokens + predicted_tokens:
            raise RuntimeError("action visual rollout has an invalid token count")
        action_actual = action_visual[:actual_tokens]
        action_predicted = action_visual[actual_tokens:]

        video_initial_relevance = np.zeros(
            (self.frame_chunk_size + 1, predicted_tokens),
            dtype=np.float32,
        )
        for future_index in range(self.frame_chunk_size):
            start = future_index * self.tokens_per_frame
            stop = start + self.tokens_per_frame
            video_initial_relevance[future_index, start:stop] = (
                1.0 / self.tokens_per_frame
            )
        video_initial_relevance[-1] = action_predicted
        video_to_actual = _weighted_attention_rollout(
            [capture["query_transition"] for capture in video],
            [capture["direct_visual"] for capture in video],
            initial_query_relevance=video_initial_relevance,
        )
        if video_to_actual.shape != (
            self.frame_chunk_size + 1,
            actual_tokens,
        ):
            raise RuntimeError("video-to-history rollout has an invalid shape")
        action_via_prediction = video_to_actual[-1]

        action_actual_frames = self._mapped_frames(
            action_actual,
            actual_signature,
            index_name="history_index",
            include_relative_history=True,
        )
        action_predicted_frames = self._mapped_frames(
            action_predicted,
            predicted_signature,
            index_name="future_index",
        )
        video_source_frames = []
        for future_index in range(self.frame_chunk_size):
            video_source_frames.append(
                {
                    "future_index": future_index,
                    "history_frames": self._mapped_frames(
                        video_to_actual[future_index],
                        actual_signature,
                        index_name="history_index",
                        include_relative_history=True,
                    ),
                }
            )

        return {
            "schema_version": 3,
            "kind": "wam_multistage_cache_aware_attention_rollout",
            "interpretation": "diagnostic_association_not_causal_attribution",
            "conditional_batch_index": 0,
            "layers": list(self.layers),
            "head_reduction": "mean",
            "grid": {
                "height": self.grid_height,
                "width": self.grid_width_per_camera,
            },
            "token_layout": {
                "camera_packing": "front_then_wrist_along_latent_width",
                "tokens_per_camera_frame": self.tokens_per_camera,
                "tokens_per_paired_frame": self.tokens_per_frame,
                "actual_history_frames": actual_count,
                "predicted_future_frames": self.frame_chunk_size,
            },
            "capture_stages": {
                "predicted_future_to_actual_history": (
                    "final_predicted_visual_cache_commit_consumed_by_action"
                ),
                "action_to_visual": (
                    "final_scheduler_consumed_action_denoise_after_"
                    "predicted_visual_cache_commit"
                ),
            },
            "rollout": {
                "direction": "final_queries_to_cached_visual_keys",
                "layer_order": (
                    f"forward_{self.layers[0]}_to_{self.layers[-1]}"
                    "_then_reverse_relevance"
                ),
                "residual_rule": "(head_mean_attention+identity)/2",
                "action_query_reduction": (
                    f"mean_first_{self.selected_action_query_tokens}_"
                    "executed_action_queries"
                ),
                "video_query_reduction": (
                    "mean_spatial_queries_within_each_predicted_future_frame"
                ),
                "omitted_paths": [
                    "text_cross_attention",
                    "feed_forward_and_mlp_gates",
                    "cached_action_groups",
                    "cross_layer_paths_through_nonselected_cached_groups",
                ],
            },
            "paths": {
                "action_to_predicted_future": {
                    "source": "executed_action_queries",
                    "target": "predicted_future_visual_keys",
                    "frames": action_predicted_frames,
                },
                "predicted_future_to_actual_history": {
                    "source": "predicted_future_visual_queries",
                    "target": "actual_history_visual_keys",
                    "source_frames": video_source_frames,
                },
                "action_to_actual_history": {
                    "source": "executed_action_queries",
                    "target": "actual_history_visual_keys",
                    "frames": action_actual_frames,
                },
                "action_via_predicted_future_to_actual_history": {
                    "source": "executed_action_queries",
                    "mediator": "predicted_future_visual_tokens",
                    "target": "actual_history_visual_keys",
                    "composition": (
                        "action_to_predicted_future_vector @ "
                        "predicted_future_to_actual_history_matrix"
                    ),
                    "frames": self._mapped_frames(
                        action_via_prediction,
                        actual_signature,
                        index_name="history_index",
                        include_relative_history=True,
                    ),
                },
            },
        }

    def cancel(self) -> None:
        self._active = False
        self._calls = {}
        self._captures = {}

    @staticmethod
    def _consistent_signature(
        captures: list[dict[str, Any]],
        *,
        name: str,
    ) -> tuple[tuple[int, int, bool], ...]:
        signatures = {capture["frame_signature"] for capture in captures}
        if len(signatures) != 1:
            raise RuntimeError(
                f"LingBot attention layers selected different {name} frames"
            )
        return next(iter(signatures))

    def _mapped_frames(
        self,
        vector: np.ndarray,
        signature: tuple[tuple[int, int, bool], ...],
        *,
        index_name: str,
        include_relative_history: bool = False,
    ) -> list[dict[str, Any]]:
        values = np.asarray(vector)
        expected = len(signature) * self.tokens_per_frame
        if values.shape != (expected,):
            raise RuntimeError(
                f"attention path expected {expected} tokens, got {values.shape}"
            )
        result = []
        for index, (cache_id, frame_in_group, is_predicted) in enumerate(signature):
            start = index * self.tokens_per_frame
            front, wrist = split_paired_camera_token_map(
                values[start : start + self.tokens_per_frame],
                grid_height=self.grid_height,
                grid_width_per_camera=self.grid_width_per_camera,
            )
            frame: dict[str, Any] = {
                index_name: index,
                "cache_id": cache_id,
                "frame_in_cache_group": frame_in_group,
                "is_predicted": is_predicted,
                "maps": {"front": front, "wrist": wrist},
                "attention_mass": {
                    "front": float(front.sum()),
                    "wrist": float(wrist.sum()),
                },
            }
            if include_relative_history:
                frame["relative_to_latest"] = index - len(signature) + 1
            result.append(frame)
        return result

    def _observe(self, layer: int, attention: Any, query: Any, key: Any) -> None:
        if not self._active:
            return
        query_tokens = query.shape[1]
        if query_tokens == self.video_query_tokens:
            stage = "video"
            target_call = self.video_forward_calls
        elif query_tokens == self.action_query_tokens:
            stage = "action"
            target_call = self.action_forward_calls
        else:
            return
        self._calls[stage][layer] += 1
        if self._calls[stage][layer] != target_call:
            return

        cache = attention.attn_caches[self.cache_name]
        valid = cache["mask"].nonzero(as_tuple=False).squeeze(-1)
        if valid.numel() != key.shape[1]:
            raise RuntimeError("LingBot attention cache/key length mismatch")
        cache_ids = cache["id"][valid].detach().cpu().numpy()
        predicted = cache["is_pred"][valid].detach().cpu().numpy()
        current_positions, current_cache_id = select_current_query_token_positions(
            cache_ids,
            query_tokens=query_tokens,
        )
        frames = visual_token_frames(
            cache_ids,
            predicted,
            tokens_per_frame=self.tokens_per_frame,
            excluded_cache_ids=frozenset({current_cache_id}),
        )
        if stage == "video":
            selected_frames = tuple(frame for frame in frames if not frame.is_predicted)
        else:
            actual = tuple(frame for frame in frames if not frame.is_predicted)
            future = tuple(frame for frame in frames if frame.is_predicted)
            selected_frames = actual + future
        if not selected_frames:
            raise RuntimeError(f"LingBot {stage} attention has no visual target frames")
        selected_positions = np.concatenate(
            [frame.positions for frame in selected_frames]
        )

        import torch

        query_conditional = query[0].float()
        key_conditional = key[0].float()
        logits = torch.einsum(
            "qhd,khd->hqk",
            query_conditional,
            key_conditional,
        ) * (query_conditional.shape[-1] ** -0.5)
        head_mean = logits.softmax(dim=-1).mean(dim=0)
        target_positions = torch.as_tensor(
            selected_positions,
            device=head_mean.device,
            dtype=torch.long,
        )
        query_key_positions = torch.as_tensor(
            current_positions,
            device=head_mean.device,
            dtype=torch.long,
        )
        direct_visual = (
            head_mean.index_select(-1, target_positions)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        ) / 2.0
        query_attention = head_mean.index_select(-1, query_key_positions)
        query_transition = (
            query_attention
            + torch.eye(
                query_tokens,
                device=head_mean.device,
                dtype=query_attention.dtype,
            )
        ) / 2.0
        self._captures[stage][layer] = {
            "direct_visual": direct_visual,
            "query_transition": (
                query_transition.detach().cpu().numpy().astype(np.float32, copy=False)
            ),
            "frame_signature": tuple(
                (
                    frame.cache_id,
                    frame.frame_in_group,
                    frame.is_predicted,
                )
                for frame in selected_frames
            ),
        }


def write_attention_audit(
    output_root: Path,
    *,
    attention: dict[str, Any],
    actual_observations: list[dict[str, np.ndarray]],
    predicted_observations: list[dict[str, np.ndarray]],
    front_key: str,
    wrist_key: str,
    prompt: str,
    now: datetime | None = None,
) -> Path:
    """Atomically write WAM maps, exact source images, and all overlays."""

    created = now or datetime.now().astimezone()
    run_id = created.strftime("%Y%m%d_%H%M%S_%f")
    root = output_root.expanduser().resolve()
    final_dir = root / run_id
    staging_dir = root / f".{run_id}.staging"
    if final_dir.exists() or staging_dir.exists():
        raise FileExistsError(f"attention audit output already exists: {run_id}")
    root.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir()
    portable_attention = _json_portable(attention)
    portable_attention["prompt"] = prompt
    portable_attention["created_at"] = created.isoformat()
    (staging_dir / "attention.json").write_text(
        json.dumps(portable_attention, indent=2, sort_keys=True) + "\n"
    )

    cameras = (("front", front_key), ("wrist", wrist_key))
    for index, observation in enumerate(actual_observations):
        for camera, key in cameras:
            image = _validated_rgb(observation[key])
            Image.fromarray(image, mode="RGB").save(
                staging_dir / f"actual_history_{index:02d}_{camera}_input.png"
            )
    for index, observation in enumerate(predicted_observations):
        for camera, key in cameras:
            image = _validated_rgb(observation[key])
            Image.fromarray(image, mode="RGB").save(
                staging_dir / f"predicted_future_{index:02d}_{camera}_input.png"
            )

    paths = attention["paths"]
    for path_name in (
        "action_to_actual_history",
        "action_via_predicted_future_to_actual_history",
    ):
        for frame in paths[path_name]["frames"]:
            index = frame["history_index"]
            _write_frame_overlays(
                staging_dir,
                prefix=f"{path_name}_{index:02d}",
                frame=frame,
                observation=actual_observations[index],
                cameras=cameras,
            )
    for frame in paths["action_to_predicted_future"]["frames"]:
        index = frame["future_index"]
        _write_frame_overlays(
            staging_dir,
            prefix=f"action_to_predicted_future_{index:02d}",
            frame=frame,
            observation=predicted_observations[index],
            cameras=cameras,
        )
    for source_frame in paths["predicted_future_to_actual_history"]["source_frames"]:
        future_index = source_frame["future_index"]
        for frame in source_frame["history_frames"]:
            history_index = frame["history_index"]
            _write_frame_overlays(
                staging_dir,
                prefix=(
                    "predicted_future_to_actual_history_"
                    f"future_{future_index:02d}_history_{history_index:02d}"
                ),
                frame=frame,
                observation=actual_observations[history_index],
                cameras=cameras,
            )
    os.replace(staging_dir, final_dir)
    return final_dir


def _write_frame_overlays(
    output: Path,
    *,
    prefix: str,
    frame: dict[str, Any],
    observation: dict[str, np.ndarray],
    cameras: tuple[tuple[str, str], ...],
) -> None:
    for camera, key in cameras:
        overlay = attention_overlay(
            _validated_rgb(observation[key]),
            np.asarray(frame["maps"][camera]),
        )
        Image.fromarray(overlay, mode="RGB").save(
            output / f"{prefix}_{camera}_overlay.png"
        )


def _json_portable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_portable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_portable(item) for item in value]
    return value


def _validated_rgb(value: np.ndarray) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"attention source image is invalid: {image.shape}")
    return image


def attention_overlay(
    rgb: np.ndarray,
    attention_map: np.ndarray,
    *,
    alpha: float = 0.65,
) -> np.ndarray:
    image = np.asarray(rgb)
    heat = np.asarray(attention_map, dtype=np.float32)
    if (
        image.ndim != 3
        or image.shape[2] != 3
        or image.dtype != np.uint8
        or heat.ndim != 2
        or not np.isfinite(heat).all()
        or np.any(heat < 0)
        or not 0 <= alpha <= 1
    ):
        raise ValueError("attention overlay inputs are invalid")
    baseline = float(np.median(heat))
    peak = float(heat.max(initial=0.0))
    normalized = (
        np.clip((heat - baseline) / (peak - baseline), 0.0, 1.0)
        if peak > baseline
        else np.zeros_like(heat)
    )
    resized = np.asarray(
        Image.fromarray(normalized, mode="F").resize(
            (image.shape[1], image.shape[0]),
            resample=Image.Resampling.BILINEAR,
        ),
        dtype=np.float32,
    )
    color = np.empty_like(image, dtype=np.float32)
    color[..., 0] = 255.0
    color[..., 1] = 64.0 + 160.0 * (1.0 - resized)
    color[..., 2] = 0.0
    blend = (alpha * np.square(resized))[..., None]
    return np.clip(image * (1.0 - blend) + color * blend, 0, 255).astype(np.uint8)
