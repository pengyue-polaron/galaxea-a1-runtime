"""Hardware-free orchestration of LingBot chunks and bounded IK replanning."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from galaxea_a1_runtime.console import success, warning
from galaxea_a1_runtime.hardware.eef_ik import A1EefIkTargetRejected


def run_lingbot_rollout(bridge: Any, *, is_shutdown: Callable[[], bool]) -> None:
    """Drive the bridge hooks; only a typed rejected IK target permits replanning."""
    bridge._prepare_execution()
    execution = bridge.execution
    first = True
    call_index = 0
    consecutive_replans = 0
    while not is_shutdown():
        if execution.max_model_calls > 0 and call_index >= execution.max_model_calls:
            bridge.live_status.break_line()
            success(
                "LingBot rollout complete: reached configured "
                f"max_model_calls={execution.max_model_calls}; "
                "the bridge will lock and stop the runtime."
            )
            return
        if not bridge._wait_for_inference_request(call_index):
            return
        chunk = bridge._infer_chunk(call_index, first=first)
        if chunk is None:
            return
        try:
            stop, key_frames, cache_eligible = bridge._execute_chunk(call_index, chunk)
        except A1EefIkTargetRejected as exc:
            if (
                not execution.execute
                or consecutive_replans >= execution.ik_replan_max_attempts
                or (
                    execution.max_model_calls > 0
                    and call_index + 1 >= execution.max_model_calls
                )
            ):
                raise
            consecutive_replans += 1
            bridge.live_status.break_line()
            warning(
                "IK target rejected; discarding the remaining chunk and replanning "
                f"({consecutive_replans}/{execution.ik_replan_max_attempts}): {exc}"
            )
            bridge._recover_ik_rejection()
            first = True
            call_index += 1
            bridge._update_live_status(call_index, phase="REPLAN", force=True)
            continue
        if stop:
            return
        cache_updated = bridge._sync_kv_cache(
            call_index, chunk, key_frames=key_frames, cache_eligible=cache_eligible
        )
        if execution.execute:
            first = not cache_updated
        # Partial/skipped execution does not replenish the retry allowance.
        if cache_updated:
            consecutive_replans = 0
        call_index += 1
        if not execution.max_model_calls or call_index < execution.max_model_calls:
            bridge._update_live_status(call_index, phase="OBSERVE")


def reset_after_ik_rejection(
    *,
    executor: Any,
    state: Any,
    client: Any,
    prompt: str,
    wait_for_feedback: Callable[[], None],
) -> None:
    """Hold, refresh feedback, and start a new model episode before re-inference."""
    executor.hold_for_replan()
    wait_for_feedback()
    # Keep the established origin if any reset operation fails. The caller's
    # normal failure cleanup locks the relay; there is no recovery of faults.
    client.reset(prompt)
    wait_for_feedback()
    state.episode_origin = None
    if state.ensure_episode_origin() is None:
        raise RuntimeError("Cannot establish a fresh episode origin for IK replanning")
