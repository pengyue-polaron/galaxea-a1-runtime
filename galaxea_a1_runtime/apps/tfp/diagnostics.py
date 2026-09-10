"""Non-invasive rollout diagnostics for TFP-Ultra's recurrent belief."""

from __future__ import annotations

import json
import math
from datetime import datetime
from statistics import fmean
from typing import Any

from galaxea_a1_runtime.apps.tfp.config_schema import TFPConfig
from galaxea_a1_runtime.console import info


def _values(tensor: Any) -> list[float]:
    return [
        float(value) for value in tensor.detach().float().reshape(-1).cpu().tolist()
    ]


def _quantile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot summarize an empty diagnostic tensor")
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _distribution(tensor: Any) -> dict[str, float]:
    values = sorted(_values(tensor))
    return {
        "min": values[0],
        "mean": fmean(values),
        "p50": _quantile(values, 0.50),
        "p95": _quantile(values, 0.95),
        "max": values[-1],
    }


def _l2(tensor: Any) -> float:
    values = _values(tensor)
    return math.sqrt(sum(value * value for value in values))


def _cosine(left: Any, right: Any) -> float | None:
    left_values = _values(left)
    right_values = _values(right)
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return sum(a * b for a, b in zip(left_values, right_values, strict=True)) / (
        left_norm * right_norm
    )


class TFPDiagnostics:
    """Capture existing LTC computations without issuing extra model queries."""

    def __init__(self, config: TFPConfig, runner: Any) -> None:
        self._config = config
        self._runner = runner
        self._model = runner.policy.diffusion_ltc
        self._query_index = 0
        self._pending: dict[str, Any] | None = None
        self._layer_ratios: dict[str, Any] = {}
        self._closed = False
        self._original_condition = self._model.condition_and_update_memory
        created = datetime.now().astimezone()
        run_id = created.strftime("%Y%m%d_%H%M%S_%f")
        self.run_dir = config.diagnostics.output_dir / run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.path = self.run_dir / "belief.jsonl"
        self._stream = self.path.open("x", encoding="utf-8")
        self._write(
            {
                "schema_version": 1,
                "kind": "run",
                "run_id": run_id,
                "created_at": created.isoformat(),
                "deployment_id": config.deployment_id,
                "task_id": config.task_id,
                "checkpoint_step": config.model.checkpoint_step,
                "actions_per_query": config.model.actions_per_query,
                "belief_semantics": (
                    "continuous LTC hidden state; not an explicit event probability"
                ),
            }
        )
        self._install_hooks()
        info(f"TFP belief diagnostics: {self.path}")

    def _install_hooks(self) -> None:
        self._hook_handles = []
        for name, module in self._model.named_modules():
            if not hasattr(module, "components") or not hasattr(
                module, "memory_projection"
            ):
                continue

            def capture(
                _module: Any, inputs: tuple[Any, ...], _output: Any, *, layer=name
            ) -> None:
                if self._pending is None or layer in self._layer_ratios:
                    return
                base, memory = _module.components(inputs[0])
                denominator = base.detach().float().norm(dim=-1).clamp_min(1e-12)
                ratio = memory.detach().float().norm(dim=-1) / denominator
                self._layer_ratios[layer] = ratio.mean().detach()

            self._hook_handles.append(module.register_forward_hook(capture))
        self._model.condition_and_update_memory = self._capture_condition

    def _capture_condition(
        self,
        batch: dict[str, Any],
        state: Any,
        elapsed_seconds: Any,
        *,
        reset_mask: Any | None = None,
    ) -> tuple[Any, Any, Any, Any]:
        condition, next_state, ltc, residual_ratio = self._original_condition(
            batch, state, elapsed_seconds, reset_mask=reset_mask
        )
        self._query_index += 1
        self._layer_ratios = {}
        self._pending = {
            "query_index": self._query_index,
            "elapsed_seconds": elapsed_seconds.detach().clone(),
            "state_before": state.detach().clone(),
            "state_after": next_state.detach().clone(),
            "effective_tau_seconds": ltc.effective_tau_seconds.detach().clone(),
            "write_gain": ltc.approximate_write_gain.detach().clone(),
            "reported_update_norm": ltc.state_update_norm.detach().clone(),
            "memory_residual_ratio": residual_ratio.detach().clone(),
        }
        return condition, next_state, ltc, residual_ratio

    def after_action(self, action_index: int, raw_action: Any) -> None:
        pending = self._pending
        if pending is None:
            return
        self._pending = None
        before = pending["state_before"]
        after = pending["state_after"]
        update = after - before
        layer_ratios = {
            name: float(value.float().cpu().item())
            for name, value in sorted(self._layer_ratios.items())
        }
        layer_values = list(layer_ratios.values())
        row: dict[str, Any] = {
            "schema_version": 1,
            "kind": "query",
            "query_index": pending["query_index"],
            "first_action_index": action_index,
            "elapsed_seconds": _values(pending["elapsed_seconds"]),
            "belief_l2_before": _l2(before),
            "belief_l2_after": _l2(after),
            "belief_update_l2": _l2(update),
            "belief_cosine_before_after": _cosine(before, after),
            "reported_update_norm": _distribution(pending["reported_update_norm"]),
            "effective_tau_seconds": _distribution(pending["effective_tau_seconds"]),
            "write_gain": _distribution(pending["write_gain"]),
            "memory_film_ratio_by_layer": layer_ratios,
            "memory_film_ratio_mean": fmean(layer_values) if layer_values else 0.0,
            "memory_film_ratio_max": max(layer_values) if layer_values else 0.0,
            "visual_residual_ratio": _distribution(pending["memory_residual_ratio"]),
            "first_action": [float(value) for value in raw_action],
        }
        if self._config.diagnostics.include_belief_vectors:
            row["belief_before"] = _values(before)
            row["belief_after"] = _values(after)
        self._write(row)
        if pending["query_index"] % self._config.diagnostics.print_every_queries == 0:
            cosine = row["belief_cosine_before_after"]
            cosine_text = "n/a" if cosine is None else f"{cosine:.4f}"
            tau = row["effective_tau_seconds"]
            gain = row["write_gain"]
            info(
                "TFP DIAG "
                f"q={pending['query_index']} action={action_index} "
                f"dt={row['elapsed_seconds'][0]:.3f}s "
                f"belief={row['belief_l2_after']:.3f} "
                f"delta={row['belief_update_l2']:.3f} cos={cosine_text} "
                f"tau50/95={tau['p50']:.3f}/{tau['p95']:.3f}s "
                f"gain50/95={gain['p50']:.3f}/{gain['p95']:.3f} "
                f"film_mean/max={row['memory_film_ratio_mean']:.4f}/"
                f"{row['memory_film_ratio_max']:.4f}"
            )

    def _write(self, payload: dict[str, Any]) -> None:
        self._stream.write(
            json.dumps(
                payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            )
            + "\n"
        )
        self._stream.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._model.condition_and_update_memory = self._original_condition
        for handle in self._hook_handles:
            handle.remove()
        self._write(
            {
                "schema_version": 1,
                "kind": "summary",
                "queries": self._query_index,
            }
        )
        self._stream.close()
