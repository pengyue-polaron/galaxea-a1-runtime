#!/usr/bin/env python3
"""Open-loop rollout eval: policy's own predicted action is fed back as the next state.

Unlike teacher-forcing, the rollout state diverges from ground-truth over time,
exposing covariate-shift behaviour.

Step 0: state = GT state[0]      → policy → pred_action_0
Step 1: state = pred_action_0    → policy → pred_action_1
Step 2: state = pred_action_1    → policy → pred_action_2
...

Camera images are still read from the recorded videos (perception cannot be simulated).
GT commanded states are loaded for error comparison only.

Outputs per demo:
  <output_dir>/<demo_name>/trajectory.json   — per-step records
  <output_dir>/<demo_name>/trajectory.html   — interactive Plotly visualization

Usage:
    python scripts/inference/openloop_rollout.py \\
        --policy-dir /home/pengyue/6000 \\
        --max-demos 1 --max-steps-per-demo 100
"""
from __future__ import annotations

import argparse
from collections import deque
import json
import math
from pathlib import Path
import pickle
import sys
import time

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datacoach.training import config as train_config_lib
from openpi.policies import policy_config


# ---------------------------------------------------------------------------
# Helpers (verbatim from eval_policy_on_processed_data.py)
# ---------------------------------------------------------------------------

def _demo_sort_key(path: Path):
    name = path.name
    if name.startswith("demo_"):
        try:
            return int(name.split("_")[-1])
        except Exception:
            return name
    return name


def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _encode_state(entry: dict) -> np.ndarray:
    data = entry.get("data", entry)
    pos = data.get("pos", [0.0, 0.0, 0.0])
    ori = data.get("ori", [0.0, 0.0, 0.0, 1.0])
    gripper = data.get("gripper", 0.0)
    if gripper is None:
        gripper = 0.0
    return np.asarray([*pos, *ori, float(gripper)], dtype=np.float32)


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    n = float(np.linalg.norm(q))
    if n <= 1e-8:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return q / n


def _quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    q1 = _normalize_quat(q1)
    q2 = _normalize_quat(q2)
    dot = float(np.dot(q1, q2))
    dot = max(-1.0, min(1.0, abs(dot)))
    return float(math.degrees(2.0 * math.acos(dot)))


def _to_summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


class LTCHistoryBuilder:
    def __init__(
        self,
        *,
        action_dim: int,
        history_len: int,
        ltc_dt_s: float,
        use_measured_dt: bool,
        dt_min_s: float,
        dt_max_s: float,
        reset_gap_s: float,
    ):
        self._action_dim = int(action_dim)
        self._history_len = max(1, int(history_len))
        self._ltc_dt_s = float(ltc_dt_s)
        self._use_measured_dt = bool(use_measured_dt)
        self._dt_min_s = float(dt_min_s)
        self._dt_max_s = float(dt_max_s)
        self._reset_gap_s = float(reset_gap_s)
        self._state_history: deque[np.ndarray] = deque(maxlen=self._history_len)
        self._time_history: deque[float] = deque(maxlen=self._history_len)

    def reset(self):
        self._state_history.clear()
        self._time_history.clear()

    def build(self, state8: np.ndarray, timestamp: float):
        padded_state = np.zeros((self._action_dim,), dtype=np.float32)
        usable_dim = min(state8.shape[0], self._action_dim)
        padded_state[:usable_dim] = state8[:usable_dim]

        reset_ltc_state = not self._time_history
        ltc_dt_s = np.float32(self._ltc_dt_s)
        if self._time_history:
            last_ts = float(self._time_history[-1])
            dt = float(timestamp - last_ts)
            if timestamp < last_ts or dt > self._reset_gap_s:
                self._state_history.clear()
                self._time_history.clear()
                reset_ltc_state = True
            elif self._use_measured_dt and dt > 0.0:
                ltc_dt_s = np.float32(np.clip(dt, self._dt_min_s, self._dt_max_s))

        self._state_history.append(padded_state.copy())
        self._time_history.append(float(timestamp))

        states = list(self._state_history)
        times = list(self._time_history)
        if len(states) < self._history_len:
            pad_count = self._history_len - len(states)
            states = [states[0]] * pad_count + states
            times = [times[0]] * pad_count + times

        proprio_seq = np.stack(states[-self._history_len:], axis=0).astype(np.float32, copy=False)
        ts = np.asarray(times[-self._history_len:], dtype=np.float32)
        deltas = np.diff(ts, prepend=ts[0]).astype(np.float32)
        deltas = np.maximum(deltas, 0.0)
        if np.all(deltas <= 0.0):
            deltas[:] = np.float32(self._ltc_dt_s)
        elif deltas[0] <= 0.0:
            positive = deltas[deltas > 0.0]
            deltas[0] = positive[0] if positive.size > 0 else np.float32(self._ltc_dt_s)

        return {
            "proprio_seq": proprio_seq,
            "time_deltas": deltas[:, None],
            "ltc_dt": np.asarray([ltc_dt_s], dtype=np.float32),
            "reset": bool(reset_ltc_state),
        }


def _extract_policy_action(action_dict: dict, *, fallback_gripper: float) -> np.ndarray:
    actions = np.asarray(action_dict["actions"], dtype=np.float32)
    if actions.ndim == 2:
        action = actions[0]
    else:
        action = actions

    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] == 7:
        action = np.concatenate([action, np.asarray([fallback_gripper], dtype=np.float32)])
    if action.shape[0] < 8:
        raise ValueError(f"Expected >=7 action dims after policy output transform, got {action.shape[0]}")
    action = action[:8].astype(np.float32, copy=False)
    action[3:7] = _normalize_quat(action[3:7])
    return action


# ---------------------------------------------------------------------------
# Open-loop rollout for one demo
# ---------------------------------------------------------------------------

def run_openloop_demo(
    *,
    demo_dir: Path,
    policy,
    history: LTCHistoryBuilder,
    prompt: str,
    episode_id: str,
    max_steps: int,
    deterministic_noise: np.ndarray | None,
) -> list[dict]:
    """Run open-loop rollout on one demo. Returns per-step records."""
    states = _load_pickle(demo_dir / "states.pkl")
    commanded = _load_pickle(demo_dir / "commanded_states.pkl")

    cap0 = cv2.VideoCapture(str(demo_dir / "cam_0_rgb_video.mp4"))
    cap1 = cv2.VideoCapture(str(demo_dir / "cam_1_rgb_video.mp4"))
    if not cap0.isOpened() or not cap1.isOpened():
        raise RuntimeError(f"Cannot open videos in {demo_dir}")

    n_steps = min(len(states) - 1, len(commanded) - 1)
    if max_steps > 0:
        n_steps = min(n_steps, max_steps)
    if n_steps <= 0:
        raise RuntimeError("Not enough states/commands to evaluate.")

    history.reset()
    if hasattr(policy, "reset"):
        policy.reset()

    rollout_state: np.ndarray | None = None  # open-loop: previous predicted action as next state
    records: list[dict] = []

    print(f"[OL] {demo_dir.name}  steps={n_steps}")

    for t in range(n_steps):
        ok0, frame0 = cap0.read()
        ok1, frame1 = cap1.read()
        if not ok0 or not ok1 or frame0 is None or frame1 is None:
            print(f"[OL] Could not read frames at step {t}, stopping.")
            break

        gt_state = _encode_state(states[t])          # [x,y,z,qx,qy,qz,qw,gripper]
        target_action = _encode_state(commanded[t + 1])
        timestamp = float(states[t].get("timestamp", float(t) / 50.0))

        # Open-loop: use previous predicted action; GT only at step 0
        if rollout_state is None:
            state_in = gt_state.copy()
        else:
            state_in = rollout_state.copy()

        ltc_inputs = history.build(state_in, timestamp)

        obs = {
            "cam_0": cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB),
            "cam_1": cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB),
            "state": state_in,
            "proprio_seq": ltc_inputs["proprio_seq"],
            "time_deltas": ltc_inputs["time_deltas"],
            "ltc_dt": ltc_inputs["ltc_dt"],
            "episode_id": episode_id,
            "reset": np.bool_(ltc_inputs["reset"]),
            "action": np.zeros_like(state_in, dtype=np.float32),
            "prompt": prompt,
            "observation.timestamp": np.float32(timestamp),
            "observation.timestamp_is_pad": np.bool_(False),
            "state_is_pad": np.bool_(False),
        }

        if deterministic_noise is None:
            pred_dict = policy.infer(obs)
        else:
            pred_dict = policy.infer(obs, noise=deterministic_noise)

        pred_action = _extract_policy_action(pred_dict, fallback_gripper=float(state_in[7]))
        rollout_state = pred_action.copy()

        pos_err = float(np.linalg.norm(pred_action[:3] - target_action[:3]))
        ori_err = _quat_angle_deg(pred_action[3:7], target_action[3:7])
        grip_err = float(abs(pred_action[7] - target_action[7]))
        action_l2 = float(np.linalg.norm(pred_action - target_action))

        records.append({
            "step": t,
            "timestamp": float(timestamp),
            "gt_state": gt_state.tolist(),
            "gt_target": target_action.tolist(),
            "rollout_state": state_in.tolist(),
            "pred_action": pred_action.tolist(),
            "pos_error_m": pos_err,
            "ori_error_deg": ori_err,
            "gripper_error": grip_err,
            "action_l2": action_l2,
        })

        if t % 20 == 0:
            print(f"  step {t:04d}  pos_err={pos_err:.4f}m  ori_err={ori_err:.2f}°  "
                  f"grip_err={grip_err:.2f}  l2={action_l2:.4f}")

    cap0.release()
    cap1.release()
    print(f"[OL] Done. {len(records)} steps collected.")
    return records


# ---------------------------------------------------------------------------
# HTML visualization
# ---------------------------------------------------------------------------

def save_html(records: list[dict], path: Path, demo_name: str):
    if not records:
        print("[OL] No records to visualize.")
        return

    steps = [r["step"] for r in records]
    gt_cmd = np.array([r["gt_target"]    for r in records], dtype=np.float64)  # [T, 8]
    pred   = np.array([r["pred_action"]  for r in records], dtype=np.float64)  # [T, 8]
    pos_errors   = [r["pos_error_m"]   for r in records]
    ori_errors   = [r["ori_error_deg"] for r in records]

    mean_pos = float(np.mean(pos_errors))
    max_pos  = float(np.max(pos_errors))
    mean_ori = float(np.mean(ori_errors))
    max_ori  = float(np.max(ori_errors))

    # --- 3D EEF trajectory (pos is already EE-space x,y,z) ---
    traces3d = [
        {
            "type": "scatter3d", "mode": "lines+markers", "name": "GT commanded",
            "x": gt_cmd[:, 0].tolist(), "y": gt_cmd[:, 1].tolist(), "z": gt_cmd[:, 2].tolist(),
            "line": {"color": "royalblue", "width": 4},
            "marker": {"size": 3, "color": steps, "colorscale": "Blues", "showscale": False},
        },
        {
            "type": "scatter3d", "mode": "lines+markers", "name": "Open-loop rollout",
            "x": pred[:, 0].tolist(), "y": pred[:, 1].tolist(), "z": pred[:, 2].tolist(),
            "line": {"color": "tomato", "width": 4},
            "marker": {"size": 3, "color": steps, "colorscale": "Reds", "showscale": False},
        },
    ]

    # --- Per-axis time series: x, y, z, gripper ---
    axis_traces = []
    colors = ["royalblue", "seagreen", "darkorange"]
    for i, axis in enumerate(["X (m)", "Y (m)", "Z (m)"]):
        axis_traces.append({
            "x": steps, "y": gt_cmd[:, i].tolist(),
            "name": f"GT {axis}", "mode": "lines",
            "line": {"color": colors[i], "width": 2},
        })
        axis_traces.append({
            "x": steps, "y": pred[:, i].tolist(),
            "name": f"Rollout {axis}", "mode": "lines",
            "line": {"color": colors[i], "width": 2, "dash": "dash"},
        })
    # gripper
    axis_traces.append({
        "x": steps, "y": gt_cmd[:, 7].tolist(),
        "name": "GT gripper", "mode": "lines",
        "line": {"color": "steelblue", "width": 2},
    })
    axis_traces.append({
        "x": steps, "y": pred[:, 7].tolist(),
        "name": "Rollout gripper", "mode": "lines",
        "line": {"color": "tomato", "width": 2, "dash": "dash"},
    })

    # --- Error over time ---
    error_traces = [
        {
            "x": steps, "y": pos_errors,
            "type": "scatter", "mode": "lines", "name": "pos error (m)",
            "line": {"color": "crimson", "width": 2},
            "fill": "tozeroy", "fillcolor": "rgba(220,20,60,0.1)",
        },
        {
            "x": steps, "y": ori_errors,
            "type": "scatter", "mode": "lines", "name": "ori error (deg)",
            "line": {"color": "darkorchid", "width": 2},
            "yaxis": "y2",
        },
    ]

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Open-Loop Rollout — {demo_name}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body {{ font-family: sans-serif; margin: 20px; background: #fafafa; }}
  h1 {{ color: #333; }}
  h2 {{ color: #555; font-size: 1em; font-weight: normal; margin-top: 0; }}
  .stat {{ display: inline-block; background: white; border-radius: 6px;
           box-shadow: 0 1px 3px rgba(0,0,0,.15); padding: 10px 20px; margin: 4px; text-align: center; }}
  .stat .val {{ font-size: 1.6em; font-weight: bold; color: #d62728; }}
  .stat .lbl {{ font-size: 0.8em; color: #888; }}
  .plot-container {{ background: white; border-radius: 8px;
                     box-shadow: 0 1px 4px rgba(0,0,0,.15); margin-bottom: 24px; padding: 8px; }}
</style>
</head>
<body>
<h1>Open-Loop Rollout: {demo_name}</h1>
<h2>State input: previous predicted action (GT only at step 0) &nbsp;|&nbsp; Camera: recorded frames</h2>

<div style="margin-bottom:16px">
  <div class="stat"><div class="val">{mean_pos*100:.2f} cm</div><div class="lbl">mean pos error</div></div>
  <div class="stat"><div class="val">{max_pos*100:.2f} cm</div><div class="lbl">max pos error</div></div>
  <div class="stat"><div class="val">{mean_ori:.2f}°</div><div class="lbl">mean ori error</div></div>
  <div class="stat"><div class="val">{max_ori:.2f}°</div><div class="lbl">max ori error</div></div>
  <div class="stat"><div class="val">{len(records)}</div><div class="lbl">steps</div></div>
</div>

<div class="plot-container"><div id="plot3d"></div></div>
<div class="plot-container"><div id="plot_axes"></div></div>
<div class="plot-container"><div id="plot_errors"></div></div>

<script>
Plotly.newPlot('plot3d', {json.dumps(traces3d)}, {{
  title: '3D EEF Trajectory — GT commanded (blue) vs Open-Loop Rollout (red)',
  scene: {{
    xaxis: {{title: 'X (m)'}},
    yaxis: {{title: 'Y (m)'}},
    zaxis: {{title: 'Z (m)'}},
    aspectmode: 'data',
  }},
  legend: {{x: 0, y: 1}}, height: 650,
}}, {{responsive: true}});

Plotly.newPlot('plot_axes', {json.dumps(axis_traces)}, {{
  title: 'Per-Axis & Gripper: GT commanded (solid) vs Open-Loop Rollout (dashed)',
  xaxis: {{title: 'Step'}},
  yaxis: {{title: 'Value'}},
  height: 420,
}}, {{responsive: true}});

Plotly.newPlot('plot_errors', {json.dumps(error_traces)}, {{
  title: 'Error over Time: pos (m, left axis) and ori (°, right axis)',
  xaxis: {{title: 'Step'}},
  yaxis: {{title: 'pos error (m)', side: 'left'}},
  yaxis2: {{title: 'ori error (deg)', side: 'right', overlaying: 'y'}},
  legend: {{x: 0, y: 1}},
  height: 320,
}}, {{responsive: true}});
</script>
</body>
</html>"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    print(f"[OL] Saved HTML: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Open-loop rollout eval: policy's predicted action is fed back as next state."
    )
    parser.add_argument(
        "--processed-root",
        default="data/processed_data/pick_twice",
        help="Processed demo root, containing demo_x directories.",
    )
    parser.add_argument("--policy-dir", required=True, help="Checkpoint directory, e.g. /home/pengyue/6000")
    parser.add_argument("--policy-config", default="pi05_ltc_pick_twice", help="Training config name")
    parser.add_argument("--prompt", default="pick twice", help="Prompt fed to policy")
    parser.add_argument("--max-demos", type=int, default=0, help="0 = all demos")
    parser.add_argument("--max-steps-per-demo", type=int, default=0, help="0 = all steps")
    parser.add_argument(
        "--output-dir",
        default="data/openloop_rollout",
        help="Root output directory; one subdirectory per demo.",
    )
    parser.add_argument("--output-json", default=None, help="Optional JSON summary path")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    processed_root = Path(args.processed_root).expanduser().resolve()
    if not processed_root.is_absolute():
        processed_root = (ROOT_DIR / args.processed_root).resolve()
    if not processed_root.exists():
        raise FileNotFoundError(f"processed root not found: {processed_root}")

    out_root = Path(args.output_dir).expanduser()
    if not out_root.is_absolute():
        out_root = ROOT_DIR / args.output_dir
    out_root = out_root.resolve()

    print(f"[OL] Loading {args.policy_config} from {args.policy_dir} ...")
    config = train_config_lib.get_config(args.policy_config)
    data_config = config.data.create(config.assets_dirs, config.model)
    policy = policy_config.create_trained_policy(
        train_config=config,
        repack_transforms=data_config.repack_transforms,
        checkpoint_dir=args.policy_dir,
        default_prompt=args.prompt,
    )
    print("[OL] Policy loaded.")

    metadata = policy.metadata or {}
    model = getattr(policy, "_model", None)
    action_dim = int(getattr(model, "action_dim", 32))
    bptt_len = int(getattr(model, "bptt_len", 8))
    ltc_dt_s = float(metadata.get("ltc_dt_s", 1.0 / 50.0))
    use_measured_dt = bool(metadata.get("ltc_use_measured_dt", False))
    dt_min_s = float(metadata.get("ltc_dt_min_s", ltc_dt_s * 0.5))
    dt_max_s = float(metadata.get("ltc_dt_max_s", ltc_dt_s * 2.0))
    reset_gap_s = float(metadata.get("ltc_history_reset_gap_s", 1.0))
    episode_prefix = str(metadata.get("ltc_episode_id", "openloop_rollout"))

    horizon = int(getattr(model, "action_horizon", 10))
    deterministic_noise = np.zeros((horizon, action_dim), dtype=np.float32)

    print(
        f"[OL] processed_root={processed_root}\n"
        f"[OL] LTC: dt={ltc_dt_s:.4f}s use_measured_dt={use_measured_dt} "
        f"dt_min={dt_min_s:.4f}s dt_max={dt_max_s:.4f}s bptt={bptt_len}"
    )

    demo_dirs = sorted([d for d in processed_root.iterdir() if d.is_dir()], key=_demo_sort_key)
    if args.max_demos > 0:
        demo_dirs = demo_dirs[: args.max_demos]
    if not demo_dirs:
        raise RuntimeError(f"No demo directories found under {processed_root}")

    # Aggregate metrics
    pos_errors_m: list[float] = []
    ori_errors_deg: list[float] = []
    gripper_errors: list[float] = []
    action_l2_errors: list[float] = []
    pred_step_pos_m: list[float] = []
    pred_step_ori_deg: list[float] = []

    evaluated_demos = 0
    skipped_demos: list[dict[str, str]] = []
    total_samples = 0

    for demo_dir in demo_dirs:
        history = LTCHistoryBuilder(
            action_dim=action_dim,
            history_len=bptt_len,
            ltc_dt_s=ltc_dt_s,
            use_measured_dt=use_measured_dt,
            dt_min_s=dt_min_s,
            dt_max_s=dt_max_s,
            reset_gap_s=reset_gap_s,
        )
        try:
            records = run_openloop_demo(
                demo_dir=demo_dir,
                policy=policy,
                history=history,
                prompt=args.prompt,
                episode_id=f"{episode_prefix}_{demo_dir.name}",
                max_steps=args.max_steps_per_demo,
                deterministic_noise=deterministic_noise,
            )
        except Exception as exc:
            skipped_demos.append({"demo": demo_dir.name, "error": str(exc)})
            continue

        # Accumulate metrics and compute smoothness
        prev_pred = None
        demo_pos: list[float] = []
        demo_ori: list[float] = []
        for r in records:
            pos_errors_m.append(r["pos_error_m"])
            ori_errors_deg.append(r["ori_error_deg"])
            gripper_errors.append(r["gripper_error"])
            action_l2_errors.append(r["action_l2"])
            demo_pos.append(r["pos_error_m"])
            demo_ori.append(r["ori_error_deg"])
            pred = np.asarray(r["pred_action"], dtype=np.float64)
            if prev_pred is not None:
                pred_step_pos_m.append(float(np.linalg.norm(pred[:3] - prev_pred[:3])))
                pred_step_ori_deg.append(_quat_angle_deg(pred[3:7], prev_pred[3:7]))
            prev_pred = pred
            total_samples += 1

        evaluated_demos += 1
        if demo_pos:
            print(
                f"[Demo {demo_dir.name}] n={len(demo_pos)} "
                f"pos_mean={np.mean(demo_pos):.4f}m ori_mean={np.mean(demo_ori):.2f}°"
            )

        out_dir = out_root / demo_dir.name
        # Save JSON
        json_path = out_dir / "trajectory.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"[OL] Saved JSON: {json_path}")
        # Save HTML
        save_html(records, out_dir / "trajectory.html", demo_dir.name)

    summary = {
        "processed_root": str(processed_root),
        "policy_config": args.policy_config,
        "policy_dir": str(Path(args.policy_dir).expanduser().resolve()),
        "mode": "open_loop_rollout",
        "demos_total": len(demo_dirs),
        "demos_evaluated": evaluated_demos,
        "demos_skipped": skipped_demos,
        "num_samples": total_samples,
        "metrics": {
            "pos_error_m": _to_summary(pos_errors_m),
            "ori_error_deg": _to_summary(ori_errors_deg),
            "gripper_error": _to_summary(gripper_errors),
            "action_l2": _to_summary(action_l2_errors),
            "pred_step_pos_m": _to_summary(pred_step_pos_m),
            "pred_step_ori_deg": _to_summary(pred_step_ori_deg),
        },
        "hit_rates": {
            "pos<=0.02m": float(np.mean(np.asarray(pos_errors_m) <= 0.02)) if pos_errors_m else None,
            "pos<=0.05m": float(np.mean(np.asarray(pos_errors_m) <= 0.05)) if pos_errors_m else None,
            "ori<=10deg": float(np.mean(np.asarray(ori_errors_deg) <= 10.0)) if ori_errors_deg else None,
            "ori<=20deg": float(np.mean(np.asarray(ori_errors_deg) <= 20.0)) if ori_errors_deg else None,
        },
        "eval_options": {
            "prompt": args.prompt,
            "max_demos": args.max_demos,
            "max_steps_per_demo": args.max_steps_per_demo,
        },
    }

    print("\n[OL Summary]")
    print(json.dumps(summary, indent=2, ensure_ascii=True))

    if args.output_json:
        out_path = Path(args.output_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        print(f"[OL] Wrote summary JSON: {out_path}")

    print(f"\n[OL] All done. Results in: {out_root}")


if __name__ == "__main__":
    main()
