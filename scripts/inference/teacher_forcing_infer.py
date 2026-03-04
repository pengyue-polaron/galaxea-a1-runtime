#!/usr/bin/env python3
"""Teacher-forcing inference: offline trajectory prediction + 3D HTML visualization.

Key design decisions (validated against training data pipeline):
  1. LTC history: model was trained with bptt_len=8 consecutive state frames. We now
     maintain a proper history buffer and pass (bptt_len, 8) proprio_seq each step.
  2. Data source: use processed_data (not raw_data). The LeRobot dataset was built from
     processed_data (~437 entries/demo, ~29 fps video). Videos are 1:1 with states.
  3. Action target offset: training action[t] = commanded_states[t+10] (10-step lookahead
     hardcoded in data_converter.py). Compare pred[t] vs cmd[t+10].
  4. State format: 8D = [j1..j6, gripper_angle_rad, gripper_mm] matches checkpoint
     norm_stats. A1Inputs + LocalDataA1LTCDataConfig repack passes proprio_seq through.

Outputs:
  - <output_dir>/trajectory.json
  - <output_dir>/trajectory.html  (interactive Plotly: 3D EEF, EEF over time, gripper, delta)

Usage:
    just teacher-forcing demo_0_20260227_225247
    just teacher-forcing demo_0_20260227_225247 -- --processed-root /home/jolia/DataCoach/data/processed_data/swap
"""
from __future__ import annotations

import argparse
from collections import deque
import importlib.util
import json
import logging
from pathlib import Path
import pickle
import sys

import cv2
import numpy as np
import pinocchio as pin

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
THIRD_PARTY_OPENPI = ROOT_DIR / "third_party" / "openpi" / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if THIRD_PARTY_OPENPI.exists() and str(THIRD_PARTY_OPENPI) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_OPENPI))

from datacoach.training import config as train_config_lib
from openpi.policies import policy_config

# ---------------------------------------------------------------------------
# Forward kinematics (Pinocchio)
# ---------------------------------------------------------------------------

_A1_URDF = ROOT_DIR / "third_party/A1_SDK/install/share/mobiman/urdf/A1/urdf/A1_URDF_0607_0028.urdf"
_fk_model: pin.Model | None = None
_fk_data: pin.Data | None = None
_fk_ee_frame_id: int | None = None


def _get_fk_model():
    global _fk_model, _fk_data, _fk_ee_frame_id
    if _fk_model is not None:
        return _fk_model, _fk_data, _fk_ee_frame_id
    model = pin.buildModelFromUrdf(str(_A1_URDF))
    data = model.createData()
    ee_name = "arm_joint6"
    frame_id = next((i for i, f in enumerate(model.frames) if f.name == ee_name), model.nframes - 1)
    _fk_model, _fk_data, _fk_ee_frame_id = model, data, frame_id
    return model, data, frame_id


def joints_to_eef(joints6: np.ndarray) -> np.ndarray:
    """Return EEF xyz (m) from 6 arm joint angles."""
    model, data, frame_id = _get_fk_model()
    q = pin.neutral(model)
    q[: min(len(joints6), model.nq)] = joints6[: model.nq]
    pin.framesForwardKinematics(model, data, q)
    return np.array(data.oMf[frame_id].translation)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_external_training_config():
    config_file = Path("/home/jolia/DataCoach/datacoach/training/config.py")
    if not config_file.exists():
        return None
    spec = importlib.util.spec_from_file_location("jolia_training_config", str(config_file))
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_train_config(config_name: str):
    try:
        return train_config_lib.get_config(config_name)
    except Exception as local_exc:
        fallback = _load_external_training_config()
        if fallback is not None:
            try:
                logging.warning("Using fallback config from /home/jolia/DataCoach")
                return fallback.get_config(config_name)
            except Exception as fallback_exc:
                raise ValueError(
                    f"Config '{config_name}' unavailable: local={local_exc}; fallback={fallback_exc}"
                ) from fallback_exc
        raise


def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# LTC history buffer  (Fix #1)
# ---------------------------------------------------------------------------

class LTCHistory:
    """Ring buffer providing proper (bptt_len, D) proprio_seq + time_deltas each step.

    This matches the multi-frame training setup: the model saw bptt_len consecutive
    state frames per step. Without history the CfC starts cold every step, producing
    poor predictions even for a well-converged model.
    """

    def __init__(self, bptt_len: int, action_dim: int, dt_s: float, reset_gap_s: float = 1.0):
        self._bptt_len = max(1, bptt_len)
        self._action_dim = action_dim
        self._dt_s = dt_s
        self._reset_gap_s = reset_gap_s
        self._states: deque = deque(maxlen=self._bptt_len)
        self._times: deque = deque(maxlen=self._bptt_len)

    def reset(self):
        self._states.clear()
        self._times.clear()

    def add(self, state8: np.ndarray, timestamp: float) -> None:
        # Reset on large time gaps or backward jumps
        if self._times:
            dt = float(timestamp) - float(self._times[-1])
            if dt < 0 or dt > self._reset_gap_s:
                self.reset()
        # Pad state to model action_dim
        padded = np.zeros((self._action_dim,), dtype=np.float32)
        n = min(state8.shape[0], self._action_dim)
        padded[:n] = state8[:n]
        self._states.append(padded.copy())
        self._times.append(float(timestamp))

    def get_proprio_seq(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (proprio_seq, time_deltas, ltc_dt) ready for A1Inputs obs dict."""
        states = list(self._states)
        times = list(self._times)
        if len(states) < self._bptt_len:
            pad = self._bptt_len - len(states)
            states = [states[0]] * pad + states
            times = [times[0]] * pad + times

        proprio_seq = np.stack(states[-self._bptt_len:], axis=0).astype(np.float32)  # (T, D)
        ts = np.array(times[-self._bptt_len:], dtype=np.float32)
        deltas = np.diff(ts, prepend=ts[0]).astype(np.float32)
        deltas = np.maximum(deltas, 0.0)
        if np.all(deltas <= 0.0):
            deltas[:] = np.float32(self._dt_s)
        elif deltas[0] <= 0.0:
            pos = deltas[deltas > 0.0]
            deltas[0] = pos[0] if pos.size > 0 else np.float32(self._dt_s)

        ltc_dt = np.array([self._dt_s], dtype=np.float32)
        return proprio_seq, deltas[:, None], ltc_dt


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

ACTION_OFFSET = 10  # data_converter.py: states = actions[:-10]; actions = actions[10:]


def infer_demo(
    *,
    demo_dir: Path,
    policy,
    ltc_history: LTCHistory,
    prompt: str,
    max_steps: int,
) -> list[dict]:
    """Teacher-forcing on one demo with proper LTC history.

    Uses processed_data (1:1 video frames to states, matching training pipeline):
      - video frame t  ↔  commanded_states[t]  (direct sequential read)
      - training action[t] = commanded_states[t + ACTION_OFFSET] (10-step lookahead)

    State: 8D = [j1..j6, gripper_angle_rad, gripper_mm]  (matches checkpoint norm_stats)
    Comparison: pred[t] vs cmd_states[t + ACTION_OFFSET]  (the actual training target)
    """
    cmd_states = _load_pickle(demo_dir / "commanded_states.pkl")
    # Limit to valid range: need t + ACTION_OFFSET < len(cmd_states)
    max_valid = len(cmd_states) - ACTION_OFFSET
    n_steps = max_valid if max_steps <= 0 else min(max_valid, max_steps)

    cap0 = cv2.VideoCapture(str(demo_dir / "cam_0_rgb_video.mp4"))
    cap1 = cv2.VideoCapture(str(demo_dir / "cam_1_rgb_video.mp4"))
    if not cap0.isOpened() or not cap1.isOpened():
        raise RuntimeError(f"Cannot open videos in {demo_dir}")

    n_frames0 = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    ltc_history.reset()
    records = []

    print(
        f"[TF] {demo_dir.name}  steps={n_steps}  "
        f"cmd_states={len(cmd_states)}  cam_frames={n_frames0},{n_frames1}"
    )

    def parse_state8(entry) -> tuple[np.ndarray, float]:
        """Parse commanded_states entry → 8D array + timestamp."""
        data = entry.get("data", entry)
        joints7 = data.get("joint", None)
        if joints7 is None:
            return None, 0.0
        state8 = np.concatenate([
            np.asarray(joints7, dtype=np.float32)[:7],   # [j1..j6, gripper_rad]
            [float(data.get("gripper", 0.0))],            # gripper_mm
        ]).astype(np.float32)
        ts = float(entry.get("timestamp", 0.0))
        return state8, ts

    for t in range(n_steps):
        # 1:1 sequential frame read — processed_data has exactly 1 frame per state
        ok0, frame0 = cap0.read()
        ok1, frame1 = cap1.read()
        if not ok0 or not ok1:
            print(f"[TF] Video ended early at t={t}")
            break

        gt_state8, timestamp = parse_state8(cmd_states[t])
        if gt_state8 is None:
            continue

        # Training action target = cmd_states[t + ACTION_OFFSET]
        tgt_state8, _ = parse_state8(cmd_states[t + ACTION_OFFSET])

        # Teacher-forcing: always feed GT state (not rollout)
        state8 = gt_state8.copy()

        # Add current state to LTC history buffer
        ltc_history.add(state8, timestamp)
        proprio_seq, time_deltas, ltc_dt = ltc_history.get_proprio_seq()

        # obs dict for LocalDataA1LTCDataConfig + A1Inputs repack
        obs = {
            "cam_0": cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB),
            "cam_1": cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB),
            "state": state8,
            "proprio_seq": proprio_seq,    # (bptt_len, action_dim) — A1Inputs passes through
            "time_deltas": time_deltas,    # (bptt_len, 1)
            "ltc_dt": ltc_dt,              # (1,)
            "action": np.zeros(8, dtype=np.float32),
            "prompt": prompt,
            "observation.timestamp": np.float32(timestamp),
            "observation.timestamp_is_pad": np.bool_(False),
            "state_is_pad": np.bool_(False),
        }

        action_dict = policy.infer(obs)
        actions = np.asarray(action_dict["actions"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[np.newaxis, :]

        # A1Outputs returns 8D; take first horizon step
        pred8 = actions[0, :8].astype(np.float32)

        # Compare pred[t] vs cmd[t+ACTION_OFFSET] = the training action target
        delta_arm   = float(np.linalg.norm(pred8[:6] - tgt_state8[:6]))
        delta_full7 = float(np.linalg.norm(pred8[:7] - tgt_state8[:7]))

        records.append({
            "step": t,
            "timestamp": float(timestamp),
            "gt_state":  gt_state8.tolist(),     # 8D current state (model input)
            "gt_target": tgt_state8.tolist(),     # 8D training target (cmd[t+10])
            "pred_action": pred8.tolist(),        # 8D predicted
            "delta_arm":   delta_arm,             # |pred[:6] - target[:6]|
            "delta_full7": delta_full7,           # |pred[:7] - target[:7]|
        })

        if t % 20 == 0:
            print(
                f"  step {t:04d}  |Δarm|={delta_arm:.4f}  |Δ7D|={delta_full7:.4f}"
                f"  gt_j2={gt_state8[1]:.3f}→{tgt_state8[1]:.3f}  pred_j2={pred8[1]:.3f}"
                f"  gt_g_rad={gt_state8[6]:.3f}  pred_g_rad={pred8[6]:.3f}"
            )

    cap0.release()
    cap1.release()
    print(f"[TF] Done. {len(records)} steps collected.")
    return records


# ---------------------------------------------------------------------------
# Save JSON
# ---------------------------------------------------------------------------

def save_json(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"[TF] Saved: {path}")


# ---------------------------------------------------------------------------
# HTML visualization
# ---------------------------------------------------------------------------

def save_html(records: list[dict], path: Path, demo_name: str):
    if not records:
        return

    steps      = [r["step"]        for r in records]
    gt         = np.array([r["gt_state"]    for r in records], dtype=np.float64)  # [T,8]
    gt_tgt     = np.array([r["gt_target"]   for r in records], dtype=np.float64)  # [T,8]
    pred       = np.array([r["pred_action"] for r in records], dtype=np.float64)  # [T,8]
    delta_arm  = [r["delta_arm"]   for r in records]
    delta_full = [r["delta_full7"] for r in records]

    print("[TF] Computing FK for EEF trajectories...")
    gt_eef      = np.array([joints_to_eef(gt[t, :6])      for t in range(len(records))])
    gt_tgt_eef  = np.array([joints_to_eef(gt_tgt[t, :6])  for t in range(len(records))])
    pred_eef    = np.array([joints_to_eef(pred[t, :6])    for t in range(len(records))])
    eef_delta   = np.linalg.norm(pred_eef - gt_tgt_eef, axis=1)

    mean_eef = float(np.mean(eef_delta))
    max_eef  = float(np.max(eef_delta))
    mean_arm = float(np.mean(delta_arm))
    max_arm  = float(np.max(delta_arm))

    traces3d = [
        {"type":"scatter3d","mode":"lines+markers","name":"GT EEF (current state)",
         "x":gt_eef[:,0].tolist(),"y":gt_eef[:,1].tolist(),"z":gt_eef[:,2].tolist(),
         "line":{"color":"royalblue","width":3},
         "marker":{"size":3,"color":steps,"colorscale":"Blues","showscale":False}},
        {"type":"scatter3d","mode":"lines+markers","name":f"GT EEF (target = cmd[t+{ACTION_OFFSET}])",
         "x":gt_tgt_eef[:,0].tolist(),"y":gt_tgt_eef[:,1].tolist(),"z":gt_tgt_eef[:,2].tolist(),
         "line":{"color":"steelblue","width":2,"dash":"dot"},
         "marker":{"size":2,"color":steps,"colorscale":"Blues","showscale":False}},
        {"type":"scatter3d","mode":"lines+markers","name":"Predicted EEF (teacher-forcing)",
         "x":pred_eef[:,0].tolist(),"y":pred_eef[:,1].tolist(),"z":pred_eef[:,2].tolist(),
         "line":{"color":"tomato","width":4},
         "marker":{"size":3,"color":steps,"colorscale":"Reds","showscale":False}},
    ]

    eef_time = []
    for i, ax in enumerate(["X","Y","Z"]):
        c = ["royalblue","seagreen","darkorange"][i]
        eef_time += [
            {"x":steps,"y":gt_tgt_eef[:,i].tolist(),"name":f"GT-target EEF {ax}","mode":"lines",
             "line":{"color":c,"width":2}},
            {"x":steps,"y":pred_eef[:,i].tolist(),"name":f"Pred EEF {ax}","mode":"lines",
             "line":{"color":c,"width":2,"dash":"dash"}},
        ]

    gripper_traces = [
        {"x":steps,"y":gt_tgt[:,6].tolist(), "name":f"GT target gripper_rad (cmd[t+{ACTION_OFFSET}])",
         "mode":"lines","line":{"color":"steelblue","width":2}},
        {"x":steps,"y":pred[:,6].tolist(),   "name":"Pred gripper_rad",
         "mode":"lines","line":{"color":"tomato","width":2,"dash":"dash"}},
        {"x":steps,"y":gt[:,6].tolist(),     "name":"GT current gripper_rad",
         "mode":"lines","line":{"color":"royalblue","width":1,"dash":"dot"}},
        {"x":steps,"y":gt[:,7].tolist(),     "name":"GT gripper_mm (current)","mode":"lines",
         "line":{"color":"seagreen","width":1,"dash":"dot"},"yaxis":"y2"},
    ]

    eef_delta_trace = [{"x":steps,"y":eef_delta.tolist(),"type":"scatter","mode":"lines",
        "name":f"‖EEF pred − GT_target‖ (m)","line":{"color":"crimson","width":2},
        "fill":"tozeroy","fillcolor":"rgba(220,20,60,0.1)"}]

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Teacher-Forcing EEF — {demo_name}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body{{font-family:sans-serif;margin:20px;background:#fafafa}}
  h1{{color:#333}}h2{{color:#666;font-size:.95em;font-weight:normal;margin:2px 0}}
  .stat{{display:inline-block;background:white;border-radius:6px;
         box-shadow:0 1px 3px rgba(0,0,0,.15);padding:10px 20px;margin:4px;text-align:center}}
  .stat .val{{font-size:1.6em;font-weight:bold;color:#d62728}}
  .stat .lbl{{font-size:.8em;color:#888}}
  .pc{{background:white;border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,.15);margin-bottom:24px;padding:8px}}
</style></head><body>
<h1>Teacher-Forcing: {demo_name}</h1>
<h2>Source: processed_data (1:1 video-to-state) &nbsp;|&nbsp; Target: cmd[t+{ACTION_OFFSET}] (training action offset)</h2>
<h2>State: 8D [j1..j6, gripper_rad, gripper_mm] &nbsp;|&nbsp; Pred: 8D (A1Outputs)</h2>
<div style="margin-bottom:16px">
  <div class="stat"><div class="val">{mean_eef*100:.1f} cm</div><div class="lbl">mean EEF error vs target</div></div>
  <div class="stat"><div class="val">{max_eef*100:.1f} cm</div><div class="lbl">max EEF error</div></div>
  <div class="stat"><div class="val">{mean_arm:.4f}</div><div class="lbl">mean |Δarm| (rad)</div></div>
  <div class="stat"><div class="val">{max_arm:.4f}</div><div class="lbl">max |Δarm|</div></div>
  <div class="stat"><div class="val">{len(records)}</div><div class="lbl">steps</div></div>
</div>
<div class="pc"><div id="plot3d"></div></div>
<div class="pc"><div id="plot_eef_time"></div></div>
<div class="pc"><div id="plot_gripper"></div></div>
<div class="pc"><div id="plot_eef_delta"></div></div>
<script>
Plotly.newPlot('plot3d',{json.dumps(traces3d)},{{
  title:'EEF 3D — GT current (blue) / GT target cmd[t+{ACTION_OFFSET}] (blue-dot) / Predicted (red)',
  scene:{{xaxis:{{title:'X (m)'}},yaxis:{{title:'Y (m)'}},zaxis:{{title:'Z (m)'}},aspectmode:'data'}},
  height:650}},{{responsive:true}});
Plotly.newPlot('plot_eef_time',{json.dumps(eef_time)},{{
  title:'EEF over time: GT-target (solid) vs Predicted (dashed)',
  xaxis:{{title:'Step'}},yaxis:{{title:'Position (m)'}},height:380}},{{responsive:true}});
Plotly.newPlot('plot_gripper',{json.dumps(gripper_traces)},{{
  title:'Gripper: angle rad (left axis) / mm current GT (right axis)',
  xaxis:{{title:'Step'}},yaxis:{{title:'rad'}},
  yaxis2:{{title:'mm',overlaying:'y',side:'right'}},height:320}},{{responsive:true}});
Plotly.newPlot('plot_eef_delta',{json.dumps(eef_delta_trace)},{{
  title:'EEF error ‖pred − gt_target‖ per step',
  xaxis:{{title:'Step'}},yaxis:{{title:'m'}},height:280}},{{responsive:true}});
</script></body></html>"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    print(f"[TF] Saved HTML: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    # processed_data has 1:1 video-to-state alignment (was used to build the LeRobot dataset)
    parser.add_argument(
        "--processed-root",
        default="/home/jolia/DataCoach/data/processed_data/swap",
        help="Root dir containing processed demo subdirectories (1:1 video-to-state)",
    )
    # Kept for backwards compat; if set and --processed-root is not, use raw_root
    parser.add_argument("--raw-root", default=None, help="(legacy) raw_data root; prefer --processed-root")
    parser.add_argument("--demo", default=None)
    parser.add_argument("--policy-dir", default="/home/pengyue/6000")
    parser.add_argument("--policy-config", default="pi05_ltc_pick_twice")
    parser.add_argument("--prompt", default="swap the position of the marker and the yellow block through the white plate")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--output-dir", default="/home/pengyue/Codespace/DataCoach/data/teacher_forcing")
    args = parser.parse_args()

    data_root = Path(args.processed_root if args.raw_root is None else args.raw_root).expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"data root not found: {data_root}")

    print(f"[TF] Loading {args.policy_config} from {args.policy_dir} ...")
    config = _resolve_train_config(args.policy_config)
    data_config = config.data.create(config.assets_dirs, config.model)
    policy = policy_config.create_trained_policy(
        train_config=config,
        repack_transforms=data_config.repack_transforms,
        checkpoint_dir=args.policy_dir,
        default_prompt=args.prompt,
    )
    print("[TF] Policy loaded.")

    # LTC history parameters from model
    model_obj = getattr(policy, "_model", None)
    action_dim = int(getattr(model_obj, "action_dim", 32))
    bptt_len   = int(getattr(model_obj, "bptt_len", 8))
    metadata   = policy.metadata or {}
    dt_s       = float(metadata.get("ltc_dt_s", 1.0 / 50.0))
    reset_gap  = float(metadata.get("ltc_history_reset_gap_s", 1.0))
    print(f"[TF] LTC params: bptt_len={bptt_len}, action_dim={action_dim}, dt_s={dt_s:.4f}")

    ltc_history = LTCHistory(bptt_len=bptt_len, action_dim=action_dim, dt_s=dt_s, reset_gap_s=reset_gap)

    demo_dirs = (
        [data_root / args.demo] if args.demo
        else sorted(d for d in data_root.iterdir() if d.is_dir() and d.name.startswith("demo_"))
    )
    for d in demo_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Demo not found: {d}")

    out_root = Path(args.output_dir).expanduser().resolve()

    for demo_dir in demo_dirs:
        records = infer_demo(
            demo_dir=demo_dir,
            policy=policy,
            ltc_history=ltc_history,
            prompt=args.prompt,
            max_steps=args.max_steps,
        )
        for r in records:
            r["prompt"] = args.prompt

        out_dir = out_root / demo_dir.name
        save_json(records, out_dir / "trajectory.json")
        save_html(records, out_dir / "trajectory.html", demo_dir.name)

    print(f"\n[TF] All done. Results in: {out_root}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
