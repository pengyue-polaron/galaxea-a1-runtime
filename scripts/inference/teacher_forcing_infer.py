#!/usr/bin/env python3
"""Teacher-forcing visualization: offline inference via WebSocket policy server.

At each step t, feeds ground-truth joint state + camera frames (teacher forcing) to the
running policy server.  Predicted actions are compared to the ground-truth next state.

Requires: `just policy` (WebSocket server on port 8000) to be running first.

Outputs per demo:
  <output_dir>/<demo_name>/trajectory.json
  <output_dir>/<demo_name>/trajectory.html  (interactive Plotly)

Usage:
    just teacher-forcing                        # all demos in default root
    just teacher-forcing demo_0                 # single demo
    just teacher-forcing -- --port 8001         # custom WebSocket port
    # or directly:
    python scripts/inference/teacher_forcing_infer.py \\
        --processed-root data/processed_data/pick_twice --prompt pick_twice
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import pickle
import sys

import cv2
import numpy as np

# pinocchio for forward kinematics (optional)
try:
    import pinocchio as pin
    _HAS_PIN = True
except ImportError:
    _HAS_PIN = False

# openpi_client WebSocket interface — PYTHONPATH set by `just teacher-forcing`
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

from openpi_client import websocket_client_policy as _ws_policy


# ---------------------------------------------------------------------------
# Forward kinematics (Pinocchio)
# ---------------------------------------------------------------------------

_A1_URDF = ROOT_DIR / "third_party/A1_SDK/install/share/mobiman/urdf/A1/urdf/A1_URDF_0607_0028.urdf"
_fk_model = None
_fk_data = None
_fk_ee_frame_id = None


def _init_fk():
    global _fk_model, _fk_data, _fk_ee_frame_id
    if _fk_model is not None:
        return True
    if not _HAS_PIN or not _A1_URDF.exists():
        return False
    model = pin.buildModelFromUrdf(str(_A1_URDF))
    data = model.createData()
    ee_name = "arm_joint6"
    frame_id = next(
        (i for i, f in enumerate(model.frames) if f.name == ee_name),
        model.nframes - 1,
    )
    _fk_model, _fk_data, _fk_ee_frame_id = model, data, frame_id
    return True


def joints_to_eef(joints6: np.ndarray) -> np.ndarray | None:
    """Return EEF xyz (m) from 6 arm joint angles. None if pinocchio unavailable."""
    if not _init_fk():
        return None
    q = pin.neutral(_fk_model)
    n = min(len(joints6), _fk_model.nq)
    q[:n] = joints6[:n]
    pin.framesForwardKinematics(_fk_model, _fk_data, q)
    return np.array(_fk_data.oMf[_fk_ee_frame_id].translation)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _parse_state7(entry) -> tuple[np.ndarray | None, float]:
    """Parse commanded_states entry → 7D [j1..j6, gripper_rad] + timestamp."""
    data = entry.get("data", entry)
    # Support both 'joint' (old format) and 'joints' (new format)
    joints = data.get("joint", data.get("joints", None))
    if joints is None:
        return None, 0.0
    state7 = np.asarray(joints, dtype=np.float32)[:7]
    ts = float(entry.get("timestamp", 0.0))
    return state7, ts


def _read_frame_rgb(cap: cv2.VideoCapture) -> np.ndarray | None:
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _demo_sort_key(path: Path) -> int | str:
    parts = path.name.split("_")
    for p in parts:
        if p.isdigit():
            return int(p)
    return path.name


# ---------------------------------------------------------------------------
# Teacher-forcing inference on one demo
# ---------------------------------------------------------------------------

def infer_demo(
    *,
    demo_dir: Path,
    policy,
    prompt: str,
    max_steps: int,
) -> list[dict]:
    """Feed GT state + camera frames to WebSocket policy at each step (teacher forcing).

    State format (7D): [j1, j2, j3, j4, j5, j6, gripper_rad]
    GT action target: commanded_states[t+1]  (immediate next joint target)
    """
    cmd_states = _load_pickle(demo_dir / "commanded_states.pkl")
    n_valid = len(cmd_states) - 1  # need t+1 for target
    n_steps = n_valid if max_steps <= 0 else min(n_valid, max_steps)

    cap0 = cv2.VideoCapture(str(demo_dir / "cam_0_rgb_video.mp4"))
    cap1 = cv2.VideoCapture(str(demo_dir / "cam_1_rgb_video.mp4"))
    if not cap0.isOpened() or not cap1.isOpened():
        raise RuntimeError(f"Cannot open videos in {demo_dir}")

    print(
        f"[TF] {demo_dir.name}  steps={n_steps}  "
        f"cmd_states={len(cmd_states)}  "
        f"video_frames={int(cap0.get(cv2.CAP_PROP_FRAME_COUNT))},"
        f"{int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))}"
    )

    records: list[dict] = []
    for t in range(n_steps):
        frame0 = _read_frame_rgb(cap0)
        frame1 = _read_frame_rgb(cap1)
        if frame0 is None or frame1 is None:
            print(f"[TF] Video ended early at t={t}")
            break

        state7, timestamp = _parse_state7(cmd_states[t])
        if state7 is None:
            continue

        target7, _ = _parse_state7(cmd_states[t + 1])
        if target7 is None:
            continue

        obs = {
            "observation/image":       frame0,   # HWC uint8 RGB
            "observation/wrist_image": frame1,   # HWC uint8 RGB
            "observation/state":       state7,   # (7,) float32
            "prompt":                  prompt,
        }

        result = policy.infer(obs)
        actions = np.asarray(result["actions"], dtype=np.float32)  # (horizon, 7)
        if actions.ndim == 1:
            actions = actions[np.newaxis, :]

        pred7 = actions[0, :7]  # immediate predicted action

        delta_arm  = float(np.linalg.norm(pred7[:6] - target7[:6]))
        delta_full = float(np.linalg.norm(pred7 - target7))
        delta_grip = float(abs(float(pred7[6]) - float(target7[6])))

        records.append({
            "step":         t,
            "timestamp":    float(timestamp),
            "gt_state":     state7.tolist(),
            "gt_target":    target7.tolist(),
            "pred_action":  pred7.tolist(),
            "pred_horizon": actions[:, :7].tolist(),
            "delta_arm":    delta_arm,
            "delta_full7":  delta_full,
            "delta_grip":   delta_grip,
        })

        if t % 20 == 0:
            print(
                f"  step {t:04d}  |Δarm|={delta_arm:.4f}  |Δ7D|={delta_full:.4f}"
                f"  j2: {float(state7[1]):.3f}→{float(target7[1]):.3f} pred={float(pred7[1]):.3f}"
                f"  grip: gt={float(target7[6]):.3f} pred={float(pred7[6]):.3f}"
            )

    cap0.release()
    cap1.release()
    print(f"[TF] Done. {len(records)} steps collected.")
    return records


# ---------------------------------------------------------------------------
# HTML visualization
# ---------------------------------------------------------------------------

_JOINT_LABELS = ["j1", "j2", "j3", "j4", "j5", "j6", "gripper_rad"]
_JOINT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]


def save_html(records: list[dict], path: Path, demo_name: str):
    if not records:
        return

    steps     = [r["step"]        for r in records]
    gt_state  = np.array([r["gt_state"]   for r in records], dtype=np.float64)  # [T, 7]
    gt_target = np.array([r["gt_target"]  for r in records], dtype=np.float64)  # [T, 7]
    pred      = np.array([r["pred_action"] for r in records], dtype=np.float64)  # [T, 7]
    delta_arm = [r["delta_arm"]  for r in records]
    delta_grip= [r["delta_grip"] for r in records]

    mean_arm  = float(np.mean(delta_arm))
    max_arm   = float(np.max(delta_arm))
    mean_grip = float(np.mean(delta_grip))

    # --- 3D EEF trajectory via FK ---
    use_3d = _init_fk()
    eef_err = np.zeros(len(records))
    mean_eef = max_eef = 0.0
    traces3d: list[dict] = []
    eef_err_trace: list[dict] = []

    if use_3d:
        print("[TF] Computing FK ...")
        _z = np.zeros(3, dtype=np.float64)
        gt_eef   = np.array([_v if (_v := joints_to_eef(gt_state[i, :6]))  is not None else _z for i in range(len(records))])
        tgt_eef  = np.array([_v if (_v := joints_to_eef(gt_target[i, :6])) is not None else _z for i in range(len(records))])
        pred_eef = np.array([_v if (_v := joints_to_eef(pred[i, :6]))      is not None else _z for i in range(len(records))])
        eef_err  = np.linalg.norm(pred_eef - tgt_eef, axis=1)
        mean_eef = float(np.mean(eef_err))
        max_eef  = float(np.max(eef_err))

        traces3d = [
            {
                "type": "scatter3d", "mode": "lines+markers", "name": "GT current EEF",
                "x": gt_eef[:, 0].tolist(), "y": gt_eef[:, 1].tolist(), "z": gt_eef[:, 2].tolist(),
                "line": {"color": "royalblue", "width": 3},
                "marker": {"size": 2, "color": steps, "colorscale": "Blues", "showscale": False},
            },
            {
                "type": "scatter3d", "mode": "lines+markers", "name": "GT next state (target)",
                "x": tgt_eef[:, 0].tolist(), "y": tgt_eef[:, 1].tolist(), "z": tgt_eef[:, 2].tolist(),
                "line": {"color": "steelblue", "width": 2, "dash": "dot"},
                "marker": {"size": 2, "color": steps, "colorscale": "Blues", "showscale": False},
            },
            {
                "type": "scatter3d", "mode": "lines+markers", "name": "Predicted EEF",
                "x": pred_eef[:, 0].tolist(), "y": pred_eef[:, 1].tolist(), "z": pred_eef[:, 2].tolist(),
                "line": {"color": "tomato", "width": 4},
                "marker": {"size": 3, "color": steps, "colorscale": "Reds", "showscale": False},
            },
        ]
        eef_err_trace = [{
            "x": steps, "y": eef_err.tolist(), "mode": "lines",
            "name": "EEF error (m)", "line": {"color": "darkorange", "width": 2},
            "fill": "tozeroy", "fillcolor": "rgba(255,165,0,0.15)",
        }]

    # --- Per-joint time series ---
    joint_traces: list[dict] = []
    for i, (label, color) in enumerate(zip(_JOINT_LABELS, _JOINT_COLORS)):
        joint_traces += [
            {
                "x": steps, "y": gt_target[:, i].tolist(),
                "name": f"GT next {label}", "mode": "lines",
                "line": {"color": color, "width": 2},
            },
            {
                "x": steps, "y": pred[:, i].tolist(),
                "name": f"Pred {label}", "mode": "lines",
                "line": {"color": color, "width": 2, "dash": "dash"},
            },
        ]

    # --- GT state vs GT target diff (shows how much the arm actually moves each step) ---
    motion_traces: list[dict] = []
    for i, (label, color) in enumerate(zip(_JOINT_LABELS[:6], _JOINT_COLORS[:6])):
        diff = (gt_target[:, i] - gt_state[:, i]).tolist()
        motion_traces.append({
            "x": steps, "y": diff,
            "name": label, "mode": "lines",
            "line": {"color": color, "width": 1},
        })

    # --- Error traces ---
    error_traces = [
        {
            "x": steps, "y": delta_arm, "name": "|Δarm| rad (j1–j6)", "mode": "lines",
            "line": {"color": "crimson", "width": 2},
            "fill": "tozeroy", "fillcolor": "rgba(220,20,60,0.10)",
        },
        {
            "x": steps, "y": delta_grip, "name": "|Δgripper| rad", "mode": "lines",
            "line": {"color": "darkorchid", "width": 2},
            "yaxis": "y2",
        },
    ]

    # --- Build HTML ---
    plot3d_div   = "<div class='pc'><div id='plot3d'></div></div>" if use_3d else ""
    eef_err_div  = "<div class='pc'><div id='plot_eef_err'></div></div>" if use_3d else ""

    plot3d_js = ""
    eef_err_js = ""
    if use_3d:
        plot3d_js = f"""Plotly.newPlot('plot3d', {json.dumps(traces3d)}, {{
  title: 'EEF 3D — GT current (blue) / GT next state (blue-dot) / Predicted (red)',
  scene: {{xaxis:{{title:'X (m)'}}, yaxis:{{title:'Y (m)'}}, zaxis:{{title:'Z (m)'}}, aspectmode:'data'}},
  height: 600}}, {{responsive: true}});"""
        eef_err_js = f"""Plotly.newPlot('plot_eef_err', {json.dumps(eef_err_trace)}, {{
  title: 'EEF error ‖pred − gt_target‖ per step',
  xaxis:{{title:'Step'}}, yaxis:{{title:'m'}}, height: 280}}, {{responsive: true}});"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Teacher-Forcing — {demo_name}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body  {{ font-family: sans-serif; margin: 20px; background: #fafafa; }}
  h1    {{ color: #333; }}
  h2    {{ color: #666; font-size: .9em; font-weight: normal; margin: 2px 0 12px; }}
  .stat {{ display: inline-block; background: white; border-radius: 6px;
           box-shadow: 0 1px 3px rgba(0,0,0,.15); padding: 10px 20px; margin: 4px; text-align: center; }}
  .stat .val {{ font-size: 1.6em; font-weight: bold; color: #d62728; }}
  .stat .lbl {{ font-size: 0.8em; color: #888; }}
  .pc   {{ background: white; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,.15);
           margin-bottom: 24px; padding: 8px; }}
</style>
</head>
<body>
<h1>Teacher-Forcing: {demo_name}</h1>
<h2>
  Model: pi05_a1_single_arm &nbsp;|&nbsp;
  State: 7D [j1..j6, gripper_rad] &nbsp;|&nbsp;
  Target: cmd[t+1] (next commanded state) &nbsp;|&nbsp;
  Steps: {len(records)}
</h2>

<div style="margin-bottom: 16px">
  <div class="stat"><div class="val">{mean_arm:.4f}</div><div class="lbl">mean |Δarm| (rad)</div></div>
  <div class="stat"><div class="val">{max_arm:.4f}</div><div class="lbl">max |Δarm| (rad)</div></div>
  <div class="stat"><div class="val">{mean_grip:.4f}</div><div class="lbl">mean |Δgripper| (rad)</div></div>
  <div class="stat"><div class="val">{mean_eef * 100:.1f} cm</div><div class="lbl">mean EEF error</div></div>
  <div class="stat"><div class="val">{max_eef * 100:.1f} cm</div><div class="lbl">max EEF error</div></div>
  <div class="stat"><div class="val">{len(records)}</div><div class="lbl">steps</div></div>
</div>

{plot3d_div}
<div class="pc"><div id="plot_joints"></div></div>
<div class="pc"><div id="plot_motion"></div></div>
<div class="pc"><div id="plot_errors"></div></div>
{eef_err_div}

<script>
{plot3d_js}

Plotly.newPlot('plot_joints', {json.dumps(joint_traces)}, {{
  title: 'Per-joint: GT next state (solid) vs Predicted (dashed)',
  xaxis: {{title: 'Step'}},
  yaxis: {{title: 'rad'}},
  height: 480,
}}, {{responsive: true}});

Plotly.newPlot('plot_motion', {json.dumps(motion_traces)}, {{
  title: 'GT motion per step (gt_target[t] − gt_state[t]) — shows ground-truth arm dynamics',
  xaxis: {{title: 'Step'}},
  yaxis: {{title: 'Δrad'}},
  height: 300,
}}, {{responsive: true}});

Plotly.newPlot('plot_errors', {json.dumps(error_traces)}, {{
  title: 'Prediction errors per step',
  xaxis: {{title: 'Step'}},
  yaxis: {{title: '|Δarm| (rad)', side: 'left'}},
  yaxis2: {{title: '|Δgripper| (rad)', overlaying: 'y', side: 'right'}},
  legend: {{x: 0, y: 1}},
  height: 300,
}}, {{responsive: true}});

{eef_err_js}
</script>
</body>
</html>"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    print(f"[TF] Saved HTML: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Teacher-forcing visualization using WebSocket policy server."
    )
    parser.add_argument(
        "--processed-root",
        default=str(ROOT_DIR / "data" / "processed_data" / "pick_twice"),
        help="Root dir with demo_N subdirectories (processed_data layout).",
    )
    parser.add_argument("--demo", default=None, help="Single demo name, e.g. demo_0")
    parser.add_argument("--host",   default="127.0.0.1", help="WebSocket policy server host")
    parser.add_argument("--port",   type=int, default=8000, help="WebSocket policy server port")
    parser.add_argument("--prompt", default="pick_twice")
    parser.add_argument("--max-steps", type=int, default=0, help="0 = all steps")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT_DIR / "data" / "teacher_forcing"),
    )
    # Accepted but ignored (legacy Justfile compat)
    parser.add_argument("--policy-dir",    default=None)
    parser.add_argument("--policy-config", default=None)
    parser.add_argument("--raw-root",      default=None)
    args = parser.parse_args()

    data_root = Path(args.processed_root).expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    print(f"[TF] Connecting to WebSocket policy at ws://{args.host}:{args.port} ...")
    policy = _ws_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    print(f"[TF] Connected. Server metadata: {policy.get_server_metadata()}")

    if args.demo:
        demo_dirs = [data_root / args.demo]
    else:
        demo_dirs = sorted(
            [d for d in data_root.iterdir() if d.is_dir() and d.name.startswith("demo_")],
            key=_demo_sort_key,
        )

    for d in demo_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Demo not found: {d}")

    out_root = Path(args.output_dir).expanduser().resolve()

    for demo_dir in demo_dirs:
        print(f"\n[TF] === {demo_dir.name} ===")
        records = infer_demo(
            demo_dir=demo_dir,
            policy=policy,
            prompt=args.prompt,
            max_steps=args.max_steps,
        )

        out_dir = out_root / demo_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / "trajectory.json"
        json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"[TF] Saved JSON: {json_path}")

        save_html(records, out_dir / "trajectory.html", demo_dir.name)

    print(f"\n[TF] All done. Results in: {out_root}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
