#!/usr/bin/env python3
"""Teacher-forcing with pi05_droid: feed training images → predicted EEF delta trajectory.

For each step in a demo:
  - Feed GT images (cam_0, cam_1) + GT joint state → pi05_droid WebSocket server
  - Get predicted 8D EEF delta action: [dx,dy,dz, rx,ry,rz, _, gripper_01]
  - Accumulate position deltas → predicted 3D EEF trajectory
  - Compute GT EEF trajectory via pinocchio FK on joint angles

Output per demo:
  <output_dir>/<demo_name>/trajectory.json
  <output_dir>/<demo_name>/trajectory.html   (interactive Plotly)

Requires:
  pi05_droid WebSocket server running on port 8000 (just policy-droid)

Usage:
  just droid-teacher-forcing
  just droid-teacher-forcing demo_0_20260227_225247
  just droid-teacher-forcing demo_0_20260227_225247 -- --max-steps 100 --pos-scale 0.5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import cv2
import numpy as np
import pinocchio as pin

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

# openpi_client must be on PYTHONPATH
from openpi_client import websocket_client_policy as _ws_policy

# ---------------------------------------------------------------------------
# Forward kinematics (Pinocchio) — reused from teacher_forcing_infer.py
# ---------------------------------------------------------------------------

_A1_URDF = ROOT_DIR / "third_party/A1_SDK/install/share/mobiman/urdf/A1/urdf/A1_URDF_0607_0028.urdf"
_fk_model = None
_fk_data = None
_fk_ee_frame_id = None


def _get_fk_model():
    global _fk_model, _fk_data, _fk_ee_frame_id
    if _fk_model is not None:
        return _fk_model, _fk_data, _fk_ee_frame_id
    model = pin.buildModelFromUrdf(str(_A1_URDF))
    data = model.createData()
    ee_name = "arm_joint6"
    frame_id = next(
        (i for i, f in enumerate(model.frames) if f.name == ee_name),
        model.nframes - 1,
    )
    _fk_model, _fk_data, _fk_ee_frame_id = model, data, frame_id
    return model, data, frame_id


def joints_to_eef(joints6: np.ndarray) -> np.ndarray:
    """Return EEF xyz (m) from 6 arm joint angles via FK."""
    model, data, frame_id = _get_fk_model()
    q = pin.neutral(model)
    q[: min(len(joints6), model.nq)] = joints6[: model.nq]
    pin.framesForwardKinematics(model, data, q)
    return np.array(data.oMf[frame_id].translation)


# ---------------------------------------------------------------------------
# Gripper normalisation
# ---------------------------------------------------------------------------

# A1 gripper joint (rad) range from training data: -1.62 (open) → -0.55 (closed)
_G_CLOSED_RAD = -0.55
_G_OPEN_RAD = -1.62


def gripper_rad_to_01(rad: float) -> float:
    span = _G_OPEN_RAD - _G_CLOSED_RAD
    return float(np.clip((rad - _G_CLOSED_RAD) / span, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _parse_state(entry) -> tuple[np.ndarray | None, float]:
    """Parse commanded_states entry → (joints6, gripper_rad, timestamp)."""
    data = entry.get("data", entry)
    joints7 = data.get("joint", None)
    if joints7 is None:
        return None, 0.0
    joints7 = np.asarray(joints7, dtype=np.float32)
    # joints7: [j1..j6, gripper_rad]
    joints6 = joints7[:6]
    gripper_rad = float(joints7[6]) if len(joints7) > 6 else 0.0
    ts = float(entry.get("timestamp", 0.0))
    return joints6, gripper_rad, ts


# ---------------------------------------------------------------------------
# Inference on one demo
# ---------------------------------------------------------------------------

def infer_demo(
    *,
    demo_dir: Path,
    policy,
    prompt: str,
    max_steps: int,
    pos_scale: float,
    flip_y: bool,
) -> list[dict]:
    cmd_states = _load_pickle(demo_dir / "commanded_states.pkl")
    n_steps = len(cmd_states) if max_steps <= 0 else min(len(cmd_states), max_steps)

    cap0 = cv2.VideoCapture(str(demo_dir / "cam_0_rgb_video.mp4"))
    cap1 = cv2.VideoCapture(str(demo_dir / "cam_1_rgb_video.mp4"))
    if not cap0.isOpened() or not cap1.isOpened():
        raise RuntimeError(f"Cannot open videos in {demo_dir}")

    n_frames = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT))
    print(
        f"[DROID-TF] {demo_dir.name}  steps={n_steps}  "
        f"cmd_states={len(cmd_states)}  cam_frames={n_frames}"
    )

    # Predicted EEF position accumulated from step 0 FK origin
    pred_pos = None   # initialised at first step from FK
    records = []

    for t in range(n_steps):
        ok0, frame0 = cap0.read()
        ok1, frame1 = cap1.read()
        if not ok0 or not ok1:
            print(f"[DROID-TF] Video ended early at t={t}")
            break

        result = _parse_state(cmd_states[t])
        if result[0] is None:
            continue
        joints6, gripper_rad, ts = result

        # Initialise predicted position at first step's FK position
        if pred_pos is None:
            pred_pos = joints_to_eef(joints6).copy()

        # GT EEF via FK
        gt_eef = joints_to_eef(joints6)

        # Resize images to 224x224 RGB
        img0 = cv2.resize(cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB), (224, 224))
        img1 = cv2.resize(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB), (224, 224))

        # Pad A1 6-DOF arm to DROID 7-DOF
        joint7 = np.zeros(7, dtype=np.float64)
        joint7[:6] = joints6
        gripper_01 = gripper_rad_to_01(gripper_rad)

        obs = {
            "observation/exterior_image_1_left": img0,
            "observation/wrist_image_left":      img1,
            "observation/joint_position":        joint7,
            "observation/gripper_position":      np.array([gripper_01]),
            "prompt": prompt,
        }

        action_dict = policy.infer(obs)
        actions = np.asarray(action_dict["actions"], dtype=np.float64)
        if actions.ndim == 1:
            actions = actions[np.newaxis, :]

        # First predicted step: [dx,dy,dz, rx,ry,rz, _, gripper_01]
        a = actions[0]
        delta_pos = a[0:3] * pos_scale
        if flip_y:
            delta_pos[1] = -delta_pos[1]
        gripper_pred_01 = float(np.clip(a[7], 0.0, 1.0))

        pred_pos = pred_pos + delta_pos

        records.append({
            "step":           t,
            "timestamp":      ts,
            "joints6":        joints6.tolist(),
            "gripper_rad_gt": gripper_rad,
            "gripper_01_gt":  gripper_01,
            "gt_eef":         gt_eef.tolist(),
            "pred_eef":       pred_pos.tolist(),
            "delta_pos":      delta_pos.tolist(),
            "delta_rot":      a[3:6].tolist(),
            "gripper_pred_01": gripper_pred_01,
            "action_mag":     float(np.linalg.norm(delta_pos)),
        })

        if t % 20 == 0:
            eef_err = float(np.linalg.norm(pred_pos - gt_eef))
            print(
                f"  step {t:04d}  Δpos=({delta_pos[0]:.3f},{delta_pos[1]:.3f},{delta_pos[2]:.3f})"
                f"  pred_eef=({pred_pos[0]:.3f},{pred_pos[1]:.3f},{pred_pos[2]:.3f})"
                f"  gt_eef=({gt_eef[0]:.3f},{gt_eef[1]:.3f},{gt_eef[2]:.3f})"
                f"  drift={eef_err:.3f}m"
                f"  grip={gripper_pred_01:.2f}"
            )

    cap0.release()
    cap1.release()
    print(f"[DROID-TF] Done. {len(records)} steps.")
    return records


# ---------------------------------------------------------------------------
# HTML visualisation
# ---------------------------------------------------------------------------

def save_html(records: list[dict], path: Path, demo_name: str, prompt: str):
    if not records:
        return

    steps       = [r["step"]            for r in records]
    gt_eef      = np.array([r["gt_eef"]      for r in records])   # (T,3)
    pred_eef    = np.array([r["pred_eef"]    for r in records])   # (T,3)
    delta_pos   = np.array([r["delta_pos"]   for r in records])   # (T,3)
    action_mag  = [r["action_mag"]           for r in records]
    grip_gt     = [r["gripper_01_gt"]        for r in records]
    grip_pred   = [r["gripper_pred_01"]      for r in records]

    # Normalise both trajectories to a shared origin (t=0 GT position)
    # so they can be visually compared in the same coordinate system.
    origin = gt_eef[0].copy()
    gt_eef_n   = gt_eef   - origin
    pred_eef_n = pred_eef - origin

    drift = np.linalg.norm(pred_eef_n - gt_eef_n, axis=1)
    mean_action_mag = float(np.mean(action_mag))
    final_drift     = float(drift[-1]) if len(drift) else 0.0

    traces3d = [
        {
            "type": "scatter3d", "mode": "lines+markers", "name": "GT EEF (FK, centred at t=0)",
            "x": gt_eef_n[:,0].tolist(), "y": gt_eef_n[:,1].tolist(), "z": gt_eef_n[:,2].tolist(),
            "line": {"color": "royalblue", "width": 3},
            "marker": {"size": 3, "color": steps, "colorscale": "Blues", "showscale": False},
        },
        {
            "type": "scatter3d", "mode": "lines+markers", "name": "Predicted EEF (accumulated deltas, centred at t=0)",
            "x": pred_eef_n[:,0].tolist(), "y": pred_eef_n[:,1].tolist(), "z": pred_eef_n[:,2].tolist(),
            "line": {"color": "tomato", "width": 3},
            "marker": {"size": 3, "color": steps, "colorscale": "Reds", "showscale": False},
        },
        {
            "type": "scatter3d", "mode": "markers", "name": "Shared start (origin)",
            "x": [0], "y": [0], "z": [0],
            "marker": {"size": 10, "color": "green", "symbol": "diamond"},
        },
        {
            "type": "scatter3d", "mode": "markers", "name": "GT end",
            "x": [gt_eef_n[-1,0]], "y": [gt_eef_n[-1,1]], "z": [gt_eef_n[-1,2]],
            "marker": {"size": 8, "color": "royalblue", "symbol": "square"},
        },
        {
            "type": "scatter3d", "mode": "markers", "name": "Pred end",
            "x": [pred_eef_n[-1,0]], "y": [pred_eef_n[-1,1]], "z": [pred_eef_n[-1,2]],
            "marker": {"size": 8, "color": "tomato", "symbol": "square"},
        },
    ]

    eef_traces = []
    for i, ax in enumerate(["X", "Y", "Z"]):
        c = ["royalblue", "seagreen", "darkorange"][i]
        eef_traces += [
            {"x": steps, "y": gt_eef_n[:,i].tolist(),   "name": f"GT EEF {ax} (centred)",
             "mode": "lines", "line": {"color": c, "width": 2}},
            {"x": steps, "y": pred_eef_n[:,i].tolist(),  "name": f"Pred EEF {ax} (centred)",
             "mode": "lines", "line": {"color": c, "width": 2, "dash": "dash"}},
        ]

    delta_traces = []
    for i, ax in enumerate(["dX", "dY", "dZ"]):
        c = ["royalblue", "seagreen", "darkorange"][i]
        delta_traces.append({
            "x": steps, "y": delta_pos[:,i].tolist(), "name": ax,
            "mode": "lines", "line": {"color": c, "width": 2},
        })

    gripper_traces = [
        {"x": steps, "y": grip_gt,   "name": "GT gripper [0=closed,1=open]",
         "mode": "lines", "line": {"color": "royalblue", "width": 2}},
        {"x": steps, "y": grip_pred, "name": "Pred gripper",
         "mode": "lines", "line": {"color": "tomato", "width": 2, "dash": "dash"}},
    ]

    drift_trace = [{
        "x": steps, "y": drift.tolist(), "name": "EEF drift pred vs GT (m)",
        "mode": "lines", "type": "scatter",
        "line": {"color": "crimson", "width": 2},
        "fill": "tozeroy", "fillcolor": "rgba(220,20,60,0.1)",
    }]

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>DROID Teacher-Forcing — {demo_name}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body{{font-family:sans-serif;margin:20px;background:#fafafa}}
  h1{{color:#333}}
  h2{{color:#666;font-size:.9em;font-weight:normal;margin:2px 0}}
  .stat{{display:inline-block;background:white;border-radius:6px;
         box-shadow:0 1px 3px rgba(0,0,0,.15);padding:10px 20px;margin:4px;text-align:center}}
  .stat .val{{font-size:1.6em;font-weight:bold;color:#d62728}}
  .stat .lbl{{font-size:.8em;color:#888}}
  .pc{{background:white;border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,.15);margin-bottom:24px;padding:8px}}
</style></head><body>
<h1>DROID Teacher-Forcing: {demo_name}</h1>
<h2>Model: pi05_droid (zero-shot)&nbsp;|&nbsp;Prompt: "{prompt}"</h2>
<h2>Input: GT images + GT joints → DROID format &nbsp;|&nbsp;
   GT traj: FK from joint angles (blue) &nbsp;|&nbsp;
   Pred traj: accumulated EEF deltas (red)</h2>
<div style="margin-bottom:16px">
  <div class="stat"><div class="val">{mean_action_mag*100:.1f} cm</div><div class="lbl">mean |Δpos| per step</div></div>
  <div class="stat"><div class="val">{final_drift*100:.1f} cm</div><div class="lbl">final drift (pred vs GT)</div></div>
  <div class="stat"><div class="val">{len(records)}</div><div class="lbl">steps</div></div>
</div>
<div class="pc"><div id="plot3d"></div></div>
<div class="pc"><div id="plot_eef_time"></div></div>
<div class="pc"><div id="plot_delta"></div></div>
<div class="pc"><div id="plot_gripper"></div></div>
<div class="pc"><div id="plot_drift"></div></div>
<script>
Plotly.newPlot('plot3d',{json.dumps(traces3d)},{{
  title:'EEF 3D: GT (blue) vs Predicted accumulated deltas (red)',
  scene:{{xaxis:{{title:'X (m)'}},yaxis:{{title:'Y (m)'}},zaxis:{{title:'Z (m)'}},aspectmode:'data'}},
  height:650}},{{responsive:true}});
Plotly.newPlot('plot_eef_time',{json.dumps(eef_traces)},{{
  title:'EEF over time: GT (solid) vs Pred accumulated (dashed)',
  xaxis:{{title:'Step'}},yaxis:{{title:'Position (m)'}},height:380}},{{responsive:true}});
Plotly.newPlot('plot_delta',{json.dumps(delta_traces)},{{
  title:'Predicted EEF position delta per step (dx,dy,dz)',
  xaxis:{{title:'Step'}},yaxis:{{title:'Delta (m)'}},height:300}},{{responsive:true}});
Plotly.newPlot('plot_gripper',{json.dumps(gripper_traces)},{{
  title:'Gripper: GT (blue) vs Predicted (red dashed) — [0=closed, 1=open]',
  xaxis:{{title:'Step'}},yaxis:{{title:'[0=closed, 1=open]',range:[-0.05,1.05]}},height:280}},{{responsive:true}});
Plotly.newPlot('plot_drift',{json.dumps(drift_trace)},{{
  title:'Cumulative drift: ‖pred_eef − gt_eef‖ (pred starts at GT position at t=0)',
  xaxis:{{title:'Step'}},yaxis:{{title:'m'}},height:260}},{{responsive:true}});
</script></body></html>"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    print(f"[DROID-TF] Saved HTML: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", default="/home/jolia/DataCoach/data/processed_data/swap")
    parser.add_argument("--demo",           default=None, help="Specific demo name; omit to run all")
    parser.add_argument("--server-host",    default="localhost")
    parser.add_argument("--server-port",    type=int, default=8000)
    parser.add_argument("--prompt",         default="swap the position of the marker and the yellow block through the white plate")
    parser.add_argument("--max-steps",      type=int, default=0)
    parser.add_argument("--pos-scale",      type=float, default=1.0,
                        help="Scale applied to position deltas before accumulation")
    parser.add_argument("--flip-y",         action="store_true", default=True,
                        help="Negate Y delta (DROID→A1 frame convention, default on)")
    parser.add_argument("--no-flip-y",      action="store_false", dest="flip_y")
    parser.add_argument("--output-dir",     default="/home/pengyue/Codespace/DataCoach/data/droid_teacher_forcing")
    args = parser.parse_args()

    processed_root = Path(args.processed_root)

    # Discover demos
    if args.demo:
        demos = [processed_root / args.demo]
    else:
        demos = sorted(
            d for d in processed_root.iterdir()
            if d.is_dir() and (d / "commanded_states.pkl").exists()
        )

    if not demos:
        print(f"[DROID-TF] No demos found under {processed_root}")
        return

    print(f"[DROID-TF] Connecting to pi05_droid at ws://{args.server_host}:{args.server_port} ...")
    policy = _ws_policy.WebsocketClientPolicy(host=args.server_host, port=args.server_port)
    print(f"[DROID-TF] Server metadata: {policy.get_server_metadata()}")
    print(f"[DROID-TF] Prompt: {args.prompt!r}")
    print(f"[DROID-TF] pos_scale={args.pos_scale}  flip_y={args.flip_y}")
    print(f"[DROID-TF] Running {len(demos)} demo(s)...\n")

    for demo_dir in demos:
        out_dir = Path(args.output_dir) / demo_dir.name
        records = infer_demo(
            demo_dir=demo_dir,
            policy=policy,
            prompt=args.prompt,
            max_steps=args.max_steps,
            pos_scale=args.pos_scale,
            flip_y=args.flip_y,
        )
        if not records:
            continue

        json_path = out_dir / "trajectory.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"[DROID-TF] Saved JSON: {json_path}")

        save_html(records, out_dir / "trajectory.html", demo_dir.name, args.prompt)
        print()


if __name__ == "__main__":
    main()
