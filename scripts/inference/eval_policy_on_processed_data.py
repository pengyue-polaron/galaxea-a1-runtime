#!/usr/bin/env python3
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

# Prefer local workspace modules.
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
THIRD_PARTY_OPENPI = ROOT_DIR / "third_party" / "openpi" / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if THIRD_PARTY_OPENPI.exists() and str(THIRD_PARTY_OPENPI) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_OPENPI))

from datacoach.training import config as train_config_lib
from openpi.policies import policy_config


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

        proprio_seq = np.stack(states[-self._history_len :], axis=0).astype(np.float32, copy=False)
        ts = np.asarray(times[-self._history_len :], dtype=np.float32)
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


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline evaluation on processed pick_twice demos.")
    parser.add_argument(
        "--processed-root",
        default="/home/pengyue/Codespace/DataCoach/data/processed_data/pick_twice",
        help="Processed demo root, containing demo_x directories.",
    )
    parser.add_argument("--policy-dir", required=True, help="Checkpoint directory, e.g. /home/pengyue/29000")
    parser.add_argument("--policy-config", default="pi05_ltc_pick_twice", help="Training config name")
    parser.add_argument("--prompt", default="pick twice", help="Prompt fed to policy")
    parser.add_argument("--max-demos", type=int, default=0, help="0 means all demos")
    parser.add_argument("--max-steps-per-demo", type=int, default=0, help="0 means all available steps")
    parser.add_argument("--stride", type=int, default=1, help="Evaluate one sample every N steps")
    parser.add_argument(
        "--deterministic-noise",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use zero diffusion noise for deterministic evaluation",
    )
    parser.add_argument(
        "--print-per-demo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print per-demo summary rows",
    )
    parser.add_argument("--output-json", default=None, help="Optional output JSON summary path")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    processed_root = Path(args.processed_root).expanduser().resolve()
    if not processed_root.exists():
        raise FileNotFoundError(f"processed root not found: {processed_root}")

    config = train_config_lib.get_config(args.policy_config)
    data_config = config.data.create(config.assets_dirs, config.model)
    policy = policy_config.create_trained_policy(
        train_config=config,
        repack_transforms=data_config.repack_transforms,
        checkpoint_dir=args.policy_dir,
        default_prompt=args.prompt,
    )

    metadata = policy.metadata or {}
    model = getattr(policy, "_model", None)
    action_dim = int(getattr(model, "action_dim", 32))
    bptt_len = int(getattr(model, "bptt_len", 8))
    ltc_dt_s = float(metadata.get("ltc_dt_s", 1.0 / 50.0))
    use_measured_dt = bool(metadata.get("ltc_use_measured_dt", False))
    dt_min_s = float(metadata.get("ltc_dt_min_s", ltc_dt_s * 0.5))
    dt_max_s = float(metadata.get("ltc_dt_max_s", ltc_dt_s * 2.0))
    reset_gap_s = float(metadata.get("ltc_history_reset_gap_s", 1.0))
    episode_prefix = str(metadata.get("ltc_episode_id", "offline_eval"))

    deterministic_noise = None
    if args.deterministic_noise:
        horizon = int(getattr(model, "action_horizon", 10))
        deterministic_noise = np.zeros((horizon, action_dim), dtype=np.float32)

    demo_dirs = sorted([d for d in processed_root.iterdir() if d.is_dir()], key=_demo_sort_key)
    if args.max_demos > 0:
        demo_dirs = demo_dirs[: args.max_demos]
    if not demo_dirs:
        raise RuntimeError(f"No demo directories found under {processed_root}")

    pos_errors_m: list[float] = []
    ori_errors_deg: list[float] = []
    gripper_errors: list[float] = []
    action_l2_errors: list[float] = []
    pred_step_pos_m: list[float] = []
    pred_step_ori_deg: list[float] = []
    pred_step_gripper: list[float] = []

    evaluated_demos = 0
    skipped_demos: list[dict[str, str]] = []
    total_samples = 0

    print(f"[Eval] processed_root={processed_root}")
    print(f"[Eval] policy_config={args.policy_config} checkpoint={args.policy_dir}")
    print(
        "[Eval] LTC metadata "
        f"dt={ltc_dt_s:.4f}s use_measured_dt={use_measured_dt} "
        f"dt_min={dt_min_s:.4f}s dt_max={dt_max_s:.4f}s bptt={bptt_len}"
    )

    for demo_idx, demo_dir in enumerate(demo_dirs):
        try:
            states = _load_pickle(demo_dir / "states.pkl")
            commanded = _load_pickle(demo_dir / "commanded_states.pkl")

            cap0 = cv2.VideoCapture(str(demo_dir / "cam_0_rgb_video.mp4"))
            cap1 = cv2.VideoCapture(str(demo_dir / "cam_1_rgb_video.mp4"))
            if not cap0.isOpened() or not cap1.isOpened():
                raise RuntimeError("Cannot open camera videos.")

            if hasattr(policy, "reset"):
                policy.reset()

            history = LTCHistoryBuilder(
                action_dim=action_dim,
                history_len=bptt_len,
                ltc_dt_s=ltc_dt_s,
                use_measured_dt=use_measured_dt,
                dt_min_s=dt_min_s,
                dt_max_s=dt_max_s,
                reset_gap_s=reset_gap_s,
            )

            max_steps = min(len(states) - 1, len(commanded) - 1)
            if args.max_steps_per_demo > 0:
                max_steps = min(max_steps, args.max_steps_per_demo)
            if max_steps <= 0:
                raise RuntimeError("Not enough states/commands to evaluate.")

            prev_pred_action = None
            demo_pos: list[float] = []
            demo_ori: list[float] = []
            demo_grip: list[float] = []

            for t in range(max_steps):
                ok0, frame0 = cap0.read()
                ok1, frame1 = cap1.read()
                if not ok0 or not ok1 or frame0 is None or frame1 is None:
                    break

                state_vec = _encode_state(states[t])
                target_action = _encode_state(commanded[t + 1])
                timestamp = float(states[t].get("timestamp", time.time()))

                ltc_inputs = history.build(state_vec, timestamp)
                if t % max(1, args.stride) != 0:
                    continue

                obs = {
                    "cam_0": cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB),
                    "cam_1": cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB),
                    "state": state_vec,
                    "proprio_seq": ltc_inputs["proprio_seq"],
                    "time_deltas": ltc_inputs["time_deltas"],
                    "ltc_dt": ltc_inputs["ltc_dt"],
                    "episode_id": f"{episode_prefix}_{demo_dir.name}",
                    "reset": np.bool_(ltc_inputs["reset"]),
                    "action": np.zeros_like(state_vec, dtype=np.float32),
                    "prompt": args.prompt,
                }
                if deterministic_noise is None:
                    pred_dict = policy.infer(obs)
                else:
                    pred_dict = policy.infer(obs, noise=deterministic_noise)
                pred_action = _extract_policy_action(pred_dict, fallback_gripper=float(state_vec[7]))

                pos_err = float(np.linalg.norm(pred_action[:3] - target_action[:3]))
                ori_err = _quat_angle_deg(pred_action[3:7], target_action[3:7])
                grip_err = float(abs(pred_action[7] - target_action[7]))
                action_l2 = float(np.linalg.norm(pred_action - target_action))

                pos_errors_m.append(pos_err)
                ori_errors_deg.append(ori_err)
                gripper_errors.append(grip_err)
                action_l2_errors.append(action_l2)

                demo_pos.append(pos_err)
                demo_ori.append(ori_err)
                demo_grip.append(grip_err)

                if prev_pred_action is not None:
                    pred_step_pos_m.append(float(np.linalg.norm(pred_action[:3] - prev_pred_action[:3])))
                    pred_step_ori_deg.append(_quat_angle_deg(pred_action[3:7], prev_pred_action[3:7]))
                    pred_step_gripper.append(float(abs(pred_action[7] - prev_pred_action[7])))
                prev_pred_action = pred_action
                total_samples += 1

            cap0.release()
            cap1.release()

            evaluated_demos += 1
            if args.print_per_demo and demo_pos:
                print(
                    f"[Demo {demo_dir.name}] n={len(demo_pos)} "
                    f"pos_mean={np.mean(demo_pos):.4f}m ori_mean={np.mean(demo_ori):.2f}deg "
                    f"grip_mean={np.mean(demo_grip):.3f}"
                )
        except Exception as exc:
            skipped_demos.append({"demo": demo_dir.name, "error": str(exc)})

    summary = {
        "processed_root": str(processed_root),
        "policy_config": args.policy_config,
        "policy_dir": str(Path(args.policy_dir).expanduser().resolve()),
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
            "pred_step_gripper": _to_summary(pred_step_gripper),
        },
        "hit_rates": {
            "pos<=0.02m": float(np.mean(np.asarray(pos_errors_m) <= 0.02)) if pos_errors_m else None,
            "pos<=0.05m": float(np.mean(np.asarray(pos_errors_m) <= 0.05)) if pos_errors_m else None,
            "ori<=10deg": float(np.mean(np.asarray(ori_errors_deg) <= 10.0)) if ori_errors_deg else None,
            "ori<=20deg": float(np.mean(np.asarray(ori_errors_deg) <= 20.0)) if ori_errors_deg else None,
            "gripper<=3": float(np.mean(np.asarray(gripper_errors) <= 3.0)) if gripper_errors else None,
            "gripper<=8": float(np.mean(np.asarray(gripper_errors) <= 8.0)) if gripper_errors else None,
        },
        "eval_options": {
            "prompt": args.prompt,
            "max_demos": args.max_demos,
            "max_steps_per_demo": args.max_steps_per_demo,
            "stride": args.stride,
            "deterministic_noise": bool(args.deterministic_noise),
        },
    }

    print("\n[Eval Summary]")
    print(json.dumps(summary, indent=2, ensure_ascii=True))

    if args.output_json:
        out_path = Path(args.output_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        print(f"[Eval] Wrote summary JSON: {out_path}")


if __name__ == "__main__":
    main()
