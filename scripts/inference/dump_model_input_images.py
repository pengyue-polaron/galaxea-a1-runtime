#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import zmq


def _parse_camera_ts(raw: bytes) -> float | None:
    try:
        ts = float(raw.decode("ascii", errors="strict"))
    except Exception:
        return None
    if ts > 1e12:
        ts = ts / 1e9
    return ts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dump camera stream frames that are sent to model input."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Camera stream host")
    parser.add_argument("--port", type=int, default=5558, help="Camera stream port")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save dumped JPEG frames",
    )
    parser.add_argument(
        "--every-n",
        type=int,
        default=30,
        help="Save one frame every N frames for each camera (default: 30)",
    )
    parser.add_argument(
        "--max-per-cam",
        type=int,
        default=200,
        help="Max saved frames per camera (<=0 means unlimited)",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=0.0,
        help="Capture duration in seconds (<=0 means run until Ctrl+C)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_root = Path(args.output_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    every_n = max(1, int(args.every_n))
    max_per_cam = int(args.max_per_cam)

    context = zmq.Context()
    sub = context.socket(zmq.SUB)
    sub.connect(f"tcp://{args.host}:{args.port}")
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    sub.setsockopt(zmq.RCVTIMEO, 200)

    seen = {}
    saved = {}
    start = time.time()

    print(f"[DumpModelInput] SUB connected to tcp://{args.host}:{args.port}")
    print(
        f"[DumpModelInput] output={out_root} every_n={every_n} max_per_cam={max_per_cam} "
        f"duration_s={args.duration_s}"
    )

    try:
        while True:
            if args.duration_s > 0 and (time.time() - start) >= args.duration_s:
                break

            try:
                parts = sub.recv_multipart()
            except zmq.Again:
                continue
            if len(parts) != 3:
                continue

            cam_id = parts[0].decode("utf-8", errors="replace")
            cam_ts = _parse_camera_ts(parts[1])
            if cam_ts is None:
                continue
            jpeg_bytes = parts[2]

            seen[cam_id] = seen.get(cam_id, 0) + 1
            if seen[cam_id] % every_n != 0:
                continue

            saved_count = saved.get(cam_id, 0)
            if max_per_cam > 0 and saved_count >= max_per_cam:
                continue

            cam_dir = out_root / cam_id
            cam_dir.mkdir(parents=True, exist_ok=True)
            out_path = cam_dir / f"{cam_ts:.6f}_{saved_count:06d}.jpg"
            out_path.write_bytes(jpeg_bytes)
            saved[cam_id] = saved_count + 1

            if saved[cam_id] % 20 == 0:
                print(f"[DumpModelInput] {cam_id}: saved={saved[cam_id]} seen={seen[cam_id]}")
    except KeyboardInterrupt:
        pass
    finally:
        sub.close(0)
        context.term()

    print("[DumpModelInput] done")
    all_cams = sorted(set(seen.keys()) | set(saved.keys()))
    for cam_id in all_cams:
        print(f"  - {cam_id}: seen={seen.get(cam_id, 0)}, saved={saved.get(cam_id, 0)}")


if __name__ == "__main__":
    main()
