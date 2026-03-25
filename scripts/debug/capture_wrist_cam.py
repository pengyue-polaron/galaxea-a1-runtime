#!/usr/bin/env python3
"""Capture two frames from the wrist camera (cam_1) via ZMQ and save as JPG.

Usage:
    python capture_wrist_cam.py
    python capture_wrist_cam.py --port 5558 --out /tmp/wrist
    python capture_wrist_cam.py --cam cam_0   # switch to main camera
"""
import argparse
import os
import time

import cv2
import numpy as np
import zmq


def decode_jpeg(data: bytes):
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def capture(cam_id: str, port: int, host: str, n: int, out_dir: str, timeout: float):
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.RCVHWM, 10)
    sub.connect(f"tcp://{host}:{port}")
    sub.setsockopt_string(zmq.SUBSCRIBE, "")

    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    deadline = time.monotonic() + timeout

    print(f"[capture] listening on tcp://{host}:{port} for cam_id='{cam_id}' ...")
    while saved < n:
        if time.monotonic() > deadline:
            print(f"[capture] timeout after {timeout}s — got {saved}/{n} frames")
            break
        try:
            parts = sub.recv_multipart(flags=zmq.NOBLOCK)
        except zmq.Again:
            time.sleep(0.005)
            continue

        if len(parts) != 3:
            continue
        recv_id = parts[0].decode("utf-8", errors="replace")
        if recv_id != cam_id:
            continue

        img = decode_jpeg(parts[2])
        if img is None:
            print("[capture] failed to decode frame")
            continue

        path = os.path.join(out_dir, f"{cam_id}_frame{saved:02d}.jpg")
        cv2.imwrite(path, img)
        print(f"[capture] saved {path}  shape={img.shape}")
        saved += 1

    sub.close()
    ctx.term()
    print(f"[capture] done. {saved} frame(s) saved to {out_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam",     default="cam_1", help="Camera ID (cam_0 or cam_1)")
    parser.add_argument("--port",    type=int, default=5558)
    parser.add_argument("--host",    default="127.0.0.1")
    parser.add_argument("--n",       type=int, default=2, help="Number of frames to capture")
    parser.add_argument("--out",     default="/tmp/wrist_capture")
    parser.add_argument("--timeout", type=float, default=15.0, help="Seconds to wait")
    args = parser.parse_args()
    capture(args.cam, args.port, args.host, args.n, args.out, args.timeout)


if __name__ == "__main__":
    main()
