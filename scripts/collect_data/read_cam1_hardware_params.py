#!/usr/bin/env python3
import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2

ROOT_DIR = Path(__file__).resolve().parents[2]


def run_cmd(cmd):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def parse_udev_properties(device):
    if shutil.which("udevadm") is None:
        return None, "udevadm not found in PATH"

    rc, out, err = run_cmd(["udevadm", "info", "--query=property", "--name", device])
    if rc != 0:
        return None, err or out or f"failed to query udev for {device}"

    props = {}
    for line in out.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        props[k.strip()] = v.strip()
    return props, None


def parse_ffmpeg_formats(device):
    if shutil.which("ffmpeg") is None:
        return None, "ffmpeg not found in PATH"

    rc, out, err = run_cmd(["ffmpeg", "-hide_banner", "-f", "v4l2", "-list_formats", "all", "-i", device])
    text = (out + "\n" + err).strip()
    if not text:
        return None, f"no output from ffmpeg for {device}"

    if rc != 0 and "Permission denied" in text:
        return None, (
            f"permission denied for {device}. "
            f"Try: sudo usermod -aG video $USER && re-login"
        )

    format_lines = []
    for line in text.splitlines():
        if "Compressed:" in line or "Raw       :" in line:
            format_lines.append(line.strip())

    formats = []
    for line in format_lines:
        body = line.split("]", 1)[-1].strip() if "]" in line else line
        left, sep, res_part = body.rpartition(":")
        if not sep:
            continue

        left_parts = [p.strip() for p in left.split(":", 2)]
        if len(left_parts) != 3:
            continue

        mode_text, pixel_format, description = left_parts
        mode = "unknown"
        if "Compressed" in mode_text:
            mode = "compressed"
        elif "Raw" in mode_text:
            mode = "raw"

        resolutions = sorted(
            {
                (int(m.group(1)), int(m.group(2)))
                for m in re.finditer(r"(\d+)x(\d+)", res_part)
            }
        )

        formats.append(
            {
                "mode": mode,
                "pixel_format": pixel_format,
                "description": description,
                "resolutions": resolutions,
            }
        )

    if not formats:
        return None, text
    return formats, None


def probe_capture_runtime(device):
    if device.startswith("/dev/video") and device[10:].isdigit():
        source = int(device[10:])
    else:
        source = device

    cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None, f"cannot open {device} via OpenCV"

    frame = None
    for _ in range(40):
        ok, img = cap.read()
        if ok and img is not None:
            frame = img
            break
        time.sleep(0.02)

    width_prop = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height_prop = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps_prop = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()

    if frame is not None:
        h, w = frame.shape[:2]
        frame_resolution = f"{w}x{h}"
    else:
        frame_resolution = "unknown (no frame captured)"

    runtime = {
        "opencv_reported_resolution": f"{width_prop}x{height_prop}",
        "opencv_reported_fps": f"{fps_prop:.2f}",
        "captured_frame_resolution": frame_resolution,
    }
    return runtime, None


def print_summary(device, props, formats, runtime):
    print(f"device: {device}")
    print()
    print("[hardware]")
    keys = [
        "ID_V4L_PRODUCT",
        "ID_VENDOR_ID",
        "ID_MODEL_ID",
        "ID_SERIAL_SHORT",
        "ID_USB_DRIVER",
        "ID_BUS",
        "ID_PATH",
    ]
    for k in keys:
        print(f"{k}: {props.get(k, 'N/A')}")

    print()
    print("[supported_formats_and_resolutions]")
    for item in formats:
        resolutions = " ".join([f"{w}x{h}" for w, h in item["resolutions"]])
        print(
            f"- mode={item['mode']} pixel_format={item['pixel_format']} "
            f"desc={item['description']}"
        )
        print(f"  resolutions: {resolutions}")

    print()
    print("[runtime_capture]")
    print(f"opencv_reported_resolution: {runtime['opencv_reported_resolution']}")
    print(f"opencv_reported_fps: {runtime['opencv_reported_fps']}")
    print(f"captured_frame_resolution: {runtime['captured_frame_resolution']}")


def main():
    parser = argparse.ArgumentParser(
        description="Read hardware params of cam1 directly from Linux video device."
    )
    parser.add_argument(
        "--device",
        default="/dev/video0",
        help="Video device path for cam1 (default: /dev/video0).",
    )
    args = parser.parse_args()

    device = args.device
    if not Path(device).exists():
        print(f"[ERROR] Device not found: {device}")
        return 2

    props, err = parse_udev_properties(device)
    if err is not None:
        print(f"[ERROR] udev: {err}")
        return 2

    formats, err = parse_ffmpeg_formats(device)
    if err is not None:
        print(f"[ERROR] ffmpeg: {err}")
        return 2

    runtime, err = probe_capture_runtime(device)
    if err is not None:
        print(f"[ERROR] runtime capture: {err}")
        return 2

    print_summary(device, props, formats, runtime)
    return 0


if __name__ == "__main__":
    sys.exit(main())
