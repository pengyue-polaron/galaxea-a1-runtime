#!/usr/bin/env python3
import argparse
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from omegaconf import OmegaConf
except Exception:
    OmegaConf = None

try:
    import yaml
except Exception:
    yaml = None

try:
    import pyrealsense2 as rs
except Exception:
    rs = None

try:
    from read_cam1_hardware_params import parse_udev_properties
except Exception:
    parse_udev_properties = None


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _load_config(cfg_path: Path):
    if OmegaConf is not None:
        return OmegaConf.load(str(cfg_path))
    if yaml is not None:
        with cfg_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    raise RuntimeError("Neither omegaconf nor pyyaml is available in this python env.")


def _resolve_video_device(device):
    device_str = str(device)
    if device_str.isdigit():
        return f"/dev/video{device_str}"
    return device_str


def _format_udev_serial(props):
    if not props:
        return None
    for key in ("ID_SERIAL_SHORT", "ID_SERIAL", "ID_USB_SERIAL_SHORT"):
        value = props.get(key)
        if value:
            return value
    return None


def _test_realsense_camera(camera_cfg):
    cam_id = str(_cfg_get(camera_cfg, "id", "cam"))
    configured_serial = _cfg_get(camera_cfg, "serial", None)

    if rs is None:
        print(f"[FAIL] {cam_id}: pyrealsense2 is not installed")
        return False

    try:
        ctx = rs.context()
        devices = list(ctx.query_devices())
    except Exception as exc:
        print(f"[FAIL] {cam_id}: failed to enumerate RealSense devices: {exc}")
        return False

    found = []
    for dev in devices:
        try:
            name = dev.get_info(rs.camera_info.name)
        except Exception:
            name = "unknown"
        try:
            serial = dev.get_info(rs.camera_info.serial_number)
        except Exception:
            serial = None
        found.append({"name": name, "serial": serial})

    if configured_serial:
        matched = next((item for item in found if item["serial"] == str(configured_serial)), None)
        if matched is None:
            available = ", ".join(item["serial"] or "unknown" for item in found) or "<none>"
            print(
                f"[FAIL] {cam_id}: configured serial={configured_serial}, "
                f"detected_serials={available}"
            )
            return False

        print(
            f"[PASS] {cam_id}: backend=realsense, serial={matched['serial']}, "
            f"name={matched['name']}"
        )
        return True

    if not found:
        print(f"[FAIL] {cam_id}: no RealSense devices detected")
        return False

    first = found[0]
    print(
        f"[PASS] {cam_id}: backend=realsense, serial={first['serial'] or 'unknown'}, "
        f"name={first['name']}"
    )
    if len(found) > 1:
        serials = ", ".join(item["serial"] or "unknown" for item in found)
        print(f"       detected_serials: {serials}")
    return True


def _test_opencv_camera(camera_cfg):
    cam_id = str(_cfg_get(camera_cfg, "id", "cam"))
    configured_device = _cfg_get(camera_cfg, "device", 0)
    device_path = _resolve_video_device(configured_device)

    if not Path(device_path).exists():
        print(f"[FAIL] {cam_id}: device node not found: {device_path}")
        return False

    props = None
    if parse_udev_properties is not None and shutil.which("udevadm") is not None:
        props, err = parse_udev_properties(device_path)
        if err is not None:
            print(f"[FAIL] {cam_id}: udev query failed for {device_path}: {err}")
            return False

    serial = _format_udev_serial(props)
    product = props.get("ID_V4L_PRODUCT", "unknown") if props else "unknown"
    print(
        f"[PASS] {cam_id}: backend=opencv, device={device_path}, "
        f"serial={serial or 'unknown'}, product={product}"
    )
    return True


def test_one_camera(camera_cfg):
    cam_id = str(_cfg_get(camera_cfg, "id", "cam"))
    enabled = bool(_cfg_get(camera_cfg, "enabled", True))

    if not enabled:
        print(f"[SKIP] {cam_id}: disabled")
        return True

    backend = str(_cfg_get(camera_cfg, "backend", "opencv")).lower()
    if backend == "realsense":
        return _test_realsense_camera(camera_cfg)
    if backend == "opencv":
        return _test_opencv_camera(camera_cfg)

    print(f"[FAIL] {cam_id}: unsupported backend '{backend}'")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Check configured cameras by enumerating device info only; do not open streams."
    )
    parser.add_argument(
        "--config",
        default=str(ROOT_DIR / "configs" / "drag_replay.yaml"),
        help="YAML config path (expects camera_server.cameras).",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=6.0,
        help="Deprecated compatibility flag. Ignored.",
    )
    parser.add_argument(
        "--save-dir",
        default=str(ROOT_DIR / "outputs" / "camera_probe"),
        help="Deprecated compatibility flag. Ignored.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Deprecated compatibility flag. Ignored.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.exists():
        print(f"[ERROR] Config not found: {cfg_path}")
        return 2

    try:
        cfg = _load_config(cfg_path)
    except Exception as exc:
        print(f"[ERROR] Failed to load config: {exc}")
        return 2

    camera_cfgs = _cfg_get(_cfg_get(cfg, "camera_server", {}), "cameras", None)
    if not camera_cfgs:
        print(f"[ERROR] No camera_server.cameras in {cfg_path}")
        return 2

    print(f"[INFO] Config: {cfg_path}")
    print(f"[INFO] Cameras: {len(camera_cfgs)}")
    print("[INFO] Mode: enumerate device info only; no stream open")

    ok = True
    for cam_cfg in camera_cfgs:
        ok = test_one_camera(cam_cfg) and ok

    if ok:
        print("[SUMMARY] Camera connection test passed.")
        return 0

    print("[SUMMARY] Camera connection test failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
