#!/usr/bin/env python3
import argparse
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


def _find_camera_cfg(cfg, camera_id: str):
    camera_cfgs = _cfg_get(_cfg_get(cfg, "camera_server", {}), "cameras", None)
    if not camera_cfgs:
        raise RuntimeError("No camera_server.cameras found in config.")

    for cam_cfg in camera_cfgs:
        if str(_cfg_get(cam_cfg, "id", "")) == camera_id:
            return cam_cfg
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Read one camera config from DataCoach YAML and print key params."
    )
    parser.add_argument(
        "--config",
        default=str(ROOT_DIR / "configs" / "drag_replay.yaml"),
        help="YAML config path (expects camera_server.cameras).",
    )
    parser.add_argument(
        "--camera-id",
        default="cam_1",
        help="Camera id to read (default: cam_1).",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.exists():
        print(f"[ERROR] Config not found: {cfg_path}")
        return 2

    try:
        cfg = _load_config(cfg_path)
        cam_cfg = _find_camera_cfg(cfg, args.camera_id)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 2

    if cam_cfg is None:
        print(f"[ERROR] Camera '{args.camera_id}' not found in {cfg_path}")
        return 2

    width = int(_cfg_get(cam_cfg, "width", -1))
    height = int(_cfg_get(cam_cfg, "height", -1))
    backend = str(_cfg_get(cam_cfg, "backend", "opencv"))
    fps = int(_cfg_get(cam_cfg, "fps", -1))
    enabled = bool(_cfg_get(cam_cfg, "enabled", True))
    serial = _cfg_get(cam_cfg, "serial", None)
    device = _cfg_get(cam_cfg, "device", None)
    backend_api = _cfg_get(cam_cfg, "backend_api", None)

    print(f"config: {cfg_path}")
    print(f"camera_id: {args.camera_id}")
    print(f"enabled: {enabled}")
    print(f"backend: {backend}")
    print(f"resolution: {width}x{height}")
    print(f"fps: {fps}")
    if serial is not None:
        print(f"serial: {serial}")
    if device is not None:
        print(f"device: {device}")
    if backend_api is not None:
        print(f"backend_api: {backend_api}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
