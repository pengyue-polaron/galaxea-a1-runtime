"""LingBot-specific deployment health checks."""

from __future__ import annotations

from pathlib import Path

from galaxea_a1_runtime.apps.lingbot.config import (
    default_config_path,
    load_lingbot_config,
)
from galaxea_a1_runtime.console import ArgumentParser
from galaxea_a1_runtime.runtime.health_checks import (
    Check,
    add_check,
    finish_checks,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def realsense_device_info(serial: str) -> object:
    from galaxea_a1_runtime.hardware.cameras import realsense_device_info as probe

    return probe(serial)


def websocket_open(host: str, port: int, *, timeout_s: float) -> bool:
    try:
        import websockets.sync.client

        with websockets.sync.client.connect(
            f"ws://{host}:{port}",
            compression=None,
            max_size=None,
            ping_interval=None,
            close_timeout=timeout_s,
            open_timeout=timeout_s,
        ) as websocket:
            websocket.recv(timeout=timeout_s)
        return True
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config_path(REPO_ROOT))
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--require-execution",
        action="store_true",
        help="Accepted for compatibility with the composed runtime doctor.",
    )
    args = parser.parse_args(argv)
    config = load_lingbot_config(args.config, repo_root=REPO_ROOT)
    checks: list[Check] = []

    wrist = config.system.cameras.wrist
    if wrist.backend == "realsense":
        try:
            wrist_info = realsense_device_info(wrist.serial)
        except Exception as exc:
            add_check(checks, "wrist_camera", False, str(exc))
        else:
            add_check(checks, "wrist_camera", wrist_info is not None, str(wrist_info))
    else:
        add_check(checks, "wrist_camera", Path(wrist.device).exists(), wrist.device)
    add_check(
        checks,
        "lingbot_server",
        websocket_open(
            config.server.host,
            config.server.port,
            timeout_s=config.server.connect_timeout_s,
        ),
        f"{config.server.host}:{config.server.port}",
    )
    return finish_checks(checks, json_output=args.json)
