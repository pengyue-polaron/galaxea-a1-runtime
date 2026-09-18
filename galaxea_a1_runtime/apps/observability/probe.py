"""Require live diagnostics on both sides of the native observation boundary."""

from __future__ import annotations

import argparse
from pathlib import Path
import time


def main() -> int:
    from diagnostic_msgs.msg import DiagnosticArray
    import rclpy
    from galaxea_a1_runtime.configuration.system import load_system_config
    from galaxea_a1_runtime.runtime.ros2 import LEGACY_DIAGNOSTICS_TOPIC

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    system = load_system_config(args.config)
    rclpy.init()
    node = rclpy.create_node("a1_observation_preflight", start_parameter_services=False)
    pending = {LEGACY_DIAGNOSTICS_TOPIC, system.observability.topics.diagnostics}
    subscriptions = [
        node.create_subscription(
            DiagnosticArray, topic, lambda message, name=topic: pending.discard(name), 1
        )
        for topic in tuple(pending)
    ]
    deadline = time.monotonic() + system.observability.startup_timeout_s
    try:
        while pending and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
        if pending:
            raise RuntimeError(
                f"observation diagnostics unavailable: {sorted(pending)}"
            )
    finally:
        for subscription in subscriptions:
            node.destroy_subscription(subscription)
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
