#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Print leader-arm joint angles in real time.

Example:

```shell
lerobot-print-leader-joints \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=my_leader \
    --fps=30
```
"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import draccus

from lerobot.teleoperators import (  # noqa: F401
    TeleoperatorConfig,
    bi_openarm_leader,
    bi_so_leader,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    openarm_leader,
    openarm_mini,
    so_leader,
    unitree_g1,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, move_cursor_up


def _extract_joint_angles(action: dict[str, object]) -> dict[str, float]:
    # Most arm teleops expose joint angles as ".pos"; some expose ".q".
    joint_angles = {
        key: float(value)
        for key, value in action.items()
        if isinstance(value, int | float) and not isinstance(value, bool) and (key.endswith(".pos") or key.endswith(".q"))
    }
    if joint_angles:
        return joint_angles

    return {
        key: float(value)
        for key, value in action.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }


@dataclass
class PrintLeaderJointsConfig:
    teleop: TeleoperatorConfig
    fps: int = 30
    duration_s: float | None = None
    calibrate: bool = True
    decimal_places: int = 2


@draccus.wrap()
def print_leader_joints(cfg: PrintLeaderJointsConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    if cfg.fps <= 0:
        raise ValueError(f"`fps` must be > 0, got {cfg.fps}.")
    if cfg.decimal_places < 0:
        raise ValueError(f"`decimal_places` must be >= 0, got {cfg.decimal_places}.")
    if cfg.duration_s is not None and cfg.duration_s <= 0:
        raise ValueError(f"`duration_s` must be > 0 when provided, got {cfg.duration_s}.")

    teleop = make_teleoperator_from_config(cfg.teleop)
    teleop.connect(calibrate=cfg.calibrate)

    print("Reading leader-arm joints. Press Ctrl+C to stop.")

    lines_printed = 0
    joint_keys: list[str] | None = None
    display_len = 0
    start = time.perf_counter()

    try:
        while True:
            loop_start = time.perf_counter()
            action = teleop.get_action()
            joint_angles = _extract_joint_angles(action)
            if not joint_angles:
                raise RuntimeError("Teleoperator returned no numeric joint values.")

            if joint_keys is None:
                joint_keys = sorted(joint_angles)
                display_len = max(len(key) for key in joint_keys)

            if lines_printed:
                move_cursor_up(lines_printed)

            elapsed = time.perf_counter() - start
            print(f"Elapsed: {elapsed:.2f}s")
            print(f"{'JOINT':<{display_len}} | {'ANGLE':>10}")
            for key in joint_keys:
                val = joint_angles.get(key, float("nan"))
                print(f"{key:<{display_len}} | {val:>10.{cfg.decimal_places}f}")
            lines_printed = len(joint_keys) + 2

            if cfg.duration_s is not None and elapsed >= cfg.duration_s:
                break

            dt_s = time.perf_counter() - loop_start
            precise_sleep(max(1.0 / cfg.fps - dt_s, 0.0))
    finally:
        teleop.disconnect()


def main():
    register_third_party_plugins()
    print_leader_joints()


if __name__ == "__main__":
    main()
