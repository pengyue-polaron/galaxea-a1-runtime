"""Build the A1 adapter against the runtime image's installed TRAC-IK library."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from galaxea_a1_runtime.configuration.system import load_system_config
from galaxea_a1_runtime.hardware.trac_ik import trac_ik_binary, verify_trac_ik_build


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/system/a1.toml")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify existing build without starting a container",
    )
    args = parser.parse_args()
    system = load_system_config(args.config, repo_root=root)
    if args.check:
        print(json.dumps(verify_trac_ik_build(system), indent=2))
        return
    binary = trac_ik_binary(system)
    binary.parent.mkdir(parents=True, exist_ok=True)
    source = root / "galaxea_a1_runtime/hardware/native/trac_ik_worker.cpp"
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{.Id}}", system.host.image],
        text=True,
    ).strip()
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            "/tmp",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--mount",
            f"type=bind,source={source},target=/source.cpp,readonly",
            "--mount",
            f"type=bind,source={binary.parent},target=/build",
            "--entrypoint",
            "bash",
            image_id,
            "-lc",
            "source /opt/ros/noetic/setup.bash && "
            "g++ -std=c++17 -O2 -Wall -Wextra /source.cpp -o /build/adapter.tmp "
            "-I/usr/include/eigen3 $(pkg-config --cflags --libs trac_ik_lib) && "
            "dpkg-query -W ros-noetic-trac-ik-lib > /build/version.txt",
        ],
        check=True,
    )
    os.replace(binary.parent / "adapter.tmp", binary)
    receipt = {
        "image_id": image_id,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "trac_ik_version": (binary.parent / "version.txt").read_text(),
        "mode": "Distance",
    }
    binary.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
