"""Launch the upstream ROS 2 Foxglove bridge with exact System-owned grants."""

import os
from pathlib import Path
import tempfile
from launch import LaunchDescription
from launch_ros.actions import Node
from galaxea_a1_runtime.configuration.system import load_system_config
from galaxea_a1_runtime.observability import (
    NO_MATCH_ALLOWLIST,
    foxglove_asset_uri_allowlist,
    foxglove_asset_uris,
    foxglove_capabilities,
    foxglove_service_whitelist,
    foxglove_topic_whitelist,
)


def generate_launch_description():
    system = load_system_config(Path(os.environ["A1_SYSTEM_CONFIG_PATH"]))
    # Vendor meshes remain read-only in the ROS 1 SDK. Register only asset
    # packages in an ephemeral ament index, without sourcing vendor ROS 1 code.
    prefix = Path(tempfile.mkdtemp(prefix="a1-ros-assets-"))
    index = prefix / "share/ament_index/resource_index/packages"
    index.mkdir(parents=True)
    vendor_share = Path("/workspace/third_party/A1_SDK/install/share")
    for package in {
        uri.removeprefix("package://").split("/", 1)[0]
        for uri in foxglove_asset_uris(system)
    }:
        source = vendor_share / package
        if not source.is_dir():
            raise RuntimeError(f"URDF asset package is unavailable: {package}")
        (index / package).touch()
        (prefix / "share" / package).symlink_to(source, target_is_directory=True)
    observation = system.observability
    return LaunchDescription(
        [
            Node(
                package="foxglove_bridge",
                executable="foxglove_bridge",
                name="a1_foxglove_bridge",
                output="screen",
                additional_env={
                    "AMENT_PREFIX_PATH": f"{prefix}:{os.environ.get('AMENT_PREFIX_PATH', '')}"
                },
                parameters=[
                    {
                        "address": observation.bind,
                        "port": observation.port,
                        "tls": False,
                        "sysinfo": False,
                        "topic_whitelist": list(foxglove_topic_whitelist(system)),
                        "service_whitelist": list(foxglove_service_whitelist(system)),
                        "param_whitelist": list(NO_MATCH_ALLOWLIST),
                        "client_topic_whitelist": list(NO_MATCH_ALLOWLIST),
                        "capabilities": list(foxglove_capabilities(system)),
                        "asset_uri_allowlist": list(
                            foxglove_asset_uri_allowlist(system)
                        ),
                    }
                ],
            )
        ]
    )
