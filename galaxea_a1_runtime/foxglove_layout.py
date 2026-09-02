"""Compose the tracked A1 layout around shared Foxglove presentation."""

from __future__ import annotations

import json
from typing import Any

from embodied_ops.foxglove import (
    COLLECTION_CONSOLE_PANEL_TYPE,
    collection_console_panel_config,
)

from galaxea_a1_runtime.configuration.system import SystemConfig
from galaxea_a1_runtime.observability import foxglove_asset_uris


_JOINT_COLORS = ("#4e98e2", "#f5774d", "#f7df71", "#7ac36a", "#be6bff", "#5ac8c8")


def build_foxglove_layout(system: SystemConfig) -> dict[str, Any]:
    """Build the canonical A1 observation and collection-control layout."""

    topics = system.observability.topics
    primary = system.topics
    urdf_uri = next(uri for uri in foxglove_asset_uris(system) if uri.endswith(".urdf"))
    measured_paths = [
        _plot_path(
            f"{primary.joint_states}.position[{index}]",
            label=f"{joint} measured",
            color=_JOINT_COLORS[index],
        )
        for index, joint in enumerate(system.joint_safety.names)
    ]
    staged_paths = [
        _plot_path(
            f"{topics.staged_joint_state}.position[{index}]",
            label=f"{joint} staged",
            color=_JOINT_COLORS[index],
            enabled=False,
        )
        for index, joint in enumerate(system.joint_safety.names)
    ]
    forwarded_paths = [
        _plot_path(
            f"{topics.host_joint_state}.position[{index}]",
            label=f"{joint} forwarded",
            color=_JOINT_COLORS[index],
            enabled=False,
        )
        for index, joint in enumerate(system.joint_safety.names)
    ]
    gripper_paths = (
        (topics.gripper_feedback_state, "measured", "#4e98e2", True),
        (topics.gripper_target_state, "target", "#f7df71", True),
        (topics.gripper_command_state, "forwarded", "#f5774d", True),
    )
    collection_console = collection_console_panel_config(
        status_topic=topics.workflow_status,
        services=system.operator_panel.services.__dict__,
    )
    return {
        "configById": {
            "Image!front": {"imageMode": {"imageTopic": topics.front_image}},
            "Image!wrist": {"imageMode": {"imageTopic": topics.wrist_image}},
            "3D!robot": {
                "cameraState": {
                    "perspective": True,
                    "distance": 1.5,
                    "phi": 60,
                    "thetaOffset": 45,
                    "targetOffset": [0, 0, 0.35],
                    "target": [0, 0, 0],
                    "targetOrientation": [0, 0, 0, 1],
                    "fovy": 45,
                    "near": 0.01,
                    "far": 100,
                },
                "followMode": "follow-pose",
                "followTf": "base_link",
                "scene": {
                    "enableStats": False,
                    "transforms": {"editable": False, "showLabel": False},
                    "meshUpAxis": "z_up",
                },
                "transforms": {},
                "topics": {},
                "layers": {
                    "grid": {
                        "visible": True,
                        "frameLocked": True,
                        "label": "Grid",
                        "instanceId": "a1-grid",
                        "layerId": "foxglove.Grid",
                        "size": 2,
                        "divisions": 20,
                        "lineWidth": 1,
                        "color": "#248eff",
                        "position": [0, 0, 0],
                        "rotation": [0, 0, 0],
                        "frameId": "base_link",
                    },
                    "a1-urdf": {
                        "displayMode": "auto",
                        "fallbackColor": "#ffffff",
                        "showAxis": False,
                        "showOutlines": True,
                        "opacity": 1,
                        "visible": True,
                        "frameLocked": True,
                        "instanceId": "a1-urdf",
                        "label": "A1 URDF",
                        "layerId": "foxglove.Urdf",
                        "sourceType": "url",
                        "url": urdf_uri,
                        "filePath": "",
                        "parameter": "",
                        "topic": "",
                        "framePrefix": "",
                    },
                },
                "imageMode": {},
            },
            "DiagnosticSummary!a1": {
                "minLevel": 0,
                "pinnedIds": [],
                "hardwareIdFilter": "",
                "topicToRender": topics.diagnostics,
                "sortByLevel": True,
            },
            "DiagnosticStatusPanel!a1": {
                "selectedHardwareId": "galaxea-a1",
                "selectedName": "A1/Relay",
                "topicToRender": topics.diagnostics,
            },
            "Plot!joints": {
                "paths": measured_paths + staged_paths + forwarded_paths,
                **_plot_options(),
            },
            "Plot!gripper": {
                "paths": [
                    _plot_path(
                        f"{topic}.position[0]",
                        label=f"gripper {label} (mm)",
                        color=color,
                        enabled=enabled,
                    )
                    for topic, label, color, enabled in gripper_paths
                ],
                **_plot_options(),
            },
            "RawMessages!diagnostics": {
                "diffEnabled": False,
                "diffMethod": "custom",
                "diffTopicPath": "",
                "showFullMessageForDiff": False,
                "topicPath": topics.diagnostics,
            },
            "RawMessages!workflow": {
                "diffEnabled": False,
                "diffMethod": "custom",
                "diffTopicPath": "",
                "showFullMessageForDiff": False,
                "topicPath": topics.workflow_status,
            },
            f"{COLLECTION_CONSOLE_PANEL_TYPE}!controls": collection_console,
            "RosOut!logs": {"searchTerms": [], "minLogLevel": 1},
            "Tab!overview-details": {
                "activeTabIdx": 0,
                "tabs": [
                    {
                        "title": "Diagnostics",
                        "layout": {
                            "direction": "row",
                            "splitPercentage": 50,
                            "first": "DiagnosticSummary!a1",
                            "second": "DiagnosticStatusPanel!a1",
                        },
                    },
                    {"title": "3D", "layout": "3D!robot"},
                ],
            },
            "Tab!a1": {
                "activeTabIdx": 0,
                "tabs": [
                    {
                        "title": "A1 Overview",
                        "layout": {
                            "direction": "row",
                            "splitPercentage": 38,
                            "first": {
                                "direction": "column",
                                "splitPercentage": 50,
                                "first": "Image!front",
                                "second": "Image!wrist",
                            },
                            "second": {
                                "direction": "column",
                                "splitPercentage": 50,
                                "first": f"{COLLECTION_CONSOLE_PANEL_TYPE}!controls",
                                "second": "Tab!overview-details",
                            },
                        },
                    },
                    {
                        "title": "Signals & Logs",
                        "layout": {
                            "direction": "row",
                            "splitPercentage": 58,
                            "first": {
                                "direction": "column",
                                "splitPercentage": 70,
                                "first": "Plot!joints",
                                "second": "Plot!gripper",
                            },
                            "second": {
                                "direction": "column",
                                "splitPercentage": 34,
                                "first": "RawMessages!diagnostics",
                                "second": {
                                    "direction": "column",
                                    "splitPercentage": 50,
                                    "first": "RawMessages!workflow",
                                    "second": "RosOut!logs",
                                },
                            },
                        },
                    },
                ],
            },
        },
        "globalVariables": {},
        "userNodes": {},
        "linkedGlobalVariables": [],
        "playbackConfig": {"speed": 1},
        "layout": "Tab!a1",
    }


def render_foxglove_layout(system: SystemConfig) -> str:
    return (
        json.dumps(build_foxglove_layout(system), indent=2, ensure_ascii=False) + "\n"
    )


def _plot_path(
    value: str,
    *,
    label: str,
    color: str,
    enabled: bool = True,
) -> dict[str, object]:
    return {
        "timestampMethod": "receiveTime",
        "value": value,
        "enabled": enabled,
        "color": color,
        "label": label,
        "showLine": True,
    }


def _plot_options() -> dict[str, object]:
    return {
        "showXAxisLabels": True,
        "showYAxisLabels": True,
        "showLegend": True,
        "legendDisplay": "floating",
        "showPlotValuesInLegend": False,
        "isSynced": True,
        "xAxisVal": "timestamp",
        "sidebarDimension": 280,
    }
