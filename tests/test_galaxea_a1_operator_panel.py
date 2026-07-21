import json
import shutil
import sys
from pathlib import Path

import pytest

from embodied_ops.operator_panel import OperatorPanelApplication, WorkflowLaunch

from galaxea_a1_runtime.apps.operator_panel import A1OperatorPanelAdapter


ROOT = Path(__file__).resolve().parents[1]


def test_a1_panel_adapter_discovers_and_builds_validated_workflows():
    adapter = A1OperatorPanelAdapter(ROOT)
    catalog = adapter.catalog()
    batch_group = next(
        group for group in catalog["configuration_groups"] if group["label"] == "Batch"
    )

    batch_paths = {item["value"] for item in batch_group["items"]}
    assert "configs/runs/lingbot/fruit_placement.toml" in batch_paths
    assert "configs/runs/lingbot/mango_placement.toml" in batch_paths
    assert len(catalog["cameras"]) == 2
    assert adapter.panel_bind == "0.0.0.0"
    assert adapter.panel_port == 8765
    assert catalog["cameras"][0]["port"] == 8088
    assert catalog["cameras"][0]["path"] == "/agent.mjpg"
    reset_form = next(item for item in catalog["workflows"] if item["id"] == "reset")
    assert reset_form["tone"] == "danger"
    prompt_form = next(
        item for item in catalog["registrations"] if item["id"] == "prompt"
    )
    distribution = next(
        field for field in prompt_form["fields"] if field["name"] == "distribution"
    )
    assert distribution["default"] == "ood"
    task_id = next(
        field for field in prompt_form["fields"] if field["name"] == "task_id"
    )
    assert (task_id["derive_from"], task_id["transform"]) == (
        "prompt",
        "snake_case",
    )
    evaluation = next(item for item in catalog["workflows"] if item["id"] == "evaluate")
    models = next(field for field in evaluation["fields"] if field["name"] == "model")
    assert any("step-1000" in option["label"] for option in models["options"])

    launch = adapter.build_launch(
        "batch",
        {
            "config": "configs/runs/lingbot/mango_placement.toml",
            "scene_note": "test scene",
            "resume": True,
        },
    )
    assert launch.name == "batch:mango-placement-scripted"
    assert "--resume" in launch.command
    assert launch.command[-1] == str(ROOT / "configs/runs/lingbot/mango_placement.toml")
    assert [action.tone for action in launch.input_actions] == [
        "primary",
        "danger",
        "quiet",
    ]

    reset = adapter.build_launch(
        "reset",
        {"pose": "configs/poses/a1_collection_start.toml"},
    )
    assert reset.command == (
        str(ROOT / "scripts/apps/reset/a1_reset_runtime.sh"),
        "--system-config",
        str(ROOT / "configs/system/a1.toml"),
        "--pose",
        str(ROOT / "configs/poses/a1_collection_start.toml"),
    )

    with pytest.raises(ValueError, match="repository TOML"):
        adapter.build_launch(
            "collect",
            {
                "config": "configs/system/a1.toml",
                "experiment": "run_01",
                "task": "pick fruit",
            },
        )


def test_a1_panel_registers_a_prompt_and_selects_it_for_evaluation(tmp_path):
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    (tmp_path / "third_party").symlink_to(
        ROOT / "third_party", target_is_directory=True
    )
    (tmp_path / "external").symlink_to(ROOT / "external", target_is_directory=True)
    adapter = A1OperatorPanelAdapter(tmp_path)
    result = adapter.register(
        "prompt",
        {
            "catalog": "configs/tasks/fruit_placement/catalog.json",
            "task_id": "green_apple_bowl",
            "prompt": "put the green apple into the bowl",
            "distribution": "ood",
        },
    )

    assert result["created"] == (
        "configs/tasks/fruit_placement/prompts/green_apple_bowl.json"
    )
    assert result["activate"] == {
        "panel": "evaluate",
        "values": {
            "config": "configs/deployments/lingbot/fruit_placement_eef.toml",
            "task": "green_apple_bowl",
        },
    }
    task_options = next(
        field["options"]
        for workflow in result["catalog"]["workflows"]
        if workflow["id"] == "evaluate"
        for field in workflow["fields"]
        if field["name"] == "task"
    )
    assert any(option["value"] == "green_apple_bowl" for option in task_options)


def test_operator_panel_blocks_registration_while_a_workflow_is_active():
    app = OperatorPanelApplication(A1OperatorPanelAdapter(ROOT))
    app.workflow.start(
        WorkflowLaunch(
            workflow="test",
            name="test",
            command=(sys.executable, "-c", "import time; time.sleep(30)"),
        )
    )
    try:
        with pytest.raises(RuntimeError, match="while a workflow is active"):
            app.register(
                {
                    "registration": "prompt",
                    "values": {
                        "catalog": "configs/tasks/fruit_placement/catalog.json",
                        "task_id": "must_not_be_created",
                        "prompt": "must not be created",
                        "distribution": "ood",
                    },
                }
            )
    finally:
        app.workflow.stop()

    assert not (
        ROOT / "configs/tasks/fruit_placement/prompts/must_not_be_created.json"
    ).exists()


def test_a1_panel_camera_health_normalizes_the_read_only_preview(monkeypatch):
    payload = {
        "ok": True,
        "streams": {
            "agent": {
                "ready": True,
                "fresh": True,
                "preview_fps": 9.87,
                "age_s": 0.031,
                "error": None,
            }
        },
    }

    class Response:
        status = 200

        def read(self, _limit):
            return json.dumps(payload).encode()

    class Connection:
        def __init__(self, host, port, *, timeout):
            assert host == "127.0.0.1"
            assert port == 8088
            assert timeout > 0

        def request(self, method, path, *, headers):
            assert (method, path) == ("GET", "/healthz")
            assert headers == {"Cache-Control": "no-store"}

        def getresponse(self):
            return Response()

        def close(self):
            return None

    monkeypatch.setattr(
        "galaxea_a1_runtime.apps.operator_panel.adapter.HTTPConnection", Connection
    )
    health = A1OperatorPanelAdapter(ROOT).camera_health()

    assert health == {
        "available": True,
        "ok": True,
        "streams": {
            "agent": {
                "ready": True,
                "fresh": True,
                "preview_fps": 9.87,
                "age_s": 0.031,
                "error": None,
            }
        },
    }


def test_a1_panel_camera_health_reports_an_offline_monitor(monkeypatch):
    class Connection:
        def __init__(self, _host, _port, *, timeout):
            assert timeout > 0

        def request(self, _method, _path, *, headers):
            assert headers == {"Cache-Control": "no-store"}
            raise ConnectionRefusedError

        def close(self):
            return None

    monkeypatch.setattr(
        "galaxea_a1_runtime.apps.operator_panel.adapter.HTTPConnection", Connection
    )

    assert A1OperatorPanelAdapter(ROOT).camera_health() == {
        "available": False,
        "ok": False,
        "streams": {},
        "reason": "Camera monitor is not running.",
    }
