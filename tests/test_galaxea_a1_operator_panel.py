import json
import sys
import time
import tomllib
from pathlib import Path

import pytest

from operator_panel.config_store import ConfigKind, RepositoryConfigStore
from operator_panel.contracts import InputAction, WorkflowLaunch
from operator_panel.process import WorkflowProcess

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
    reset_form = next(item for item in catalog["workflows"] if item["id"] == "reset")
    assert reset_form["tone"] == "danger"

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

    with pytest.raises(ValueError, match="repository TOML"):
        adapter.build_launch(
            "collect",
            {
                "config": "configs/system/a1.toml",
                "experiment": "run_01",
                "task": "pick fruit",
            },
        )


def test_operator_process_accepts_one_announced_input(tmp_path: Path):
    process = WorkflowProcess(tmp_path)
    launch = WorkflowLaunch(
        workflow="test",
        name="test",
        command=(
            sys.executable,
            "-u",
            "-c",
            'print(\'@@OPERATOR_PANEL {"input":["enter"]}\'); '
            "input(); print('workflow complete')",
        ),
        input_actions=(InputAction("enter", "Next", "\n", "primary"),),
    )

    process.start(launch)
    with pytest.raises(RuntimeError, match="already active"):
        process.start(launch)
    status = _wait_for(process, lambda value: bool(value["input_actions"]))
    assert status["input_actions"] == [
        {"id": "enter", "label": "Next", "tone": "primary"}
    ]
    process.send("enter")
    with pytest.raises(RuntimeError, match="not waiting"):
        process.send("enter")

    status = _wait_for(
        process,
        lambda value: not value["active"] and "workflow complete" in value["logs"],
    )
    assert status["exit_code"] == 0


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


def test_repository_config_store_validates_and_creates_without_overwrite(
    tmp_path: Path,
):
    directory = tmp_path / "configs/demo"
    directory.mkdir(parents=True)
    template = directory / "base.toml"
    template.write_text("value = 1\n")

    def validate(path: Path) -> None:
        data = tomllib.loads(path.read_text())
        if set(data) != {"value"} or not isinstance(data["value"], int):
            raise ValueError("demo config requires one integer value")

    store = RepositoryConfigStore(
        tmp_path,
        (ConfigKind("demo", "Demo", Path("configs/demo"), validate),),
    )
    assert store.template("demo", "configs/demo/base.toml")["content"] == "value = 1\n"
    assert store.validate("demo", "second", "value = 2")["valid"] is True
    assert store.create("demo", "second", "value = 2") == {
        "created": "configs/demo/second.toml"
    }
    assert (directory / "second.toml").read_text() == "value = 2\n"
    with pytest.raises(FileExistsError, match="already exists"):
        store.create("demo", "second", "value = 3")


def _wait_for(process: WorkflowProcess, predicate) -> dict:
    deadline = time.monotonic() + 3.0
    status = process.snapshot()
    while not predicate(status) and time.monotonic() < deadline:
        time.sleep(0.01)
        status = process.snapshot()
    return status
