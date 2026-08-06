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
    assert catalog["product"] == {"brand": "GALAXEA A1", "title": "Operator Panel"}
    assert [workflow["id"] for workflow in catalog["workflows"]] == [
        "hardware",
        "collect",
        "reset",
        "dataset-doctor",
        "export-v21",
        "evaluate",
        "batch",
    ]
    assert {
        (item["extension"], item["language"]) for item in catalog["configuration_types"]
    } == {(".toml", "TOML")}
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
    cameras = adapter.build_launch("camera", {"action": "start"})
    assert cameras.command[-1] == "start"
    hardware = adapter.build_launch(
        "hardware", {"config": "configs/teleop/a1_so100.toml"}
    )
    assert hardware.command == (
        sys.executable,
        "-m",
        "galaxea_a1_runtime.cli",
        "hardware",
        "--repo-root",
        str(ROOT),
        "--config",
        str(ROOT / "configs/teleop/a1_so100.toml"),
    )
    dataset_doctor = adapter.build_launch(
        "dataset-doctor",
        {
            "config": "configs/teleop/a1_so100.toml",
            "experiment": "plug_insertion_v1",
        },
    )
    assert dataset_doctor.command[:3] == (
        sys.executable,
        "-m",
        "galaxea_a1_runtime.cli",
    )
    assert dataset_doctor.command[-3:] == (
        "--config",
        str(ROOT / "configs/teleop/a1_so100.toml"),
        "plug_insertion_v1",
    )
    export = adapter.build_launch(
        "export-v21",
        {
            "config": "configs/teleop/a1_so100.toml",
            "experiment": "plug_insertion_v1",
        },
    )
    assert export.command[-3:] == (
        "--repo-root",
        str(ROOT),
        "plug_insertion_v1",
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
    result = OperatorPanelApplication(adapter).register(
        {
            "registration": "prompt",
            "values": {
                "catalog": "configs/tasks/fruit_placement/catalog.json",
                "task_id": "green_apple_bowl",
                "prompt": "put the green apple into the bowl",
                "distribution": "ood",
            },
        }
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


def test_a1_panel_suggests_existing_experiments_but_allows_a_new_name(tmp_path):
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    (tmp_path / "third_party").symlink_to(
        ROOT / "third_party", target_is_directory=True
    )
    (tmp_path / "external").symlink_to(ROOT / "external", target_is_directory=True)
    dataset_root = tmp_path / "data/datasets"
    (dataset_root / "fruit_placement_v1").mkdir(parents=True)
    (dataset_root / "plug_insertion_v1").mkdir()
    (dataset_root / ".incomplete-staging").mkdir()
    (dataset_root / "invalid name").mkdir()

    catalog = A1OperatorPanelAdapter(tmp_path).catalog()
    collect = next(item for item in catalog["workflows"] if item["id"] == "collect")
    experiment = next(
        field for field in collect["fields"] if field["name"] == "experiment"
    )
    task = next(field for field in collect["fields"] if field["name"] == "task")

    assert experiment["type"] == "combobox"
    assert experiment["depends_on"] == "config"
    assert "append episodes" in experiment["help_text"]
    assert [option["value"] for option in experiment["options"]] == [
        "fruit_placement_v1",
        "plug_insertion_v1",
    ]
    assert all(
        option["depends_value"] == "configs/teleop/a1_so100.toml"
        for option in experiment["options"]
    )
    assert task["type"] == "combobox"
    assert "multiple prompts" in task["help_text"]
    assert {
        option["value"]
        for option in task["options"]
        if option["label"].startswith("plug-insertion-v1")
    } == {
        "pick up the charger and insert it into the first socket from the left on the power strip",
        "pick up the charger and insert it into the second socket from the left on the power strip",
        "pick up the charger and insert it into the third socket from the left on the power strip",
        "pick up the charger and insert it into the fourth socket from the left on the power strip",
    }


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


def test_a1_panel_uses_shared_camera_health_provider(monkeypatch):
    health = {
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

    monkeypatch.setattr(
        "galaxea_a1_runtime.apps.operator_panel.adapter.fetch_camera_health",
        lambda port: health if port == 8088 else None,
    )
    assert A1OperatorPanelAdapter(ROOT).camera_health() == health
