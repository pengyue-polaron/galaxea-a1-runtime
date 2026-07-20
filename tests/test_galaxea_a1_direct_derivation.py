from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import galaxea_a1_runtime.lerobot.derive as derive_module
from galaxea_a1_runtime.lerobot.derivation_config import (
    DerivativeOutput,
    load_derivation_config,
)
from galaxea_a1_runtime.lerobot.direct_recording import (
    DirectDatasetIdentity,
    DirectLeRobotEpisode,
    discover_direct_dataset,
)
from galaxea_a1_runtime.lerobot.eef_pack import pack_eef_v3_dataset
from galaxea_a1_runtime.lerobot.v21 import export_v21_dataset
from galaxea_a1_runtime.schema import (
    CameraSpec,
    DIRECT_DATASET_SCHEMA_VERSION,
    EEF_ACTION_NAMES,
    canonical_dataset_contract,
)


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "tests/fixtures/direct_derivation.toml"


def test_direct_derivation_config_has_one_canonical_source():
    config = load_derivation_config(CONFIG)

    assert config.source_root == REPO / "data/datasets/direct-test"
    assert config.joint_v21.repo_id == "galaxea-a1/direct-test-joint-v21"
    assert config.eef_v3.target_root == REPO / "data/processed/direct_test_eef_v3"
    assert config.system.path == REPO / "configs/system/a1.toml"


def test_direct_derivation_rejects_duplicate_final_targets(tmp_path: Path):
    path = tmp_path / "derive.toml"
    path.write_text(
        CONFIG.read_text().replace(
            'target_root = "data/processed/direct_test_eef_v3"',
            'target_root = "data/processed/direct_test_joint_v21"',
        )
    )

    with pytest.raises(ValueError, match="target roots must be unique"):
        load_derivation_config(path, repo_root=REPO)


def test_direct_derivation_rejects_nested_final_targets(tmp_path: Path):
    path = tmp_path / "derive.toml"
    path.write_text(
        CONFIG.read_text().replace(
            'target_root = "data/processed/direct_test_eef_v3"',
            'target_root = "data/processed/direct_test_joint_v21/eef"',
        )
    )

    with pytest.raises(ValueError, match="target roots must not overlap"):
        load_derivation_config(path, repo_root=REPO)


def test_each_derivative_starts_from_the_canonical_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config = load_derivation_config(CONFIG)
    output = lambda name: DerivativeOutput(  # noqa: E731 - compact test fixture.
        target_root=tmp_path / name,
        archive_path=tmp_path / f"{name}.tar.gz",
        repo_id=f"test/{name}",
    )
    config = replace(
        config,
        source_root=tmp_path / "canonical",
        joint_v21=output("joint-v21"),
        eef_v3=output("eef-v3"),
        eef_v21=output("eef-v21"),
    )
    calls: list[tuple[str, Path]] = []
    source = SimpleNamespace(
        identity=SimpleNamespace(
            target_root=config.source_root,
            repo_id="test/canonical",
        )
    )
    monkeypatch.setattr(
        derive_module, "discover_direct_dataset", lambda *_args, **_kwargs: source
    )

    def pack_eef(**kwargs):
        calls.append(("eef", kwargs["source_root"]))
        return {"format": "eef-v3"}

    monkeypatch.setattr(derive_module, "pack_eef_v3_dataset", pack_eef)

    def export_v21(**kwargs):
        calls.append(("joint-v2.1", kwargs["source_root"]))
        return {"source_root": str(kwargs["source_root"])}

    monkeypatch.setattr(derive_module, "export_v21_dataset", export_v21)

    result = derive_module.build_derivatives(config)

    assert set(result) == {
        derive_module.JOINT_V21,
        derive_module.EEF_V3,
        derive_module.EEF_V21,
    }
    assert calls[0] == ("joint-v2.1", config.source_root)
    assert calls[1] == ("eef", config.source_root)
    assert calls[2] == ("eef", config.source_root)
    assert calls[3][0] == "joint-v2.1"
    assert calls[3][1].name == "eef-v3"
    assert calls[3][1] != config.eef_v3.target_root


def test_v21_exports_and_eef_v3_accept_the_direct_canonical_contract(tmp_path: Path):
    source_root = tmp_path / "canonical"
    identity = DirectDatasetIdentity(
        target_root=source_root,
        repo_id="test/canonical",
        fps=30,
        contract=canonical_dataset_contract(
            cameras=(
                CameraSpec("front", height=64, width=64),
                CameraSpec("wrist", height=64, width=64),
            )
        ),
        experiment="canonical",
    )
    state = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, *([0.0] * 6), 0.5],
        dtype=np.float32,
    )
    action = np.array([*([0.0] * 6), 0.5], dtype=np.float32)
    with DirectLeRobotEpisode(
        identity=identity,
        task="test direct derivation",
        provenance={"quality_checks": {}},
    ) as episode:
        episode.add_frame(
            {
                "observation.state": state,
                "action": action,
                "observation.images.front": np.zeros((64, 64, 3), dtype=np.uint8),
                "observation.images.wrist": np.zeros((64, 64, 3), dtype=np.uint8),
                "task": "test direct derivation",
            }
        )
        episode.commit()

    discovered = discover_direct_dataset(source_root, contract=identity.contract)
    assert discovered.identity.repo_id == identity.repo_id
    assert discovered.state.task == "test direct derivation"

    joint = export_v21_dataset(
        source_root=source_root,
        target_root=tmp_path / "joint",
        repo_id="test/joint",
        source_dataset=identity.repo_id,
    )
    eef = pack_eef_v3_dataset(
        source_root=source_root,
        target_root=tmp_path / "eef",
        urdf_path=(
            REPO
            / "third_party/A1_SDK/install/share/mobiman/urdf/A1/urdf/A1_URDF_0607_0028.urdf"
        ),
        repo_id="test/eef",
        gripper_stroke_min_mm=0.0,
        gripper_stroke_max_mm=104.0,
        base_link="base_link",
        tip_link="arm_seg6",
        source_dataset=identity.repo_id,
    )
    eef_v21 = export_v21_dataset(
        source_root=tmp_path / "eef",
        target_root=tmp_path / "eef-v21",
        repo_id="test/eef-v21",
        source_dataset=identity.repo_id,
    )

    assert joint["format"] == "v2.1"
    assert eef["source_format"] == DIRECT_DATASET_SCHEMA_VERSION
    assert eef_v21["format"] == "v2.1"
    for derivative in (tmp_path / "joint", tmp_path / "eef", tmp_path / "eef-v21"):
        assert not (derivative / "meta/galaxea_a1.json").exists()
        assert (derivative / "meta/source_galaxea_a1.json").is_file()
    eef_info = json.loads((tmp_path / "eef/meta/info.json").read_text())
    assert eef_info["features"]["action"]["names"] == list(EEF_ACTION_NAMES)
    eef_v21_info = json.loads((tmp_path / "eef-v21/meta/info.json").read_text())
    assert eef_v21_info["codebase_version"] == "v2.1"
    assert eef_v21_info["features"]["action"]["names"] == list(EEF_ACTION_NAMES)
