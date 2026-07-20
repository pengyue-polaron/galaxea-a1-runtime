import pytest
from pathlib import Path

from galaxea_a1_runtime.configuration.system import load_system_config
from galaxea_a1_runtime.constants import LEROBOT_DATASET_FORMAT
from galaxea_a1_runtime.schema import (
    ActionMode,
    CameraSpec,
    camera_specs_from_system,
    canonical_dataset_contract,
    validate_frame_keys,
)


def test_canonical_contract_targets_lerobot_v3():
    repo = Path(__file__).resolve().parents[1]
    system = load_system_config(repo / "configs/system/a1.toml", repo_root=repo)
    contract = canonical_dataset_contract(cameras=camera_specs_from_system(system))

    assert contract.dataset_format == LEROBOT_DATASET_FORMAT
    assert contract.dataset_format == "v3.0"
    assert contract.action_mode == ActionMode.JOINT_ABSOLUTE
    assert contract.features()["observation.images.front"]["shape"] == (480, 480, 3)


def test_canonical_contract_exposes_expected_feature_keys():
    contract = canonical_dataset_contract(
        cameras=(CameraSpec("front", height=480, width=640),)
    )
    features = contract.features()

    assert sorted(features) == [
        "action",
        "observation.images.front",
        "observation.state",
    ]
    assert features["observation.images.front"]["dtype"] == "video"
    assert features["observation.state"]["shape"] == (14,)
    assert features["action"]["shape"] == (7,)


def test_depth_camera_spec_marks_lerobot_depth_feature():
    spec = CameraSpec(
        "front_depth",
        height=480,
        width=640,
        channels=1,
        is_depth_map=True,
        depth_unit="mm",
    )
    feature = spec.feature()

    assert feature["dtype"] == "video"
    assert feature["shape"] == (480, 640, 1)
    assert feature["info"] == {"is_depth_map": True, "depth_unit": "mm"}


def test_validate_frame_keys_reports_missing_required_keys():
    contract = canonical_dataset_contract(
        cameras=(CameraSpec("front", height=480, width=640),)
    )

    with pytest.raises(ValueError, match="observation.images.front"):
        validate_frame_keys(
            {
                "observation.state": [0.0] * 14,
                "action": [0.0] * 7,
            },
            contract=contract,
        )
