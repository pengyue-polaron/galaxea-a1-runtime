from pathlib import Path

import pytest

from galaxea_a1_runtime.lerobot.dataset import (
    DatasetConfig,
    build_dataset_create_kwargs,
)
from galaxea_a1_runtime.schema import CameraSpec, canonical_dataset_contract


def test_build_dataset_create_kwargs_for_lerobot_v3():
    contract = canonical_dataset_contract(
        cameras=(CameraSpec("front", height=480, width=640),)
    )
    config = DatasetConfig(repo_id="galaxea/a1_test", root=Path("/tmp/a1"), fps=20)

    kwargs = build_dataset_create_kwargs(config=config, contract=contract)

    assert kwargs["repo_id"] == "galaxea/a1_test"
    assert kwargs["fps"] == 20
    assert kwargs["robot_type"] == "galaxea_a1"
    assert kwargs["features"]["observation.images.front"]["dtype"] == "video"
    assert kwargs["use_videos"] is True
    assert "vcodec" not in kwargs


def test_dataset_config_requires_namespaced_repo_id():
    contract = canonical_dataset_contract(
        cameras=(CameraSpec("front", height=480, width=640),)
    )
    config = DatasetConfig(repo_id="a1_test", root=Path("/tmp/a1"), fps=20)

    with pytest.raises(ValueError, match="repo_id"):
        build_dataset_create_kwargs(config=config, contract=contract)
