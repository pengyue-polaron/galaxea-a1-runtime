from pathlib import Path

import numpy as np

from galaxea_a1_runtime.apps.lingbot.config import load_lingbot_config
from galaxea_a1_runtime.apps.lingbot.smoke import synthetic_observation


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/deployments/lingbot/fruit_placement_eef.toml"


def test_synthetic_observation_uses_tracked_camera_contract():
    config = load_lingbot_config(CONFIG, repo_root=REPO)

    observation = synthetic_observation(config)

    assert set(observation) == {
        "observation.images.front",
        "observation.images.wrist",
    }
    assert observation["observation.images.front"].shape == (480, 480, 3)
    assert observation["observation.images.wrist"].shape == (480, 640, 3)
    assert all(value.dtype == np.uint8 for value in observation.values())
