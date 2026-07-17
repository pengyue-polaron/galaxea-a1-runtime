from pathlib import Path

from scripts.models.model_store import configured_registry_paths


REPO = Path(__file__).resolve().parents[1]


def test_model_registry_keeps_configured_paths_lexical() -> None:
    paths = configured_registry_paths(REPO)

    assert (
        paths[
            "model:lingbot/a1_fruit_placement_eef@"
            "90e017bdbc6afac2e441b4634c9192776bbcb8b7"
        ]
        == (
            REPO
            / "models/artifacts/lingbot/a1_fruit_placement_eef"
            / "90e017bdbc6afac2e441b4634c9192776bbcb8b7"
        ).absolute()
    )
    assert (
        paths[
            "model:openpi_pi05/a1_fruit_placement_eef@"
            "e1a3e53832ce99edc188fb01e5ec303ac305d552"
        ]
        == (
            REPO
            / "models/artifacts/openpi_pi05/a1_fruit_placement_eef"
            / "e1a3e53832ce99edc188fb01e5ec303ac305d552"
        ).absolute()
    )
    assert len(paths) == 2
