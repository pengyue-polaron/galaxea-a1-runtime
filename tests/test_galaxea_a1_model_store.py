from pathlib import Path

from scripts.models.model_store import configured_registry_paths, configured_slots


REPO = Path(__file__).resolve().parents[1]


def test_model_registry_keeps_configured_paths_lexical() -> None:
    paths = configured_registry_paths(REPO)

    assert paths["lingbot-base"] == (REPO / "models/base/lingbot-va-base").absolute()
    assert configured_slots(REPO)["lingbot-base"] == Path("base/lingbot-va-base")
