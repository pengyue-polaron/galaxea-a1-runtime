from pathlib import Path
from types import SimpleNamespace

import hashlib
import pytest

from galaxea_a1_runtime.models.config import ModelFile
from galaxea_a1_runtime.models.registry import resolve_registered_model
from galaxea_a1_runtime.models.store import fetch_artifact
from scripts.models.model_store import configured_registry_paths


REPO = Path(__file__).resolve().parents[1]


def test_registered_model_selector_accepts_descriptor_id_and_pinned_id() -> None:
    by_name = resolve_registered_model(
        "mango_placement_eef", repo_root=REPO, backend="lingbot_va"
    )
    by_id = resolve_registered_model(
        "lingbot/a1_mango_placement_eef", repo_root=REPO, backend="lingbot_va"
    )
    by_pin = resolve_registered_model(
        "lingbot/a1_mango_placement_eef@step-200",
        repo_root=REPO,
        backend="lingbot_va",
    )

    assert by_name == by_id == by_pin
    assert by_name.checkpoint_step == 200


def test_registered_model_selector_rejects_unknown_models() -> None:
    with pytest.raises(ValueError, match="unknown registered lingbot_va model"):
        resolve_registered_model("not-registered", repo_root=REPO, backend="lingbot_va")


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
    assert (
        paths[
            "model:lingbot/a1_mango_plate_eef@0fb7f5a46dbdeac0770ae46d6a71411171348eb6"
        ]
        == (
            REPO
            / "models/artifacts/lingbot/a1_mango_plate_eef"
            / "0fb7f5a46dbdeac0770ae46d6a71411171348eb6"
        ).absolute()
    )
    assert (
        paths[
            "model:lingbot/a1_mango_placement_eef@"
            "bf15da70c432e39c3a971c50143f4d91ff671ac1"
        ]
        == (
            REPO
            / "models/artifacts/lingbot/a1_mango_placement_eef"
            / "bf15da70c432e39c3a971c50143f4d91ff671ac1"
        ).absolute()
    )
    assert len(paths) == 4


def test_model_fetch_reuses_identical_local_files(tmp_path, monkeypatch) -> None:
    shared = b"shared immutable model bytes"
    unique = b"new checkpoint bytes"
    source = tmp_path / "models/artifacts/lingbot/existing" / ("a" * 40) / "shared.bin"
    source.parent.mkdir(parents=True)
    source.write_bytes(shared)
    target = tmp_path / "models/artifacts/lingbot/new" / ("b" * 40)
    files = (
        ModelFile(
            path=Path("shared.bin"),
            size=len(shared),
            sha256=hashlib.sha256(shared).hexdigest(),
        ),
        ModelFile(
            path=Path("unique.bin"),
            size=len(unique),
            sha256=hashlib.sha256(unique).hexdigest(),
        ),
    )
    config = SimpleNamespace(
        artifact_root=target,
        repo_root=tmp_path,
        source=SimpleNamespace(
            provider="huggingface",
            repo_id="owner/model",
            revision="b" * 40,
        ),
        manifest=SimpleNamespace(files=files, sha256="manifest-digest"),
    )

    def fake_snapshot_download(**kwargs) -> None:
        assert kwargs["allow_patterns"] == ["unique.bin"]
        (Path(kwargs["local_dir"]) / "unique.bin").write_bytes(unique)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    result = fetch_artifact(config)

    assert result.files == 2
    assert (target / "shared.bin").read_bytes() == shared
    assert (target / "unique.bin").read_bytes() == unique
    assert (target / "shared.bin").stat().st_ino == source.stat().st_ino
