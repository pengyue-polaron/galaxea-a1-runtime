import pytest

from galaxea_a1_runtime.foxglove_layout_publish import (
    canonical_layout_id,
    layout_payload,
    select_layout_id,
)


def test_foxglove_layout_publish_selects_one_exact_organization_layout() -> None:
    layouts = [
        {"id": "other", "name": "Other Layout"},
        {"id": "a1-layout", "name": "Galaxea A1 Operations"},
    ]

    assert select_layout_id(layouts, name="Galaxea A1 Operations") == "a1-layout"
    assert len(canonical_layout_id(name="Galaxea A1 Operations")) == 36
    assert layout_payload(
        {"layout": "Tabs!"},
        name="Galaxea A1 Operations",
        folder="Galaxea A1",
        permission="ORG_WRITE",
    ) == {
        "name": "Galaxea A1 Operations",
        "folderName": "Galaxea A1",
        "permission": "ORG_WRITE",
        "data": {"layout": "Tabs!"},
    }


def test_foxglove_layout_publish_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="ambiguous"):
        select_layout_id(
            [
                {"id": "first", "name": "Galaxea A1 Operations"},
                {"id": "second", "name": "Galaxea A1 Operations"},
            ],
            name="Galaxea A1 Operations",
        )

    with pytest.raises(ValueError, match="ORG_WRITE"):
        layout_payload(
            {},
            name="Galaxea A1 Operations",
            folder="Galaxea A1",
            permission="ORG_READ",
        )
