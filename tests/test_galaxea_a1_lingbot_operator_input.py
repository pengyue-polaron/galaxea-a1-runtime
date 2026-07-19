import io

import pytest

from galaxea_a1_runtime.apps.lingbot.operator_input import (
    SceneNoteCancelled,
    prompt_scene_note,
    validate_scene_note,
)


def test_scene_note_is_required_trimmed_and_unicode_safe():
    assert validate_scene_note("  桌面偏左，光照正常  ") == "桌面偏左，光照正常"
    with pytest.raises(ValueError, match="must not be empty"):
        validate_scene_note("  ")
    with pytest.raises(ValueError, match="single line"):
        validate_scene_note("one\ntwo")


def test_scene_note_prompt_retries_invalid_input_and_can_cancel():
    answers = iter(("", "blue plate shifted left"))
    output = io.StringIO()

    assert (
        prompt_scene_note(input_fn=lambda: next(answers), output=output)
        == "blue plate shifted left"
    )
    assert "must not be empty" in output.getvalue()
    with pytest.raises(SceneNoteCancelled, match="cancelled"):
        prompt_scene_note(input_fn=lambda: "q", output=io.StringIO())
