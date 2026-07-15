from pathlib import Path

import pytest

from galaxea_a1_runtime.collection import (
    CameraMetadata,
    EpisodeDecision,
    StateMode,
    TeleopRawEpisodeMetadata,
    action_columns,
    find_joint_action_step_violation,
    metadata_to_json_dict,
    next_episode_index,
    normalize_episode_decision,
    state_columns,
    state_names_for_mode,
    teleop_frame_header,
    validate_existing_camera_shape,
    validate_episode_layout,
    validate_existing_schema,
    validate_experiment_name,
)
from galaxea_a1_runtime.collection.schema import TELEOP_RAW_SCHEMA_VERSION
from galaxea_a1_runtime.constants import JOINT_TRACKER_NODE_NAME, SAFE_RELAY_SCRIPT
from galaxea_a1_runtime.schema import ActionMode, JOINT_ACTION_NAMES


@pytest.mark.parametrize("name", ["../escape", "nested/name", "", ".", ".."])
def test_experiment_name_rejects_path_traversal_and_empty_names(name):
    with pytest.raises(ValueError):
        validate_experiment_name(name)


def test_experiment_name_accepts_operator_run_identity():
    assert validate_experiment_name("pick_cube.v2-01") == "pick_cube.v2-01"


def test_state_names_for_modes_are_explicit():
    assert state_names_for_mode(StateMode.EEF) == (
        "eef_x",
        "eef_y",
        "eef_z",
        "eef_qx",
        "eef_qy",
        "eef_qz",
        "eef_qw",
        "gripper",
    )
    assert state_names_for_mode(StateMode.JOINT) == (
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
        "joint_6",
        "gripper",
    )
    assert len(state_names_for_mode(StateMode.EEF_JOINT)) == 14


def test_teleop_frame_header_prefixes_state_and_action_columns():
    header = teleop_frame_header(
        state_names=("eef_x", "joint_1"),
        action_names=("joint_1", "gripper"),
        camera_dirs=("cam0", "cam0_depth"),
    )

    assert header == (
        "frame_index",
        "wall_time_ns",
        "ros_stamp_s",
        "cam0_seq",
        "cam0_monotonic_s",
        "cam1_seq",
        "cam1_monotonic_s",
        "cam0_relpath",
        "cam0_depth_relpath",
        "state.eef_x",
        "state.joint_1",
        "action.joint_1",
        "action.gripper",
    )
    assert state_columns(("joint_1",)) == ("state.joint_1",)
    assert action_columns(JOINT_ACTION_NAMES[:1]) == ("action.joint_1",)


def test_episode_decision_matches_old_teleop_interaction():
    assert normalize_episode_decision("") == EpisodeDecision.SAVE
    assert normalize_episode_decision("d") == EpisodeDecision.DISCARD
    assert normalize_episode_decision("discard") == EpisodeDecision.DISCARD
    assert normalize_episode_decision("q") == EpisodeDecision.QUIT


def test_joint_action_quality_check_rejects_discontinuity():
    violation = find_joint_action_step_violation(
        [(0.0, 0.0), (0.1, 0.2), (0.15, 1.0)],
        action_names=("joint_1", "joint_2"),
        max_step_rad=0.35,
    )

    assert violation is not None
    assert violation.frame_index == 2
    assert violation.joint_name == "joint_2"
    assert violation.step_rad == 0.8


def test_joint_action_quality_check_accepts_continuous_actions():
    assert (
        find_joint_action_step_violation(
            [(0.0, 0.0, 0.0), (0.1, -0.1, 1.0), (0.2, -0.2, 0.0)],
            action_names=("joint_1", "joint_2", "gripper"),
            max_step_rad=0.35,
        )
        is None
    )


def test_next_episode_index_uses_existing_episode_prefix(tmp_path):
    (tmp_path / "episode_000_20260708_120000").mkdir()
    (tmp_path / "episode_002_20260708_120100").mkdir()
    (tmp_path / "notes").mkdir()

    assert next_episode_index(Path(tmp_path)) == 3


def test_episode_layout_rejects_crash_leftovers(tmp_path):
    (tmp_path / ".episode_000.staging-deadbeef").mkdir()
    incomplete = tmp_path / "episode_001_20260708_120000"
    incomplete.mkdir()
    (incomplete / "frames.csv").write_text("frame_index\n")

    try:
        validate_episode_layout(tmp_path)
    except ValueError as exc:
        message = str(exc)
        assert "staging:" in message
        assert "incomplete:" in message
    else:
        raise AssertionError("crash leftovers must block collection")


def test_metadata_json_explains_topics_and_control_path():
    metadata = TeleopRawEpisodeMetadata(
        schema_version=TELEOP_RAW_SCHEMA_VERSION,
        collection_mode="teleop",
        task="pick cube",
        experiment="pick_cube",
        episode_index=0,
        frame_count=3,
        fps_target=30.0,
        state_mode=StateMode.EEF_JOINT,
        action_mode=ActionMode.JOINT_ABSOLUTE,
        state_names=("eef_x", "joint_1"),
        action_names=("joint_1",),
        state_topics={"eef": "/end_effector_pose"},
        action_topics={"joint_target": "/arm_joint_target_position"},
        control_path=(
            "/arm_joint_target_position",
            JOINT_TRACKER_NODE_NAME,
            "/arm_joint_command_a1_staged",
            SAFE_RELAY_SCRIPT,
            "/arm_joint_command_host",
        ),
        cameras=(CameraMetadata("front", "cam0", 640, 480),),
        config_path="configs/teleop/a1_so100.toml",
        quality_checks={"max_joint_action_step_rad": 0.35},
    )

    payload = metadata_to_json_dict(metadata)

    assert payload["state_mode"] == "eef_joint"
    assert payload["action_mode"] == "joint_absolute"
    assert payload["action_topics"]["joint_target"] == "/arm_joint_target_position"
    assert payload["control_path"][-1] == "/arm_joint_command_host"
    assert payload["cameras"][0]["modality"] == "rgb"
    assert payload["config_path"] == "configs/teleop/a1_so100.toml"
    assert payload["quality_checks"]["max_joint_action_step_rad"] == 0.35


def test_existing_experiment_rejects_mixed_camera_shapes(tmp_path: Path):
    episode = tmp_path / "episode_000_20260714_000000"
    episode.mkdir()
    (episode / "metadata.json").write_text(
        '{"cameras": [{"name": "front", "width": 640, "height": 480}]}'
    )

    try:
        validate_existing_camera_shape(
            tmp_path, camera_name="front", width=480, height=480
        )
    except ValueError as exc:
        assert "cannot append front 480x480" in str(exc)
    else:
        raise AssertionError("mixed camera dimensions should be rejected")


def test_existing_experiment_rejects_old_gripper_schema(tmp_path: Path):
    episode = tmp_path / "episode_000_20260714_000000"
    episode.mkdir()
    (episode / "metadata.json").write_text(
        '{"schema_version": "galaxea_a1_teleop_raw_v1"}'
    )

    try:
        validate_existing_schema(tmp_path, expected=TELEOP_RAW_SCHEMA_VERSION)
    except ValueError as exc:
        assert "use a new experiment name" in str(exc)
    else:
        raise AssertionError("old binary gripper schema should be rejected")
