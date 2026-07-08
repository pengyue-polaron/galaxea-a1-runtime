from pathlib import Path

from galaxea_a1_runtime.collection import (
    CameraMetadata,
    EpisodeDecision,
    StateMode,
    TeleopRawEpisodeMetadata,
    action_columns,
    metadata_to_json_dict,
    next_episode_index,
    normalize_episode_decision,
    state_columns,
    state_names_for_mode,
    teleop_frame_header,
)
from galaxea_a1_runtime.collection.schema import TELEOP_RAW_SCHEMA_VERSION
from galaxea_a1_runtime.schema import ActionMode, JOINT_ACTION_NAMES


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


def test_next_episode_index_uses_existing_episode_prefix(tmp_path):
    (tmp_path / "episode_000_20260708_120000").mkdir()
    (tmp_path / "episode_002_20260708_120100").mkdir()
    (tmp_path / "notes").mkdir()

    assert next_episode_index(Path(tmp_path)) == 3


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
            "jointTracker_demo_node",
            "/arm_joint_command_a1_staged",
            "safe_arm_command_relay_v2.py",
            "/arm_joint_command_host",
        ),
        cameras=(CameraMetadata("front", "cam0", 640, 480),),
    )

    payload = metadata_to_json_dict(metadata)

    assert payload["state_mode"] == "eef_joint"
    assert payload["action_mode"] == "joint_absolute"
    assert payload["action_topics"]["joint_target"] == "/arm_joint_target_position"
    assert payload["control_path"][-1] == "/arm_joint_command_host"
    assert payload["cameras"][0]["modality"] == "rgb"
