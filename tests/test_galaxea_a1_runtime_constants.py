import pytest

from galaxea_a1_runtime.constants import (
    EE_TRACKER_NODE,
    EE_TRACKER_NODE_NAME,
    JOINT_TRACKER_NODE,
    JOINT_TRACKER_NODE_NAME,
)


@pytest.mark.parametrize(
    ("launch_name", "graph_name"),
    (
        (EE_TRACKER_NODE_NAME, EE_TRACKER_NODE),
        (JOINT_TRACKER_NODE_NAME, JOINT_TRACKER_NODE),
    ),
)
def test_tracker_names_separate_roslaunch_basename_from_graph_name(
    launch_name: str, graph_name: str
) -> None:
    assert "/" not in launch_name
    assert graph_name == f"/{launch_name}"
