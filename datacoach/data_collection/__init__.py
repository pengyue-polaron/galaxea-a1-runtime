from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datacoach.data_collection.data_collector import ReplayAction

__all__ = ["ReplayAction"]


def __getattr__(name):
    # Delay importing heavy runtime deps (cv2, etc.) until ReplayAction is actually used.
    if name == "ReplayAction":
        from datacoach.data_collection.data_collector import ReplayAction as _ReplayAction

        return _ReplayAction
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
