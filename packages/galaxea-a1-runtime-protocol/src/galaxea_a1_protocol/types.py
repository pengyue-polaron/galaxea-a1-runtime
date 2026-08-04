"""A1 Runtime transport-owned session types."""

from enum import StrEnum

PROTOCOL_VERSION = 1


class SessionMode(StrEnum):
    OBSERVE = "observe"
    COMMAND = "command"
