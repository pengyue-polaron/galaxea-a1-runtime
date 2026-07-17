"""Contract-checked synchronous inference websocket client."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import websockets.sync.client

from galaxea_a1_runtime.console import info, success
from galaxea_a1_runtime.inference.msgpack_numpy import Packer, unpackb


class WebsocketInferenceClient:
    def __init__(
        self,
        host: str,
        port: int,
        *,
        connect_timeout_s: float,
        close_timeout_s: float,
        expected_metadata: dict[str, Any],
        validate_metadata: Callable[[object, dict[str, Any]], None],
        label: str,
    ) -> None:
        self.uri = f"ws://{host}:{port}"
        self.packer = Packer()
        self.ws = None
        self.label = label
        info(f"Connecting to {label}: {self.uri}")
        try:
            self.ws = websockets.sync.client.connect(
                self.uri,
                compression=None,
                max_size=None,
                ping_interval=None,
                close_timeout=close_timeout_s,
                open_timeout=connect_timeout_s,
            )
            self.metadata = unpackb(self.ws.recv())
            validate_metadata(self.metadata, expected_metadata)
        except BaseException:
            self.close()
            raise
        success(f"{label} connected: contract={expected_metadata['contract_sha256']}")

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self.ws is None:
            raise RuntimeError(f"{self.label} client is closed")
        self.ws.send(self.packer.pack(observation))
        response = self.ws.recv()
        if isinstance(response, str):
            raise RuntimeError(response)
        decoded = unpackb(response)
        if not isinstance(decoded, dict):
            raise RuntimeError(
                f"{self.label} response must be a dictionary, got "
                f"{type(decoded).__name__}"
            )
        return decoded

    def close(self) -> None:
        websocket, self.ws = self.ws, None
        if websocket is not None:
            websocket.close()
