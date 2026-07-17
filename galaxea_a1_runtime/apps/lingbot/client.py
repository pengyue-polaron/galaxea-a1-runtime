"""LingBot commands layered on the shared websocket inference client."""

from __future__ import annotations

from typing import Any

from galaxea_a1_runtime.apps.lingbot.protocol import validate_server_metadata
from galaxea_a1_runtime.inference.websocket_client import WebsocketInferenceClient


class LingBotClient(WebsocketInferenceClient):
    def __init__(
        self,
        host: str,
        port: int,
        *,
        connect_timeout_s: float,
        close_timeout_s: float,
        expected_metadata: dict[str, Any],
    ) -> None:
        super().__init__(
            host,
            port,
            connect_timeout_s=connect_timeout_s,
            close_timeout_s=close_timeout_s,
            expected_metadata=expected_metadata,
            validate_metadata=validate_server_metadata,
            label="LingBot",
        )

    def reset(self, prompt: str) -> None:
        self.infer({"reset": True, "prompt": prompt})
