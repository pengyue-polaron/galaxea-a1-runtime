"""Model-service boundaries shared by inference applications."""

from .websocket_client import WebsocketInferenceClient

__all__ = ["WebsocketInferenceClient"]
