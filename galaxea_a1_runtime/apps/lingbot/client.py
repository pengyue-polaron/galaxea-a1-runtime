"""LingBot websocket protocol and NumPy MessagePack codec."""

from __future__ import annotations

import functools

import msgpack
import numpy as np
import websockets.sync.client

from galaxea_a1_runtime.console import info, success


def _pack_array(obj):
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"]
        )
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=_pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


class LingBotClient:
    def __init__(
        self,
        host: str,
        port: int,
        *,
        connect_timeout_s: float,
        close_timeout_s: float,
    ):
        self.uri = f"ws://{host}:{port}"
        self.packer = Packer()
        self.ws = None
        info(f"Connecting to LingBot: {self.uri}")
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
        except BaseException:
            self.close()
            raise
        success(f"LingBot connected: metadata={self.metadata}")

    def infer(self, obs: dict) -> dict:
        if self.ws is None:
            raise RuntimeError("LingBot client is closed")
        self.ws.send(self.packer.pack(obs))
        response = self.ws.recv()
        if isinstance(response, str):
            raise RuntimeError(response)
        return unpackb(response)

    def reset(self, prompt: str) -> None:
        self.infer({"reset": True, "prompt": prompt})

    def close(self) -> None:
        ws, self.ws = self.ws, None
        if ws is not None:
            ws.close()
