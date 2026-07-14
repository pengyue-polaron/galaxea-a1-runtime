"""LingBot websocket protocol and NumPy MessagePack codec."""

from __future__ import annotations

import functools

import msgpack
import numpy as np
import websockets.sync.client


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
    def __init__(self, host: str, port: int):
        self.uri = f"ws://{host}:{port}"
        self.packer = Packer()
        print(f"[LingBot] Connecting to {self.uri} ...")
        self.ws = websockets.sync.client.connect(
            self.uri,
            compression=None,
            max_size=None,
            ping_interval=None,
            close_timeout=10,
        )
        self.metadata = unpackb(self.ws.recv())
        print(f"[LingBot] Connected. metadata={self.metadata}")

    def infer(self, obs: dict) -> dict:
        self.ws.send(self.packer.pack(obs))
        response = self.ws.recv()
        if isinstance(response, str):
            raise RuntimeError(response)
        return unpackb(response)

    def reset(self, prompt: str) -> None:
        self.infer({"reset": True, "prompt": prompt})
