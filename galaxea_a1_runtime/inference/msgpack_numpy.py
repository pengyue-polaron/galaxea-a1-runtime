"""Safe NumPy MessagePack codec shared by model service adapters."""

from __future__ import annotations

import functools

import msgpack
import numpy as np


def _pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"unsupported inference array dtype: {obj.dtype}")
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
        dtype = np.dtype(obj[b"dtype"])
        if dtype.kind in ("V", "O", "c"):
            raise ValueError(f"unsupported inference array dtype: {dtype}")
        return np.ndarray(buffer=obj[b"data"], dtype=dtype, shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        dtype = np.dtype(obj[b"dtype"])
        if dtype.kind in ("V", "O", "c"):
            raise ValueError(f"unsupported inference array dtype: {dtype}")
        return dtype.type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=_pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)
