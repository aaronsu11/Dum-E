"""Bounded implementation of Galaxea's ndarray/msgpack wire representation."""
import math
import msgpack
import numpy as np

MAX_MESSAGE = 16 * 1024**2


def _encode(value):
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf":
            raise ValueError("Unsupported ndarray dtype")
        return {"__ndarray__": True, "data": value.tobytes(),
                "dtype": value.dtype.str, "shape": value.shape}
    if isinstance(value, np.generic):
        return {"__npgeneric__": True, "data": value.item(), "dtype": value.dtype.str}
    raise TypeError(f"Unsupported wire value: {type(value).__name__}")


def _decode(value):
    value = {k.decode() if isinstance(k, bytes) else k: v for k, v in value.items()}
    if "__ndarray__" in value:
        if set(value) != {"__ndarray__", "data", "dtype", "shape"} or value["__ndarray__"] is not True:
            raise ValueError("Malformed ndarray envelope")
        shape = value["shape"]
        dtype = np.dtype(value["dtype"])
        if (dtype.kind not in "biuf" or dtype.itemsize > 8
                or not isinstance(shape, (list, tuple)) or len(shape) > 4
                or any(type(n) is not int or n < 0 for n in shape)):
            raise ValueError("Invalid ndarray shape/dtype")
        size = math.prod(shape) * dtype.itemsize
        if not isinstance(value["data"], bytes) or size > MAX_MESSAGE or len(value["data"]) != size:
            raise ValueError("Invalid ndarray byte count")
        return np.frombuffer(value["data"], dtype=dtype).reshape(shape).copy()
    if "__npgeneric__" in value:
        dtype = np.dtype(value["dtype"])
        if dtype.kind not in "biuf" or dtype.itemsize > 8:
            raise ValueError("Unsupported scalar dtype")
        return dtype.type(value["data"])
    return value


def packb(value):
    result = msgpack.packb(value, default=_encode, use_bin_type=True)
    if len(result) > MAX_MESSAGE:
        raise ValueError("Message too large")
    return result


def unpackb(value):
    if not isinstance(value, bytes) or len(value) > MAX_MESSAGE:
        raise ValueError("Invalid message size/type")
    return msgpack.unpackb(value, raw=False, object_hook=_decode,
                          max_bin_len=MAX_MESSAGE, max_array_len=4096,
                          max_map_len=256, max_str_len=65536)
