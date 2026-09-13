"""Bounded JSON wire format: no pickle, checkpoint paths, or executable payloads."""
import base64
import binascii
import hashlib
import numpy as np

MAX_BODY_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 1024 * 1024


def encode_image(image):
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("RGB uint8 image required")
    return {"shape": list(image.shape),
            "rgb_base64": base64.b64encode(image.tobytes()).decode("ascii")}


def decode_image(data):
    shape = data["shape"]
    if (not isinstance(shape, list) or len(shape) != 3 or shape[-1] != 3
            or any(type(n) is not int or n <= 0 for n in shape)
            or shape[0] * shape[1] > MAX_IMAGE_PIXELS):
        raise ValueError("Invalid or excessive RGB dimensions")
    try:
        raw = base64.b64decode(data["rgb_base64"], validate=True)
    except (ValueError, binascii.Error, TypeError):
        raise ValueError("Invalid RGB encoding") from None
    if len(raw) != shape[0] * shape[1] * 3:
        raise ValueError("RGB byte count does not match shape")
    return np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy()


def decode_request(data):
    if not isinstance(data, dict) or set(data) != {"state", "front", "wrist", "task", "seed"}:
        raise ValueError("Expected state, front, wrist, task and seed only")
    state = np.asarray(data["state"])
    if state.shape != (6,) or state.dtype.kind not in "fi" or not np.isfinite(state).all():
        raise ValueError("Six finite numeric state values required")
    task = data["task"]
    if not isinstance(task, str) or not task.strip() or len(task) > 1024:
        raise ValueError("Nonempty task of at most 1024 characters required")
    seed = data["seed"]
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an unsigned 32-bit integer")
    return state.astype(np.float32), decode_image(data["front"]), decode_image(data["wrist"]), task, seed


def prefix_digest(prefix):
    if prefix is None:
        return hashlib.sha256(b"initial").hexdigest()
    return hashlib.sha256(np.asarray(prefix, dtype="<f8").tobytes()).hexdigest()


def decode_rtc_request(data):
    if not isinstance(data, dict) or set(data) != {
            "state", "front", "wrist", "task", "seed", "rtc"}:
        raise ValueError("Invalid RTC request fields")
    rtc = data["rtc"]
    if not isinstance(rtc, dict) or set(rtc) != {
            "epoch", "request_id", "prefix_arm", "delay_steps"}:
        raise ValueError("Invalid RTC metadata")
    for key in ("epoch", "request_id"):
        if type(rtc[key]) is not int or not 0 <= rtc[key] < 2**53:
            raise ValueError("Invalid RTC identity")
    prefix = rtc["prefix_arm"]
    if prefix is None:
        if type(rtc["delay_steps"]) is not int or rtc["delay_steps"] != 0:
            raise ValueError("Initial RTC request requires zero delay")
    else:
        prefix = np.asarray(prefix)
        if (prefix.shape != (25, 6) or prefix.dtype.kind not in "fiu"
                or not np.isfinite(prefix).all()):
            raise ValueError("RTC requires 25 finite bounded six-joint targets")
        if type(rtc["delay_steps"]) is not int or not 1 <= rtc["delay_steps"] < 25:
            raise ValueError("Invalid RTC delay")
    return decode_request({k: v for k, v in data.items() if k != "rtc"}), rtc
