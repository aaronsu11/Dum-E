import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
import numpy as np
import pytest

from policy_lab.profiles import get_profile
from policy_lab.protocol import decode_request, encode_image
from policy_lab.server import handler


def payload():
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    return {"state": [0.] * 6, "front": encode_image(rgb), "wrist": encode_image(rgb),
            "task": "pick the banana", "seed": 123}


def test_profiles_keep_full_native_horizons_and_no_physical_approval():
    assert (get_profile("pi05-base").horizon, get_profile("pi05-base").action_dim) == (50, 32)
    assert (get_profile("molmoact2-so101").horizon, get_profile("molmoact2-so101").action_dim) == (30, 6)
    assert not get_profile("pi05-base").physical_ready
    assert not get_profile("molmoact2-so101").physical_ready
    with pytest.raises(ValueError):
        get_profile("../../model")


@pytest.mark.parametrize("field,value", [
    ("state", [float("nan")] * 6), ("state", [True] * 6), ("state", [0] * 32),
    ("seed", -1), ("seed", True), ("task", ""), ("task", "x" * 1025),
    ("front", {"shape": [100000, 100000, 3], "rgb_base64": ""}),
    ("wrist", {"shape": [2, 3, 3], "rgb_base64": "not-base64"}),
])
def test_untrusted_payload_is_rejected(field, value):
    data = payload()
    data[field] = value
    with pytest.raises((ValueError, TypeError)):
        decode_request(data)


def test_rgb_roundtrip_and_extra_fields_rejected():
    data = payload()
    state, front, wrist, task, seed = decode_request(data)
    assert state.dtype == np.float32 and front.shape == wrist.shape == (2, 3, 3)
    assert task == "pick the banana" and seed == 123
    data["checkpoint"] = "untrusted"
    with pytest.raises(ValueError):
        decode_request(data)


def test_http_rejects_concurrent_inference_and_remains_responsive():
    entered, release = threading.Event(), threading.Event()
    class Runtime:
        def health(self):
            return {"status": "ready", "physical_ready": False}
        def infer(self, *args):
            entered.set()
            assert release.wait(5)
            return {"physical_ready": False}
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler(Runtime()))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    errors = []
    def infer():
        try:
            urllib.request.urlopen(endpoint + "/infer", json.dumps(payload()).encode()).close()
        except Exception as exc:
            errors.append(exc)
    worker = threading.Thread(target=infer)
    try:
        worker.start()
        assert entered.wait(3)
        with urllib.request.urlopen(endpoint + "/health") as response:
            assert json.load(response)["physical_ready"] is False
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(endpoint + "/infer", json.dumps(payload()).encode())
        assert err.value.code == 409
    finally:
        release.set()
        worker.join(5)
        server.shutdown()
        server.server_close()
        thread.join(5)
    assert not errors
