"""Opt-in two-request GPU smoke against an operator-owned loopback server.

No container is stopped, replaced or configured by this suite. Provision the
server described in docs/POLICY-SERVING.md before setting the opt-in variable.
"""
import os
import numpy as np
import pytest
from lerobot.async_inference.helpers import RemotePolicyConfig
from policy.backends.lerobot.features import build_lerobot_features, CAMERA_KEYS, ROBOT_STATE_KEYS
from policy.backends.lerobot.session import LeRobotPolicySession

pytestmark = pytest.mark.skipif(os.getenv("DUME_RUN_LIVE_LEROBOT_TESTS") != "1", reason="Explicit live GPU server required")


def test_two_complete_chunks_from_live_server():
    address = os.getenv("DUME_LEROBOT_POLICY_ADDRESS", "127.0.0.1:8080")
    if address.split(":")[0] not in ("127.0.0.1", "localhost"):
        raise ValueError("Use a loopback endpoint or SSH tunnel")
    session = LeRobotPolicySession(address)
    specs = RemotePolicyConfig(policy_type="groot", pretrained_name_or_path="/checkpoints/model",
        lerobot_features=build_lerobot_features(robot_state_keys=ROBOT_STATE_KEYS, camera_keys=CAMERA_KEYS),
        actions_per_chunk=16, device="cuda", rename_map={})
    try:
        session.connect(specs)
        rng = np.random.RandomState(0)
        for offset in (0., 0.1):
            obs = dict.fromkeys(ROBOT_STATE_KEYS, offset)
            obs.update({cam: rng.randint(0, 255, (480, 640, 3), dtype=np.uint8) for cam in CAMERA_KEYS})
            obs["task"] = "Pick up the banana and put it on the plate"
            actions = session.infer(obs)
            values = np.asarray([a.get_action().detach().cpu().numpy() for a in actions])
            assert values.shape == (16, 6) and np.isfinite(values).all()
    finally:
        session.close()
