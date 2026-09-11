"""Arm-free replay reusing the owned guarded full-chunk serving implementation."""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from policy_guard.replay_contract import JOINT_ORDER, observed_model, worker_main  # noqa: E402


class LocalContext:
    """Local handler protocol; no listener, channel, or remote pickle."""

    def peer(self):
        return "offline-replay"

    def abort(self, code, detail):
        raise ValueError(f"{code}: {detail}")


def diagnostic_server(checkpoint, device):
    import torch
    from lerobot.async_inference.configs import PolicyServerConfig
    from lerobot.policies.groot.modeling_groot import GrootPolicy
    from server import (
        DumEGrootPolicyServer, GrootConfig, GR00TN17, fixup_policy_features,
        make_pre_post_processors, serving_preprocessor_overrides,
        snapshot_from_loaded, assert_groot_serving_contract,
    )
    from policy.lerobot.features import build_lerobot_features

    class Fp32Policy(GrootPolicy):
        def _create_groot_model(self):
            return GR00TN17.from_pretrained(
                self.config.base_model_path, dtype=torch.float32, load_bf16=False,
                use_flash_attention=False, num_inference_timesteps=4,
                transformers_loading_kwargs={
                    "trust_remote_code": True, "local_files_only": True,
                    "dtype": torch.float32, "attn_implementation": "sdpa",
                }, local_files_only=True,
            ).float().eval()

    server = DumEGrootPolicyServer(PolicyServerConfig())
    server.device = device
    server.policy_type = "groot"
    server.actions_per_chunk = 16
    server.lerobot_features = build_lerobot_features()
    config = GrootConfig(
        base_model_path=str(checkpoint), embodiment_tag="new_embodiment",
        model_params_fp32=True, use_bf16=False, use_flash_attention=False,
        num_inference_timesteps=4,
    )
    fixup_policy_features(
        config, camera_keys=server.CAMERA_KEYS, height=server.FRAME_HEIGHT,
        width=server.FRAME_WIDTH, state_dim=6, action_dim=6,
    )
    server.policy = Fp32Policy.from_pretrained(str(checkpoint), config=config).to(device).eval()
    overrides = {
        "device_processor": {"device": device},
        "rename_observations_processor": {"rename_map": {}},
        **serving_preprocessor_overrides(),
    }
    server.preprocessor, server.postprocessor = make_pre_post_processors(
        config, pretrained_path=str(checkpoint), preprocessor_overrides=overrides,
        postprocessor_overrides={"device_processor": {"device": device}},
    )
    snapshot = snapshot_from_loaded(
        server.policy.config, server.preprocessor, server.postprocessor, configured_actions_per_chunk=16,
    )
    assert_groot_serving_contract(snapshot)
    return server


class LeRobotReplay:
    backend = "lerobot"

    def __init__(self, checkpoint: Path, purpose: str, device: str):
        from lerobot.async_inference.configs import PolicyServerConfig
        from lerobot.async_inference.helpers import RemotePolicyConfig
        from policy.lerobot.features import build_lerobot_features, assert_state_ordering
        from server import DumEGrootPolicyServer

        self.purpose = purpose
        if purpose == "operational":
            self.server = DumEGrootPolicyServer(PolicyServerConfig())
            context = LocalContext()
            self.server.Ready(SimpleNamespace(), context)
            features = build_lerobot_features()
            assert_state_ordering(features)
            specs = RemotePolicyConfig("groot", str(checkpoint), features, actions_per_chunk=16, device=device)
            self.server.SendPolicyInstructions(SimpleNamespace(data=pickle.dumps(specs)), context)
            if self.server.policy is None:
                raise RuntimeError("real load handler did not construct a policy")
            self.path = "DumEGrootPolicyServer.SendPolicyInstructions -> _predict_action_chunk"
        else:
            self.server = diagnostic_server(checkpoint, device)
            self.path = "owned fp32 loader + SAFE-01 -> DumEGrootPolicyServer._predict_action_chunk"
        self.raw_model = self.server.policy._groot_model
        self.observed = observed_model(self.raw_model)
        self.server.policy._groot_model = self.observed
        self.model = self.server.policy
        self.model.eval()

    def attention_implementations(self):
        config = self.raw_model.backbone.model.config
        return {
            getattr(config, "_attn_implementation"),
            getattr(config.text_config, "_attn_implementation"),
            getattr(config.vision_config, "_attn_implementation"),
        }

    def effective_configuration(self):
        from policy_guard.groot_guard import snapshot_from_loaded
        return {
            "model": self.raw_model.config.to_dict(), "policy": self.server.policy.config,
            "serving": snapshot_from_loaded(
                self.server.policy.config, self.server.preprocessor, self.server.postprocessor,
                configured_actions_per_chunk=self.server.actions_per_chunk,
            ),
        }

    def predict(self, arrays, entry):
        from lerobot.async_inference.helpers import TimedObservation

        raw = {key: float(value) for key, value in zip(JOINT_ORDER, arrays["state"], strict=True)}
        raw.update(front=arrays["video_front"], wrist=arrays["video_wrist"], task=entry["instruction"])
        observation = TimedObservation(timestamp=time.time(), timestep=0, observation=raw, must_go=True)
        actions = self.server._predict_action_chunk(observation)
        return np.stack([action.get_action().float().numpy() for action in actions])


if __name__ == "__main__":
    raise SystemExit(worker_main("lerobot", LeRobotReplay))
