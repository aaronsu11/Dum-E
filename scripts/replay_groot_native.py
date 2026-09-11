"""Native Python 3.10 replay through unmodified pinned GR00T components."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_guard.replay_contract import CAMERA_ORDER, observed_model, worker_main  # noqa: E402


def diagnostic_model(checkpoint, device):
    import torch
    import gr00t.model  # noqa: F401 -- upstream AutoModel registration
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7

    model = Gr00tN1d7.from_pretrained(
        checkpoint, torch_dtype=torch.float32, load_bf16=False,
        use_flash_attention=False, num_inference_timesteps=4,
        transformers_loading_kwargs={
            "trust_remote_code": True, "local_files_only": True,
            "torch_dtype": torch.float32, "attn_implementation": "sdpa",
        }, local_files_only=True,
    )
    return model.to(device=device, dtype=torch.float32).eval()


def stock_capacity(checkpoint, device):
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    # Stock lifetime: policy remains alive while the separate fair_model loads.
    policy = Gr00tPolicy("new_embodiment", str(checkpoint), device=device)
    fair_model = diagnostic_model(checkpoint, device)
    return policy, fair_model


class NativeReplay:
    backend = "native"

    def __init__(self, checkpoint: Path, purpose: str, device: str):
        import torch
        from transformers import AutoProcessor
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        self.purpose = purpose
        self.tag = EmbodimentTag.NEW_EMBODIMENT
        if purpose == "operational":
            self.policy = Gr00tPolicy(self.tag, str(checkpoint), device=device)
            self.raw_model = self.policy.model
            self.observed = observed_model(self.raw_model)
            self.policy.model = self.observed
            self.processor = self.policy.processor
            self.path = "Gr00tPolicy.get_action -> upstream _get_action -> client joint mapping"
        else:
            self.raw_model = diagnostic_model(checkpoint, device)
            self.observed = observed_model(self.raw_model)
            self.processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
            self.processor.eval()
            self.path = "native VLAStepData -> processor/collator -> fp32 model.get_action -> decode_action"
        self.model = self.raw_model
        self.model.eval()
        self.device = device
        self.dtype = torch.float32

    def attention_implementations(self):
        config = self.raw_model.backbone.model.config
        return {
            getattr(config, "_attn_implementation"),
            getattr(config.text_config, "_attn_implementation"),
            getattr(config.vision_config, "_attn_implementation"),
        }

    def predict(self, arrays, entry):
        from gr00t.data.types import MessageType, VLAStepData
        from gr00t.policy.gr00t_policy import _rec_to_dtype

        state = arrays["state"]
        states = {"single_arm": state[:5][None], "gripper": state[5:6][None]}
        if self.purpose == "operational":
            observation = {
                "video": {key: arrays[f"video_{key}"][None, None] for key in CAMERA_ORDER},
                "state": {key: value[None] for key, value in states.items()},
                "language": {"annotation.human.task_description": [[entry["instruction"]]]},
            }
            actions, _ = self.policy.get_action(observation)
        else:
            step = VLAStepData(
                embodiment=self.tag,
                images={key: arrays[f"video_{key}"][None] for key in CAMERA_ORDER},
                states=states, actions={}, text=entry["instruction"],
            )
            processed = self.processor([{"type": MessageType.EPISODE_STEP.value, "content": step}])
            collated = _rec_to_dtype(self.processor.collator([processed]), self.dtype)
            raw = self.observed.get_action(**collated)["action_pred"]
            actions = self.processor.decode_action(
                raw.detach().cpu().numpy(), self.tag,
                {key: value[None] for key, value in states.items()},
            )
        # Same native client mapping without controller imports or relative math.
        if set(actions) != {"single_arm", "gripper"}:
            raise ValueError(f"unexpected decoded modalities: {set(actions)}")
        if actions["single_arm"].shape != (1, 16, 5) or actions["gripper"].shape != (1, 16, 1):
            raise ValueError("native decoded horizon/dimension mismatch")
        return np.concatenate((actions["single_arm"][0], actions["gripper"][0]), axis=-1).astype(np.float32)


if __name__ == "__main__":
    raise SystemExit(worker_main("native", NativeReplay, stock_capacity))
