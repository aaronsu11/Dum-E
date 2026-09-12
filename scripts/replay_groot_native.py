"""Native Python 3.10 replay through unmodified pinned GR00T components."""

from __future__ import annotations

import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_guard.replay_contract import BACKBONE_REVISION, CAMERA_ORDER, PrerequisiteError, observed_model, worker_main  # noqa: E402


@contextmanager
def pinned_native_cache(hub_root=None):
    """Resolve the unchanged model ID as a local path in an owned temporary cwd.

    Transformers 4.57.3 skips tokenizer Hub metadata discovery for a local
    directory. The literal ID must stay unchanged for native backbone dispatch.
    The cached snapshot and its blob links remain read-only and unmodified.
    """
    if hub_root is None:
        from huggingface_hub.constants import HF_HUB_CACHE
        hub_root = HF_HUB_CACHE
    model = Path(hub_root).resolve() / "models--nvidia--Cosmos-Reason2-2B"
    snapshot = model / "snapshots" / BACKBONE_REVISION
    if not snapshot.is_dir() or not (model / "refs/main").is_file():
        raise PrerequisiteError("pinned native backbone snapshot is absent")
    if (model / "refs/main").read_text().strip() != BACKBONE_REVISION:
        raise PrerequisiteError("native backbone default revision differs from pin")
    if not snapshot.resolve().is_relative_to(model):
        raise ValueError("native snapshot escapes model cache")
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        if not (snapshot / name).is_file():
            raise PrerequisiteError(f"pinned snapshot missing {name}")
    for path in snapshot.rglob("*"):
        if path.is_symlink() and (not path.exists() or not path.resolve().is_relative_to(model)):
            raise ValueError(f"native snapshot blob escapes model cache: {path.name}")
    previous = os.open(".", os.O_RDONLY)
    try:
        with tempfile.TemporaryDirectory(prefix="dume-native-cache-", dir="/tmp") as temporary:
            root = Path(temporary)
            (root / "nvidia").mkdir()
            (root / "nvidia/Cosmos-Reason2-2B").symlink_to(snapshot, target_is_directory=True)
            try:
                os.chdir(root)
                yield snapshot
            finally:
                os.fchdir(previous)
    finally:
        os.close(previous)


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
    checkpoint = checkpoint.resolve()
    with pinned_native_cache():
        policy = Gr00tPolicy("new_embodiment", str(checkpoint), device=device)
        fair_model = diagnostic_model(checkpoint, device)
    return policy, fair_model


class NativeReplay:
    backend = "native"

    def __init__(self, checkpoint: Path, purpose: str, device: str):
        checkpoint = checkpoint.resolve()
        with pinned_native_cache():
            self._construct(checkpoint, purpose, device)

    def _construct(self, checkpoint: Path, purpose: str, device: str):
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

    def effective_configuration(self):
        processor = {name: getattr(self.processor, name) for name in (
            "use_percentiles", "use_mean_std", "clip_outliers", "apply_sincos_state_encoding",
            "use_relative_action", "exclude_state", "state_dropout_prob", "letter_box_transform",
            "formalize_language", "model_name", "model_type", "max_state_dim", "max_action_dim",
            "max_action_horizon", "image_crop_size", "image_target_size", "shortest_image_edge",
            "crop_fraction", "use_albumentations", "training",
        )}
        processor["modalities"] = self.processor.get_modality_configs()
        processor["statistics"] = self.processor.statistics
        processor["tokenizer_padding_side"] = self.processor.processor.tokenizer.padding_side
        return {"model": self.raw_model.config.to_dict(), "processor": processor}

    def predict(self, arrays, entry):
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
            from gr00t.data.types import MessageType, VLAStepData
            from gr00t.policy.gr00t_policy import _rec_to_dtype

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
