"""Pinned Project-IRA checkpoint contract, separate from the unmapped base model."""
import hashlib
import json
from pathlib import Path

JOINT_NAMES = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
               "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
CHECKPOINT_SUBDIR = "outputs_V8/train/pi05_6gpu_fsdp_V2/checkpoints/008000/pretrained_model"


def validate_config(config):
    if (config.get("type") != "pi05" or config.get("action_feature_names") != JOINT_NAMES
            or config.get("use_relative_actions") is not False
            or config.get("chunk_size") != 50 or config.get("n_action_steps") != 50
            or config.get("empty_cameras") != 0
            or config.get("image_resolution") != [224, 224]):
        raise ValueError("Pi0.5 SO101 action/chunk contract mismatch")
    inputs, outputs = config["input_features"], config["output_features"]
    expected = {"observation.state", "observation.images.wrist_left", "observation.images.desk_view"}
    if (set(inputs) != expected or inputs["observation.state"]["shape"] != [6]
            or set(outputs) != {"action"} or outputs["action"]["shape"] != [6]):
        raise ValueError("Pi0.5 SO101 feature contract mismatch")
    if config.get("normalization_mapping") != {
            "ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}:
        raise ValueError("Pi0.5 SO101 normalization mismatch")


def verify_checkpoint(path):
    manifest = json.loads(Path(__file__).with_name("pi05_so101_manifest.json").read_text())
    for name, expected in manifest.items():
        file = path / name
        if not file.is_file() or file.stat().st_size != expected["size"]:
            raise ValueError(f"Missing or wrong-size SO101 checkpoint artifact: {name}")
        with file.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected["sha256"]:
            raise ValueError(f"SO101 checkpoint artifact hash mismatch: {name}")
    validate_config(json.loads((path / "config.json").read_text()))

import os


def load(runtime):
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.factory import make_pre_post_processors
    config = PreTrainedConfig.from_pretrained(runtime.snapshot)
    config.device = "cuda"
    config.dtype = "bfloat16"
    config.compile_model = False
    # Upstream from_pretrained can return randomly initialized weights when
    # the file loader raises. Load strictly here; never turn that into a pass.
    from safetensors.torch import load_file
    weights = Path(runtime.snapshot) / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(weights)
    runtime.policy = PI05Policy(config)
    state = load_file(str(weights), device="cpu")
    state = runtime.policy._fix_pytorch_state_dict_keys(state, config)
    state = {key if key.startswith("model.") else "model." + key: value
             for key, value in state.items()}
    runtime.policy.load_state_dict(state, strict=True)
    del state
    runtime.pre, runtime.post = make_pre_post_processors(
        config, pretrained_path=runtime.snapshot,
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "tokenizer_processor": {
                "tokenizer_name": os.environ.get("MODEL_SWAP_TOKENIZER", "/opt/pi05-tokenizer"),
            },
        },
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    if runtime.profile.name == "pi05-so101":
        stats = load_file(str(Path(runtime.snapshot) /
            "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))
        runtime.rtc_mean = stats["action.mean"].to("cuda")
        runtime.rtc_std = stats["action.std"].to("cuda")
        if not (runtime.rtc_std > 0).all():
            raise ValueError("RTC requires invertible saved action scaling")
