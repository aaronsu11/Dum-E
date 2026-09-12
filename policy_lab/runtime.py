"""LeRobot policy/processor adapter, GPU only, serialized and evaluation only."""
from collections import Counter
from pathlib import Path
import os
import time
import threading

from .profiles import get_profile


class ModelRuntime:
    def __init__(self, profile_name):
        import torch
        from huggingface_hub import snapshot_download

        self.profile = get_profile(profile_name)
        self.lock = threading.Lock()
        self.fault = None
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; CPU model inference is disabled")
        if self.profile.policy_type == "groot":
            self.snapshot = os.environ.get("MODEL_SWAP_GROOT_CHECKPOINT", "/checkpoints/model")
            self._verify_groot_checkpoint()
        else:
            self.snapshot = snapshot_download(
                self.profile.repo, revision=self.profile.revision,
                allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.model"],
            )
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        if self.profile.policy_type == "pi05":
            self._load_pi05()
        elif self.profile.policy_type == "molmoact2":
            self._load_molmoact2()
        else:
            self._load_groot()
        self.policy.eval().to("cuda")
        torch.cuda.synchronize()
        self.load_s = time.monotonic() - started
        self.parameter_dtypes = dict(Counter(
            str(p.dtype) for p in self.policy.parameters()
        ))
        if any(p.device.type != "cuda" for p in self.policy.parameters()):
            raise RuntimeError("Every model parameter must reside on CUDA")

    def _verify_groot_checkpoint(self):
        from policy_guard.replay_contract import checkpoint_inventory, fingerprint_configuration
        refs, _ = checkpoint_inventory(Path(self.snapshot))
        if fingerprint_configuration(refs) != self.profile.revision:
            raise ValueError("GR00T checkpoint differs from the validated Phase 7 input lock")

    def _load_groot(self):
        import importlib.util
        from lerobot.policies.groot.configuration_groot import GrootConfig
        from lerobot.policies.factory import make_pre_post_processors
        from policy_guard.groot_guard import (
            EXPECTED_TAG, serving_preprocessor_overrides,
            snapshot_from_loaded, assert_groot_serving_contract,
        )
        source = Path("/app/docker/lerobot-policy/server.py")
        spec = importlib.util.spec_from_file_location("dume_validated_groot_server", source)
        server = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(server)
        config = GrootConfig(base_model_path=self.snapshot, embodiment_tag=EXPECTED_TAG,
                             model_params_fp32=False)
        server.fixup_policy_features(config, ["front", "wrist"], 480, 640, 6, 6)
        self.policy = server.DumEGrootPolicy.from_pretrained(self.snapshot, config=config)
        self.pre, self.post = make_pre_post_processors(
            self.policy.config, pretrained_path=self.snapshot,
            preprocessor_overrides={
                "device_processor": {"device": "cuda"},
                "rename_observations_processor": {"rename_map": {}},
                **serving_preprocessor_overrides(),
            },
            postprocessor_overrides={"device_processor": {"device": "cuda"}},
        )
        assert_groot_serving_contract(
            snapshot_from_loaded(self.policy.config, self.pre, self.post, self.profile.horizon)
        )

    def _load_pi05(self):
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.policies.factory import make_pre_post_processors
        config = PreTrainedConfig.from_pretrained(self.snapshot)
        config.device = "cuda"
        config.dtype = "bfloat16"
        config.compile_model = False
        # Upstream from_pretrained can return randomly initialized weights when
        # the file loader raises. Load strictly here; never turn that into a pass.
        from safetensors.torch import load_file
        weights = Path(self.snapshot) / "model.safetensors"
        if not weights.is_file():
            raise FileNotFoundError(weights)
        self.policy = PI05Policy(config)
        state = load_file(str(weights), device="cpu")
        state = self.policy._fix_pytorch_state_dict_keys(state, config)
        state = {key if key.startswith("model.") else "model." + key: value
                 for key, value in state.items()}
        self.policy.load_state_dict(state, strict=True)
        del state
        self.pre, self.post = make_pre_post_processors(
            config, pretrained_path=self.snapshot,
            preprocessor_overrides={
                "device_processor": {"device": "cuda"},
                "tokenizer_processor": {
                    "tokenizer_name": os.environ.get("MODEL_SWAP_TOKENIZER", "/opt/pi05-tokenizer"),
                },
            },
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )

    def _load_molmoact2(self):
        from lerobot.configs import PolicyFeature, FeatureType
        from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
        from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
        from lerobot.policies.molmoact2.processor_molmoact2 import make_molmoact2_pre_post_processors
        config = MolmoAct2Config(
            checkpoint_path=self.snapshot, device="cuda", model_dtype="bfloat16",
            norm_tag="so100_so101_molmoact2", action_mode="continuous",
            inference_action_mode="continuous", normalize_gripper=True,
            enable_inference_cuda_graph=False, num_inference_steps=10,
            chunk_size=30, n_action_steps=30,
            image_keys=["observation.images.front", "observation.images.wrist"],
            input_features={
                "observation.state": PolicyFeature(FeatureType.STATE, (6,)),
                "observation.images.front": PolicyFeature(FeatureType.VISUAL, (3, 480, 640)),
                "observation.images.wrist": PolicyFeature(FeatureType.VISUAL, (3, 480, 640)),
            },
            output_features={"action": PolicyFeature(FeatureType.ACTION, (6,))},
        )
        self.policy = MolmoAct2Policy(config)
        self.pre, self.post = make_molmoact2_pre_post_processors(config)

    def health(self):
        import torch
        return {
            "profile": self.profile.to_dict(), "status": "failed" if self.fault else "ready",
            "fault": self.fault, "load_s": self.load_s,
            "gpu": torch.cuda.get_device_name(), "parameter_dtypes": self.parameter_dtypes,
            "allocated_mib": torch.cuda.memory_allocated() / 2**20,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "reserved_mib": torch.cuda.memory_reserved() / 2**20,
        }

    def infer(self, state, front, wrist, task, seed):
        import torch
        with self.lock:
            if self.fault:
                raise RuntimeError("Model session failed; restart required")
            try:
                return self._infer(state, front, wrist, task, seed)
            except torch.cuda.OutOfMemoryError:
                self.fault = "CUDA out of memory"
                raise

    def _infer(self, state, front, wrist, task, seed):
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.policy.reset()
        def image(array):
            return torch.from_numpy(array).permute(2, 0, 1).float() / 255
        if self.profile.policy_type == "pi05":
            padded = torch.zeros(32)
            padded[:6] = torch.from_numpy(state)
            batch = {
                "observation.state": padded, "task": task,
                "observation.images.base_0_rgb": image(front),
                "observation.images.left_wrist_0_rgb": image(wrist),
                # Explicit placeholder for smoke testing the base model only.
                "observation.images.right_wrist_0_rgb": torch.zeros_like(image(wrist)),
            }
        else:
            batch = {"observation.state": torch.from_numpy(state), "task": task,
                     "observation.images.front": image(front),
                     "observation.images.wrist": image(wrist)}
        times = {}
        start = time.perf_counter()
        with torch.inference_mode():
            batch = self.pre(batch)
            torch.cuda.synchronize()
            predicted_at = time.perf_counter()
            actions = self.policy.predict_action_chunk(batch)
            if self.profile.policy_type == "groot":
                actions = actions[:, :self.profile.horizon]
            torch.cuda.synchronize()
            decoded_at = time.perf_counter()
            actions = self.post(actions).detach().float().cpu()
        expected = (1, self.profile.horizon, self.profile.action_dim)
        if tuple(actions.shape) != expected or not torch.isfinite(actions).all():
            raise RuntimeError(f"Invalid action chunk: {tuple(actions.shape)}, expected {expected}")
        end = time.perf_counter()
        times.update(preprocess_ms=1000 * (predicted_at - start),
                     generation_ms=1000 * (decoded_at - predicted_at),
                     postprocess_ms=1000 * (end - decoded_at),
                     total_ms=1000 * (end - start))
        return {"actions": actions[0].tolist(), "timings": times, "health": self.health(),
                "physical_ready": False, "seed": seed}
