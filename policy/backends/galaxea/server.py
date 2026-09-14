"""Bounded Galaxea-native SO101 WebSocket serving, without robot ownership."""
import argparse
import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path
import threading
import time
import traceback

import numpy as np
from policy.backends.galaxea.codec import MAX_MESSAGE, packb, unpackb
from policy.backends.galaxea.protocol import CODEC_PARTS, validate_codec_presence

SOURCE_REVISION = "89f2322b4ad016e192437adc1a2c253b05bab246"
CHECKPOINT_REVISION = "e312be81e90c56a55bcb26b57429bd39a335b449"
HORIZON = 32
TOKEN_LIMIT = 300


class Runtime:
    def __init__(self, checkpoint_root):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; no CPU inference fallback")
        self.lock = threading.Lock()
        self.fault = None
        self.tokens_generated = 0
        root = Path(checkpoint_root)
        manifest = json.loads(Path(__file__).with_name("checkpoint_manifest.json").read_text())
        for ref in manifest["files"]:
            path = root / ref["path"]
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024**2), b""):
                    digest.update(block)
            if digest.hexdigest() != ref["sha256"]:
                raise ValueError(f"Checkpoint identity mismatch: {ref['path']}")
        import scripts.serve_policy as native
        from g05.models.g05.inferencer import PolicyInferencer
        checkpoint = root / "g05-so101/checkpoints/model_state_dict.pt"
        run_dir = native.find_run_dir(str(checkpoint))
        cfg = native.load_config_from_run_dir(run_dir, str(checkpoint), [
            "eval_embodiment=so100", "model.model_weights_to_bf16=true",
            "model.use_torch_compile=false", "model.model_arch.attn_implementation=sdpa",
            f"model.model_arch.ar.max_new_tokens={TOKEN_LIMIT}",
        ])
        native.filter_embodiment(cfg, "so100")
        if dict(cfg.model.model_arch.AT_CONFIG.parts_meta) != CODEC_PARTS:
            raise ValueError("Unexpected ActionCodec group dimensions")
        if int(cfg.data.action_size) != HORIZON:
            raise ValueError("Unexpected SO101 action horizon")
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        # Stage/deserialise weights on CPU, cast there, then move the complete
        # model onto CUDA. No model inference happens during CPU staging.
        self.policy, self.processor = native.setup(cfg, device="cpu")
        self.policy.to("cuda").eval()
        # Native setup moves this non-registered tokenizer separately.
        self.policy.action_tokenizer.to("cuda")
        if any(p.device.type != "cuda" for p in self.policy.parameters()):
            raise RuntimeError("Every model parameter must reside on CUDA")
        self.parameter_dtypes = dict(Counter(str(p.dtype) for p in self.policy.parameters()))
        if "torch.bfloat16" not in self.parameter_dtypes:
            raise RuntimeError("BF16 policy materialization was not observed")
        self._bound_ar_generation()
        self.inferencer = PolicyInferencer(self.policy, self.processor, device="cuda")
        self.native = native
        torch.cuda.synchronize()
        self.load_s = time.monotonic() - started

    def _bound_ar_generation(self):
        helper = self.policy.model.ar_helper
        original = helper.infer

        def bounded(*args, **kwargs):
            remaining = TOKEN_LIMIT - self.tokens_generated
            if remaining <= 0:
                raise TimeoutError("Autoregressive token budget exhausted")
            requested = kwargs.get("max_new_tokens") or helper.max_new_tokens
            limit = min(int(requested), remaining)
            kwargs["max_new_tokens"] = limit
            result = original(*args, **kwargs)
            ids = result["generated_ids"]
            generated = int(ids.shape[-1])
            self.tokens_generated += generated
            model = args[0] if args else kwargs["model"]
            stops = set(kwargs.get("stop_token_ids") or [])
            stops.update(x for x in (getattr(model.cfg, "eos_token_id", None),
                                     getattr(helper, "eov_token_id", None)) if x is not None)
            if self.tokens_generated > TOKEN_LIMIT or (
                generated >= limit and generated and int(ids[0, -1]) not in stops
            ):
                raise TimeoutError("Incomplete autoregressive output at token limit")
            return result
        helper.infer = bounded

    def health(self):
        import torch
        return {
            "profile": "g05-so101", "checkpoint_revision": CHECKPOINT_REVISION,
            "source_revision": SOURCE_REVISION, "status": "failed" if self.fault else "ready",
            "fault": self.fault, "gpu": torch.cuda.get_device_name(),
            "load_s": self.load_s, "parameter_dtypes": self.parameter_dtypes,
            "allocated_mib": torch.cuda.memory_allocated() / 2**20,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "action_steps": HORIZON, "max_new_tokens": TOKEN_LIMIT,
            "attention": "sdpa", "compile": False, "physical_ready": False,
        }

    def infer(self, raw):
        import torch
        with self.lock:
            if self.fault:
                raise RuntimeError("Model session failed; restart required")
            try:
                if not isinstance(raw, dict) or set(raw) - {
                    "images", "state", "task", "embodiment_type", "frequency", "_dume_seed"
                }:
                    raise ValueError("Unexpected observation fields")
                if raw.get("embodiment_type") != "so100":
                    raise ValueError("Only the pinned SO101 embodiment is served")
                if set(raw.get("images", {})) != {"exterior", "wrist_left", "wrist_right"}:
                    raise ValueError("Expected the three explicit SO101 camera slots")
                for image in raw["images"].values():
                    if not isinstance(image, np.ndarray) or image.shape != (3, 480, 640) or image.dtype != np.uint8:
                        raise ValueError("Images must be CHW RGB uint8 480x640")
                if set(raw.get("state", {})) != {"right_arm"}:
                    raise ValueError("Expected only right_arm state")
                state = np.asarray(raw["state"]["right_arm"])
                if state.shape != (6,) or state.dtype.kind not in "if" or not np.isfinite(state).all():
                    raise ValueError("Invalid SO101 state")
                task = raw.get("task")
                if not isinstance(task, str) or not task.strip() or len(task) > 1024:
                    raise ValueError("Invalid task")
                seed = raw.pop("_dume_seed", 0)
                if type(seed) is not int or not 0 <= seed < 2**32:
                    raise ValueError("Invalid seed")
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                self.tokens_generated = 0
                observation = self.native.build_obs_dict(raw, self.processor)
                actions, timings = self.inferencer.infer_with_timing([observation])
                action = actions[0]
                cot = action.pop("_cot_text", None)
                absent = action.pop("_absent_keys", set())
                validate_codec_presence(absent)
                if set(action) != {"right_arm"}:
                    raise ValueError(f"Incomplete/extra action groups: {set(action)}, absent={absent}")
                chunk = action["right_arm"]
                if isinstance(chunk, torch.Tensor):
                    chunk = chunk.detach().float().cpu().numpy()
                chunk = np.asarray(chunk, dtype=np.float32)
                if chunk.shape != (1, HORIZON, 6) or not np.isfinite(chunk).all():
                    raise ValueError(f"Invalid complete chunk: {chunk.shape}")
                return chunk[0], {"timings": timings, "health": self.health(),
                                  "cot_text": cot, "generated_tokens": self.tokens_generated,
                                  "seed": seed}
            except (torch.cuda.OutOfMemoryError, TimeoutError) as exc:
                self.fault = f"{type(exc).__name__}: {exc}"
                raise


async def serve(runtime, port):
    import websockets
    lease = asyncio.Lock()

    async def handler(ws):
        if lease.locked():
            await ws.send(packb({"error": {"code": 409, "message": "One policy client at a time"}}))
            return
        async with lease:
            await ws.send(packb({"action_steps": HORIZON, "health": runtime.health()}))
            chunk, index, metadata = None, 0, None
            async for message in ws:
                try:
                    raw = unpackb(message)
                    if raw == {"__reset__": True}:
                        chunk, index, metadata = None, 0, None
                        await ws.send(packb({"__reset__": True}))
                        continue
                    if raw == {"__health__": True}:
                        await ws.send(packb(runtime.health()))
                        continue
                    recompute = chunk is None or index >= HORIZON
                    if recompute:
                        chunk, metadata = await asyncio.to_thread(runtime.infer, raw)
                        index = 0
                    elif raw != {}:
                        raise ValueError("Reset before sending a new observation within a cached chunk")
                    reply = {"action": {"right_arm": chunk[index]}, "need_obs": index == HORIZON - 1}
                    if recompute:
                        reply.update(metadata)
                        print(json.dumps({"event": "chunk", **metadata}), flush=True)
                    index += 1
                    await ws.send(packb(reply))
                except Exception as exc:
                    chunk, index, metadata = None, 0, None
                    traceback.print_exc()
                    await ws.send(packb({"error": {"code": 400 if isinstance(exc, ValueError) else 500,
                                                   "message": f"{type(exc).__name__}: {exc}"}}))
    async with websockets.serve(handler, "127.0.0.1", port, max_size=MAX_MESSAGE):
        await asyncio.Future()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", default="/checkpoints")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--evidence-dir", type=Path, default=Path("/evidence"))
    args = parser.parse_args()
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    try:
        runtime = Runtime(args.checkpoint_root)
        (args.evidence_dir / "startup.json").write_text(json.dumps(runtime.health(), indent=2))
        print(json.dumps(runtime.health()), flush=True)
        asyncio.run(serve(runtime, args.port))
    except Exception as exc:
        (args.evidence_dir / "startup-failure.json").write_text(json.dumps(
            {"error": f"{type(exc).__name__}: {exc}", "physical_ready": False}, indent=2))
        raise


if __name__ == "__main__":
    main()
