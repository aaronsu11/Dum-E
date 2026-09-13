"""Small GPU-only Pi0.5 SO101 RTC check using recorded inputs; no robot access."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from policy_lab.runtime import ModelRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    report = {"status": "failed", "motor_commands_sent": 0, "runs": [],
              "input_mode": "recorded trial-11 observations",
              "lerobot_version": importlib.metadata.version("lerobot"),
              "execution_horizon": 25, "period_s": 0.05,
              "cpu_threads": torch.get_num_threads(),
              "seed": 20265907, "warmups_excluded": True}
    try:
        runtime = ModelRuntime("pi05-so101")
        policy = runtime.policy
        from lerobot.policies.rtc.configuration_rtc import RTCConfig

        if not policy.supports_rtc():
            raise RuntimeError("Installed policy does not support RTC")
        report["profile"] = runtime.profile.to_dict()
        report["load_s"] = runtime.load_s
        report["parameter_dtypes"] = runtime.parameter_dtypes
        report["num_inference_steps"] = policy.config.num_inference_steps
        report["compile_model"] = policy.config.compile_model
        report["device"] = torch.cuda.get_device_name()
        report["source_sha256"] = {
            "check_pi05_rtc.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        }
        report["input_sha256"] = {
            f"chunk-{index}": hashlib.sha256(
                (args.recording / f"chunk-{index}" / "live-observation.npz").read_bytes()
            ).hexdigest() for index in range(1, 4)
        }
        import lerobot.policies.pi05.modeling_pi05 as pi05_module
        import lerobot.policies.rtc.modeling_rtc as rtc_module
        for module in (pi05_module, rtc_module):
            report["source_sha256"][module.__name__] = hashlib.sha256(
                Path(module.__file__).read_bytes()).hexdigest()

        def configure(enabled):
            policy.config.rtc_config = RTCConfig(
                enabled=enabled, execution_horizon=25, max_guidance_weight=10.0)
            policy.init_rtc_processor()
            policy.reset()

        def generate(index, previous=None, delay=0):
            path = args.recording / f"chunk-{index}" / "live-observation.npz"
            with np.load(path, allow_pickle=False) as observation:
                def image(key):
                    return torch.from_numpy(observation[key].copy()).permute(2, 0, 1).float() / 255
                raw_batch = {
                    "observation.state": torch.from_numpy(observation["state"].astype(np.float32)),
                    "observation.images.desk_view": image("front"),
                    "observation.images.wrist_left": image("wrist"),
                    "task": "Grab a banana and put it on the plate",
                }
            torch.manual_seed(report["seed"])
            torch.cuda.manual_seed_all(report["seed"])
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            # RTC locally enables autograd for guidance: inference_mode is invalid.
            with torch.no_grad():
                batch = runtime.pre(raw_batch)
                torch.cuda.synchronize()
                predicted = time.perf_counter()
                kwargs = {} if previous is None else {
                    "prev_chunk_left_over": previous.detach().clone(),
                    "inference_delay": delay, "execution_horizon": 25,
                }
                normalized = policy.predict_action_chunk(batch, **kwargs)
                torch.cuda.synchronize()
                decoded = time.perf_counter()
                normalized = normalized.detach().clone()
                # Preserve the normalized prefix; RTC must never receive degrees.
                physical = runtime.post(normalized.clone()).detach().float().cpu()
                torch.cuda.synchronize()
            if normalized.shape != (1, 50, 6) or not torch.isfinite(normalized).all():
                raise RuntimeError("Invalid normalized RTC chunk")
            if physical.shape != (1, 50, 6) or not torch.isfinite(physical).all():
                raise RuntimeError("Invalid decoded RTC chunk")
            return normalized, physical, {
                "generation_ms": 1000 * (decoded - predicted),
                "total_ms": 1000 * (time.perf_counter() - started),
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            }

        configure(False)
        previous, _, cold = generate(1)
        report["cold_ordinary"] = cold
        configure(True)
        _, _, warm = generate(1, previous[:, 25:], delay=20)
        report["cold_guided"] = warm
        delay = min(24, math.ceil(warm["total_ms"] / 50) + 2)
        report["inference_delay_steps"] = delay
        report["prefix_space"] = "saved processor normalized six action dimensions"
        report["rtc_config"] = {
            "schedule": "LINEAR", "max_guidance_weight": 10.0,
            "execution_horizon": 25, "debug": False,
        }
        for index in range(1, 4):
            prefix = previous[:, 25:].detach().clone()
            configure(False)
            ordinary, ordinary_physical, baseline = generate(index)
            configure(True)
            guided, guided_physical, rtc = generate(index, prefix, delay)
            baseline["prefix_rmse_normalized"] = (
                (ordinary[:, :delay] - prefix[:, :delay]).square().mean().sqrt().item())
            rtc["prefix_rmse_normalized"] = (
                (guided[:, :delay] - prefix[:, :delay]).square().mean().sqrt().item())
            entry = {"index": index, "ordinary": baseline, "rtc": rtc,
                     "rtc_under_1250ms_chunk_overlap": rtc["total_ms"] < 1250,
                     "rtc_within_assumed_delay": rtc["total_ms"] <= delay * 50}
            report["runs"].append(entry)
            np.savez(args.output / f"pair-{index}.npz",
                     prefix_normalized=prefix.cpu().numpy(),
                     ordinary_normalized=ordinary.cpu().numpy(),
                     rtc_normalized=guided.cpu().numpy(),
                     ordinary_degrees=ordinary_physical.numpy(),
                     rtc_degrees=guided_physical.numpy())
            previous = guided
            print(json.dumps(entry), flush=True)
        # Bounded real-clock handoff to a dummy consumer; no hardware access.
        consumed = []
        future = None
        active = guided_physical[0].numpy()
        active_index = 0
        handoff = None
        with ThreadPoolExecutor(max_workers=1) as worker:
            start = time.perf_counter()
            for tick in range(50):
                time.sleep(max(0, start + tick * 0.05 - time.perf_counter()))
                if tick == 25:
                    future = worker.submit(generate, 3, previous[:, 25:].clone(), delay)
                if future is not None and future.done() and handoff is None:
                    _, replacement, timing = future.result()
                    elapsed_steps = tick - 25
                    if elapsed_steps >= 25:
                        raise RuntimeError("RTC replacement missed the available prefix")
                    active = replacement[0].numpy()
                    active_index = elapsed_steps
                    handoff = {"tick": tick, "discarded_elapsed_steps": elapsed_steps,
                               "timings": timing}
                if active_index >= len(active):
                    raise RuntimeError("Dummy consumer exhausted its action buffer")
                consumed.append({"tick": tick, "at_s": time.perf_counter() - start,
                                 "action": active[active_index].tolist()})
                active_index += 1
        report["async_replay"] = {
            "consumer": "in-memory only, 50 targets at nominal 20 Hz",
            "handoff": handoff, "targets": consumed,
            "max_interval_ms": 1000 * max(
                b["at_s"] - a["at_s"] for a, b in zip(consumed, consumed[1:])),
        }
        if handoff is None:
            raise RuntimeError("No RTC handoff completed during the dummy consumer replay")
        report["status"] = "recorded_input_rtc_passed"
        report["limitations"] = [
            "Dummy async consumer only: no transport, live sensors, motors or physical validation.",
            "Three recorded observations; prefix continuity is not task accuracy.",
            "Synthetic 25-step execution advance; delay is estimated, not a validated deadline.",
        ]
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
