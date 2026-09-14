"""Bounded Pi0.5 RTC queue and attended-trial loop; no implicit retargeting."""
from concurrent.futures import ThreadPoolExecutor
import json
import secrets
import time

import numpy as np
from embodiment.so_arm10x.mappings.galaxea import JOINTS
from embodiment.so_arm10x.validation import bounded_command, check_camera


from policy.execution.rtc import BoundedRTCQueue, accept_reply


def run_rtc_trial(controller, policy, stop, workspace, result, origin, limits,
                  check_inputs, *, chunks=3, period_s=0.05, deadline_s=0.75):
    """Three generated chunks produce 100 played targets with 25-step overlap."""
    queue = BoundedRTCQueue(origin, limits, bound_command=bounded_command)
    epoch = secrets.randbits(52)
    instruction = policy.language_instruction
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pi05-rtc")
    pending = None
    requested = 0
    consumed = 0
    budget = 50 + (chunks - 1) * 25
    result["rtc"] = {"epoch": epoch, "deadline_s": deadline_s, "delay_steps": 15,
                     "request_remaining": 25, "action_budget": budget, "handoffs": []}

    def checks():
        stop.check()
        check_inputs()
        if policy.language_instruction != instruction:
            raise RuntimeError("RTC instruction changed; restart trial with fresh epoch")
        if not policy.rtc_ping():
            raise RuntimeError("RTC policy unavailable; hold")
        stop.check()

    def observation():
        current = controller.get_observation()
        for role in ("front", "wrist"):
            check_camera(current[role], role)
        stop.check()
        return current, np.array([current[k] for k in JOINTS], dtype=float)

    def submit(current, prefix):
        nonlocal requested
        requested += 1
        directory = workspace / f"chunk-{requested}"
        directory.mkdir()
        frozen = {key: (value.copy() if isinstance(value, np.ndarray) else value)
                  for key, value in current.items()}
        np.savez(directory / "live-observation.npz",
                            state=[current[k] for k in JOINTS],
                            front=frozen["front"], wrist=frozen["wrist"])
        if prefix is not None:
            np.save(directory / "queued-prefix-degrees.npy", prefix)
        started = time.monotonic()
        future = executor.submit(
            policy.get_rtc_action, frozen, prefix, epoch=epoch,
            request_id=requested, delay_steps=0 if prefix is None else 15)
        return {"future": future, "started": started, "deadline": deadline_s,
                "epoch": epoch, "consumed": consumed, "index": requested - 1,
                "directory": directory}

    def install(current_observed):
        nonlocal pending
        elapsed = consumed - pending["consumed"]
        accept_reply(pending, time.monotonic(), epoch, elapsed)
        raw, metadata = pending["future"].result()
        stop.check()
        queue.install(raw, current_observed, elapsed, pending["index"])
        (pending["directory"] / "raw-actions.json").write_text(json.dumps(raw.tolist()))
        # Projection/evidence I/O is part of installation, not exempt from expiry.
        accept_reply(pending, time.monotonic(), epoch, elapsed)
        stop.check()
        result["chunks"].append({
            "index": pending["index"], "model_metadata": metadata,
            "chunk_rpc_ms": metadata.get("client_rpc_ms"),
            "request_to_install_ms": 1000 * (time.monotonic() - pending["started"]),
            "discarded_elapsed_steps": elapsed,
        })
        result["rtc"]["handoffs"].append({"at_action": consumed, "elapsed": elapsed})
        pending = None

    try:
        checks()
        current, observed = observation()
        pending = submit(current, None)
        while not pending["future"].done():
            checks()
            accept_reply(pending, time.monotonic(), epoch, 0)
            time.sleep(0.01)
        checks()
        current, observed = observation()
        install(observed)
        while consumed < budget:
            tick = time.monotonic()
            checks()
            current, observed = observation()
            if pending is not None:
                accept_reply(pending, time.monotonic(), epoch, consumed - pending["consumed"])
                if pending["future"].done():
                    install(observed)
            if pending is None and len(queue.items) == 25 and requested < chunks:
                pending = submit(current, queue.prefix())
            stop.check()
            check_inputs()
            if pending is not None:
                accept_reply(pending, time.monotonic(), epoch, consumed - pending["consumed"])
            item = queue.consume(observed)
            sent = controller.set_target_state(dict(zip(JOINTS, map(float, item["command"]))))
            stop.check()
            if (not isinstance(sent, dict) or set(sent) != set(JOINTS)
                    or not np.allclose([sent[k] for k in JOINTS], item["command"],
                                       rtol=0, atol=1e-8)):
                raise RuntimeError("Controller changed queued RTC target; hold")
            result["motor_targets_sent"] = True
            result["actions"].append({
                "index": consumed, "time": tick, "observed": observed.tolist(),
                "raw": item["raw"].tolist(), "bounded": item["command"].tolist(),
                "sent": sent, "chunk_index": item["chunk_index"],
                "action_index": item["action_index"],
                "limited": bool(np.any(item["raw"] != item["command"])),
            })
            consumed += 1
            time.sleep(max(0, period_s - (time.monotonic() - tick)))
        if requested != chunks or pending is not None:
            raise RuntimeError("RTC trial ended with incomplete inference")
    finally:
        # A blocked/late worker cannot access hardware or install a result later.
        if pending is not None:
            pending["future"].cancel()
        executor.shutdown(wait=False, cancel_futures=True)
