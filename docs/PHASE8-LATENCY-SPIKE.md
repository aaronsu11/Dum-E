# Phase 8 latency spike investigation

The integrated physical trial stopped correctly when a GPU chunk took 322.126 ms, exceeding the configured 250.098 ms request deadline. It completed 1,568 actions over 78.617 s. No old-epoch action followed retarget and no action followed the stop. Aaron confirmed apple redirection, safe stop, and audible failure.

## Evidence and bounded diagnostic

The recorded physical run had 199 server telemetry entries, including setup/warmup. Excluding the first two, median generation was 163.864 ms; the next-largest value was 185.065 ms, followed by the isolated 322.126 ms outlier. The slow interval is inside the server's synchronized prediction timing, before its final telemetry log. Network round-trip overhead alone cannot explain it. This timing includes preprocessing, model execution, decoding and host scheduling; it does not identify an individual CUDA kernel or prove GPU contention.

A separate frozen-observation GPU-only diagnostic performed two warmups plus 40 back-to-back and 40 requests paced at 400 ms. No controller or camera was instantiated. Results:

| Requests | Median RPC | Maximum RPC | Above 250.098 ms |
| --- | ---: | ---: | ---: |
| Back-to-back | 175.348 ms | 179.855 ms | 0/40 |
| 400 ms paced | 165.151 ms | 179.176 ms | 0/40 |

The original configuration used a 150.098 ms empirical p99 from 100 sequential calls plus 100 ms margin. That small sample was not a tail guarantee. The new diagnostic does not replace its calibration or establish a new safe deadline. It used a frozen banana observation/instruction, sequential blocks, and GPU clock polling; it did not recreate live cameras, apple state, motion, or spoken interaction. GPU process inspection after the sample showed only the inference Python compute process. This cannot establish what shared the GPU at the original failure.

## Conclusion and next action

A real server-side latency outlier caused the stop; its underlying cause remains unresolved. Pacing alone did not reproduce it. Do not change the watchdog or infer that the sustained-run acceptance passed. No production code or safety settings changed.

Before the next physical attempt, capture a small paced replay with stage-level timing and concurrent host/GPU utilization under the live voice workload. Distinguish preprocessing/host stalls from model execution before choosing mitigation. Any revised deadline must still fit the 400 ms overlap and retain the bounded stop behavior; it needs measured justification, not just a value larger than this outlier. Startup optimization remains deferred.

Artifacts: `corpus/phase8-latency-diagnosis-20260912/{summary.json,rpc-samples.json,gpu-samples.json,gpu-server.log,reproduce.py}`; original evidence in `corpus/phase8-integrated-closeout-trial1-20260912/`.

## Disposition — 2026-09-12

Aaron requested backlogging this investigation and retaining the notes while completing the remaining Phase8 closeout work. The latency investigation and interrupted sustained-run acceptance are deferred, not passed. Track them in `.planning/todos/pending/async-latency-tail-and-sustained-voice.md`. No safety deadline change or further physical run is implied.
