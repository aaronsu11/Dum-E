# Add another embodiment

The existing end-to-end mappings are SO101-specific. **Galaxea G0.5 model support is not Galaxea R1 Pro robot support.** R1 Pro is a future integration target here, not a validated configuration. Obtain its exact hardware/SDK and checkpoint schema before defining sensors, joints or action groups; this guide does not assume those specifications.

The extension points are deliberately separate:

| Responsibility | Interface / current example | What a new embodiment supplies |
|---|---|---|
| Robot IO and ownership | `shared.IRobotController`; `embodiment/so_arm10x/controller.py` | Connect/read/command/disconnect; actual calibration, limits, units and resource ownership |
| Policy transport | `shared.IPolicyBackend`; `policy/factory.py` | Explicit backend selection, health, task instruction, reset/close and complete validated chunks |
| Observation/action mapping | `policy/so101_contract.py`, `policy/galaxea/modalities.py` | Named sensor roles, frame conventions and reversible model↔robot conversions |
| GPU model runtime | `policy_lab/profiles.py`, `runtime.py`, `protocol.py`; `docker/galaxea-policy/` | Pinned checkpoint, processor, native horizon/action groups, request schema and model identity |
| Execution lifecycle | SO101 `agent.py`, `skills.py`, `safety.py` | Suitable skills, reset poses, serial/SDK ownership, stop/hold behavior and failure publication |
| Scheduling | `policy/lerobot/async_chunks.py`, `policy_guard/rtc_trial.py` | Explicit supported timing/chunk contract; model-specific prefix semantics if applicable |

`policy_lab` is a maintained inference-server package despite its historical name. The HTTP bridge's shared class is `SO101HTTPPolicyBackend`: its transport is shared across current models, but its six-joint/camera contract remains explicit. Do not subclass it for a robot with different action groups and merely change the model name.

## Implementation order

1. Write a small contract table: exact checkpoint revision/hash; input cameras and timestamp/shape/color conventions; state names, order and units; output groups, absolute/relative semantics, normalization owner; native versus executed horizon; supported scheduler. Record missing modalities explicitly. Reject absent required sensors instead of fabricating a shape-compatible input.
2. Add an embodiment module for hardware IO and mappings. Keep native SDK/robot state out of policy server processes. Parameterize addresses and calibration paths in configuration; use an explicit backend identifier rather than a silent fallback.
3. Write reversible conversion tests with asymmetric values, including signs, offsets, gripper/action groups and calibration changes. Add malformed shape, nonfinite output, missing camera and wrong checkpoint tests. Shape agreement alone cannot establish physical semantics.
4. Add a server profile and independent request schema where needed. Current HTTP `decode_request` is six-state/two-camera SO101; extend with a discriminated schema or a separate adapter rather than padding an R1 Pro observation into it. Preserve strict checkpoint loading and GPU-only inference.
5. Add an embodiment-aware offline recording loader and benchmark adapter. The current portable NPZ schema is SO101-only. Use a few recorded observations to verify preprocessing, full-chunk decode, latency and memory before touching hardware. Keep raw outputs and transformations inspectable.
6. Implement a controller-specific bounded executor: one owner, calibrated limits, finite complete chunks, no catch-up bursts, stale-reply rejection, current operator readiness, safe stop and explicit recovery. SO101's Feetech source audit and hold behavior do not transfer to another SDK.
7. Qualify synchronous execution first. Async requires an execution period, measured request deadline and buffer policy that fit that embodiment. Pi0.5 RTC additionally requires guidance in the checkpoint's normalized action space corresponding to the commands actually queued. No automatic async capability is inherited from transport or model branding.
8. Register skills with the agent/factory only after their starting poses, action budget and cancellation behavior are explicit. Keep existing MCP/voice task lifecycle interfaces, then test simulated failure/retarget/cancel and finally one attended physical trial.

A minimum extension should add a small contract/config, mapping adapter, controller/skill integration and focused tests. It should reuse transport, evidence IO and task lifecycle code where semantics agree. Avoid copying the archived phase harnesses or adding an entire new validation framework. Accuracy and fine-tuning can be separate milestones after plumbing and bounded execution are demonstrated.
