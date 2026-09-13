# Pi0.5 SO101 configuration

The mapped profile is `pi05-so101`. It uses Project-IRA's existing SO101-fine-tuned
checkpoint; no Dum-E fine-tuning was performed. The original `pi05-base` profile
remains an unmapped evaluation profile and must not command the arm.

## Pinned artifacts

- Repository: `Project-IRA/TPSoSe2026_Pi05_LeRobot_SO101_Finetuning_V7_Full_V2`
- Revision: `4b48932cc74a61f685841a4fff467ef31caa9ce1`
- Author-recommended checkpoint: `008000`
- Subdirectory: `outputs_V8/train/pi05_6gpu_fsdp_V2/checkpoints/008000/pretrained_model`
- Weight SHA-256: `7f8056e0cebbd8767a93c0f51bf5e20751baad455cf4b9965c07efca2df0fdef`
- All six runtime artifact sizes/hashes: `policy_lab/pi05-so101-manifest.json`

The runtime verifies the weight, config and processor hashes before loading.
Weights load strictly with `PI05Policy`; loading failures cannot fall back to
random initialization. Inference uses the existing GPU runtime, BF16 with its
required FP32 components, compilation off and ten denoising steps.

The weight file is 16.57 GB, despite the model card's approximate 13 GB estimate.
Only this checkpoint's inference artifacts were downloaded, excluding optimizer
state and all other training checkpoints. Local free space was insufficient, so
weights reside on the already-mounted EC2 EFS volume; neither disk was expanded.

## State, action and camera contract

| Dum-E input/output | Checkpoint convention |
|---|---|
| `shoulder_pan.pos` | Absolute centered degrees, joint 1 |
| `shoulder_lift.pos` | Absolute centered degrees, joint 2 |
| `elbow_flex.pos` | Absolute centered degrees, joint 3 |
| `wrist_flex.pos` | Absolute centered degrees, joint 4 |
| `wrist_roll.pos` | Absolute centered degrees, joint 5 |
| `gripper.pos` | Absolute 0–100 points, joint 6 |
| Live `front`, RGB 640×480 | `observation.images.desk_view` |
| Live `wrist`, RGB 640×480 | `observation.images.wrist_left` |

There are no axis flips, 90-degree offsets, degree-to-percent conversion, or
synthetic third camera for this checkpoint. The author's recording script leaves
`use_degrees` at its default. Their locked LeRobot 0.5.1 wheel explicitly defaults
to `use_degrees=True`; its follower assigns the gripper `RANGE_0_100`.
This differs from assuming older LeRobot defaults or borrowing the G0.5/Molmo
frame transform. Dum-E's actual loaded calibration remains unchanged.

The saved processor performs STATE/ACTION `MEAN_STD` normalization and inverse
normalization using the checkpoint's own six-element statistics. The client
supplies degrees and receives degrees after postprocessing. It does not apply
these statistics a second time. Relative actions are disabled.

The checkpoint explicitly declares six action features in the order above.
LeRobot's policy decoder uses that trained output dimension, yielding **50×6**
absolute targets per chunk. Its internal padded 32-dimensional representation
does not make the base checkpoint's first six values a valid substitute.

The training desk camera was 800×600, but the author's inference scripts use
640×480 for both cameras; the model resizes/pads to 224×224 internally. Our two
existing streams match that published inference setup in resolution and role.
Physical camera geometry, calibration endpoints and scene/task distribution are
still specific to each robot; this is not evidence of task accuracy.

## Selection and bounded trial

The shared factory accepts `DUME_POLICY_BACKEND=pi05-so101`. Supply the loopback
tunnel host/port (`127.0.0.1:18081`) through the existing backend constructor or
per-session agent configuration. `DUME_PI05_POLICY_PORT` sets the default when a
port is not passed. No global `my-dum-e.yaml` values were changed.

The implemented bridge is synchronous and explicitly rejects
`DUME_ASYNC_INFERENCE=1`; the author's async deployment is not a claim that this
bridge implements their scheduler.

```bash
MPLCONFIGDIR=/tmp/dume-matplotlib .venv/bin/python scripts/check_live_so101_policy.py \
  --workspace corpus/a-new-pi05-so101-input-check

MPLCONFIGDIR=/tmp/dume-matplotlib .venv/bin/python scripts/run_integration_trial.py \
  prepare --profile pi05-so101 --workspace corpus/a-new-pi05-so101-trial
```

The first physical trial is prepared for **three chunks / 150 commands**, with
0.25-degree command steps, 3.75-degree target lead and a shared 5-degree total
excursion cap. Gripper limits use points on its 0–100 scale. Existing calibration,
camera, server-health, stop and torque-hold checks remain active. A prepared
snapshot alone does not authorize motor execution.

## Deployment and sources

ECR image: `177118830501.dkr.ecr.us-west-2.amazonaws.com/dume/model-swap:phase10-pi05-so101-v1`

Digest: `sha256:6a1ff7e22a1c4202a68f7ccfd85d6559dcd7d51ea08433461956d7f9d68d34fb`

The server is staged on the existing L4 workstation, with weights mounted
read-only from EFS and no arm/camera device mounts. The local client reaches it
through SSM forwarding on loopback port 18081.

Primary sources, retrieved and pinned on 2026-09-13 UTC:

- [Author model card and recommended checkpoint](https://huggingface.co/Project-IRA/TPSoSe2026_Pi05_LeRobot_SO101_Finetuning_V7_Full_V2/blob/4b48932cc74a61f685841a4fff467ef31caa9ce1/README.md)
- [Checkpoint config and processors](https://huggingface.co/Project-IRA/TPSoSe2026_Pi05_LeRobot_SO101_Finetuning_V7_Full_V2/tree/4b48932cc74a61f685841a4fff467ef31caa9ce1/outputs_V8/train/pi05_6gpu_fsdp_V2/checkpoints/008000/pretrained_model)
- [Author recording/inference reference](https://github.com/Project-IRA/interactive-robotic-arm/tree/a339b00b09b8438f6feedcc3a3e42bb299a44034)
- The reference `uv.lock` pins LeRobot 0.5.1 wheel SHA-256
  `bbd11021023fde0947b6d1ff1c52fe91c86a28ab09a96359892f3ef7e8866862`.
  That exact wheel was read, not installed, to verify follower units.

The author's model card identifies the checkpoint license as CC BY-SA 4.0.
Research captures, source hashes and normalization inspection are retained in
`corpus/phase10-integration-20260913/pi05-so101-research/`.

## Validation result

54 scoped software tests pass. Three fresh live front/wrist/state checks returned
finite 50×6 outputs without motor dispatch. Warm GPU generation was 327–330 ms;
workstation-to-EC2 RPC was 3.49–3.51 seconds (first chunk: 4.50 seconds).
Peak PyTorch allocation was 9044.50 MiB. Evidence lives in
`corpus/phase10-integration-20260913/pi05-so101-live-mapped-check/`.
The bounded physical trial is prepared but has not run.

### Pi0.5 SO101 first physical trial (2026-09-13 UTC)

Aaron explicitly confirmed presence/workspace readiness and authorized the
prepared first physical run. Trial 11 completed three chunks / 150 mapped
commands with actual front/wrist images and state; no software errors or
controller clamp warnings occurred. Deliberate target projection is recorded.
Evidence: `corpus/phase10-integration-20260913/trial-11-pi05-so101-candidate/`.
Chunk RPC times (ms): 3350.61, 3389.46, 3387.36.
GPU generation times (ms): 326.65, 324.50, 329.71.
Maximum measured displacement per joint: [4.571428571428571, 4.219780219780219, 2.197802197802204, 3.1648351648351536, 1.5824175824175768, 4.735758407687028].
Within-chunk command interval median/max (ms): 50.071626999852015 / 100.36749099890585.
Operator movement/smoothness/safe-hold observation is pending. No further
physical run started. No task-accuracy or async claim is made.

### Pi0.5 physical-trial operator observation

Aaron confirmed: “Yes it was visible.” Visible policy-driven movement is confirmed
for trial 11, alongside live camera/state inference and mapped bounded dispatch.
Smoothness and safe hold were not separately stated, so those are not recorded
as explicitly confirmed. Task accuracy is unscored. No additional run started.
