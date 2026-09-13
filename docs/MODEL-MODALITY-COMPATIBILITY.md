# SO101 model compatibility — Phase 9

This is a software-contract audit, not physical approval. No robot or camera was
opened during this phase. The four-model physical session remains pending.

## Local arm

`my-dum-e.yaml` selects `so101_follower`, robot ID
`my_awesome_follower_arm`, serial `/dev/ttyACM0`, wrist camera index 0 and front
camera index 2. The controller captures RGB 640×480 images. Five body joints use
centered LeRobot degrees; the gripper uses RANGE_0_100, even when
`use_degrees=True`. Do not convert the gripper as degrees.

Calibration file:
`~/.cache/huggingface/lerobot/calibration/robots/so101_follower/my_awesome_follower_arm.json`

SHA256: `5bd471fbbb4e1be0c6ede80472d365b527bc0808befdd67f123148ed49e3dc50`

At audit time no `/dev/video*` or `/dev/ttyACM*` devices were visible. Saved camera
indices are configuration, not proof of current identity, orientation or exposure.
Before physical trials, verify front/wrist identity using live images and verify
calibration and signs at a known stationary pose.

## Model contracts

Joint order throughout is shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
wrist_roll, gripper. Wire dictionaries retain `.pos` names.

| Model | Images | Model state/action frame | Current status |
|---|---|---|---|
| GR00T Dum-E SO101 | front + wrist | Training statistics indicate five RANGE_M100_100 joints; gripper 0–100 | Existing physically validated serving configuration retained. Its documented degrees/normalized mismatch is not silently changed. |
| Pi0.5 base | Two image slots plus a placeholder; padded 32D state/action | No verified SO101 normalization or six-joint action mapping | GPU/transport smoke only. Needs a suitable SO101 checkpoint and its processor before a physical trial. |
| MolmoAct2 SO100_101 | front then wrist, RGB | Legacy SO100/101 degree frame, absolute joint targets, gripper normalized | Reference conversion prepared; Dum-E camera placement, calibration and behavior still require verification. Phase8.1 HTTP benchmark is not a production controller adapter. |
| G0.5 SO101 | front→exterior, wrist→wrist_right; unused wrist_left explicitly black | Legacy SO100/101 degree frame, six values under `right_arm`; native postprocessor returns absolute targets | Native backend implemented and GPU checked. Physical calibration/sign/orientation and starting pose remain unverified. |

### G0.5 and Molmo reference frame

From current arm frame to model frame:

```text
model = arm * [1, -1, 1, 1, 1, 1] + [0, 90, 90, 0, 0, 0]
arm   = (model - [0, 90, 90, 0, 0, 0]) * [1, -1, 1, 1, 1, 1]
```

Evidence:

- GalaxeaVLA revision `89f2322b4ad016e192437adc1a2c253b05bab246`,
  `experiments/so100/so100_policy_client.py`.
- MolmoAct2 revision `66b87e64efd99dfd103241418113955cf64dfa9c` links the
  SO101 reference implementation by Irene Grace. Its revision
  `9819f6a9ea3ab71382f2f5d753f5de016b62a9e6`,
  `molmoact_so101/setup/frame_transforms.py` and `inference.py`, supplies the
  same conversion and current LeRobot degrees configuration.

`policy/so101_contract.py` provides explicit forward/inverse conversions.
G0.5's adapter uses its named conversion directly. The shared utility does not
automatically alter existing GR00T or Molmo serving behavior.

### GR00T normalized frame

See `docs/UNITS-VERDICT.md` for the existing measured discrepancy. With this
calibration, degrees per normalized point are:

| Joint | Degrees per point |
|---|---:|
| shoulder_pan | 1.165275 |
| shoulder_lift | 1.037363 |
| elbow_flex | 0.963516 |
| wrist_flex | 1.007912 |
| wrist_roll | 1.677802 |

The conversion is `degrees / scale` toward the normalized model frame and
`normalized * scale` back. It leaves gripper values unchanged. This follows
LeRobot's centered endpoint convention; it is not an estimate from output
statistics. The helper requires an explicit calibration file. Changing the
validated GR00T controller path requires its own bounded regression trial.

### G0.5 codec and preprocessing

The pinned checkpoint has **20 codec dimensions**, arranged as 9+1+9+1, not the
27 dimensions assumed in the early roadmap. SO101's six values, including its
gripper, occupy the first six values of `right_control`. The remaining three
control values are padding. `left_control`, `left_gripper` and `right_gripper`
are virtual/absent. The native inverse processor returns exactly `right_arm`
with shape `(1,32,6)`. Runtime validation checks that exact presence pattern
and rejects a missing right-control group or any extra returned arm group.

There is no manual five-joint/gripper split in the client. Native processors
handle relative action normalization, padding, codec decoding and conversion
back to absolute targets. All six physical outputs are retained.

Although the dataset subsection says 224×224, the effective model processor
overrides all three camera slots to **256×256**, as observed during startup.
The client supplies CHW RGB uint8 480×640. A missing front or wrist image is an
error; only the unused wrist_left slot is padded.

## Limits of the recorded-observation checks

The frozen corpus comes from `aaronsu11/so101_fruit`, dataset revision
`872af48c8442e6c4e8ef968691670dc390c03458`, codebase format v2.1.
The prior three-model checks fed its raw numeric state through their evaluation
interfaces. The G0.5 check exercises its real controller backend, including the
current-arm-to-legacy conversion. Interpreting those frozen numeric values as
current arm degrees is a **transport/shape test input**, not proof that the
recording's original calibration matches the current arm.

Consequently, these measurements establish GPU fit, complete finite action
chunks and latency, not task success, action equivalence across models or
physical suitability. Exact repeatability between client locations is useful
for transport integrity; cross-model action differences have no parity meaning.

## Before the four-model physical session

1. Reconnect the arm and both cameras; check current identity and image orientation.
2. Confirm the calibration hash and each joint's stationary pose/sign. Retain the
   controller's degrees convention and its existing safety limits.
3. Give each checkpoint an appropriate initial pose. The GR00T ready pose is
   not automatically suitable for G0.5 or Molmo.
4. Complete the Molmo controller transport/action adapter and select a verified
   SO101 Pi0.5 checkpoint/processor. Base-model padding does not satisfy this.
5. Test one model and one bounded trial at a time with Aaron beside the arm.

The roughly 30-second dispatch/startup optimization from Phase8 remains a
separate backlog item. Phase9 does not claim to remove cold-start latency.
