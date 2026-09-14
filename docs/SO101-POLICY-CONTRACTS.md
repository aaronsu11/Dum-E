# SO101 modalities and action frames

The shared named-joint order is `shoulder_pan.pos`, `shoulder_lift.pos`, `elbow_flex.pos`, `wrist_flex.pos`, `wrist_roll.pos`, `gripper.pos`. Portable benchmark recordings and the bounded runner use centered degrees for the first five joints and 0–100 points for the gripper. A gripper point is not an angle.

Both camera inputs are RGB `uint8` arrays of shape `(480,640,3)`. OpenCV capture is BGR and must be converted once. Verify front/wrist identity, orientation and exposure from live frames; saved camera indices alone do not establish identity. `check_camera` rejects nearly uniform images before a physical trial.

| Profile | Camera mapping | State/action conversion |
|---|---|---|
| G0.5 SO101 | front→exterior, wrist→wrist_right; wrist_left is the declared unused placeholder | `model = arm * [1,-1,1,1,1,1] + [0,90,90,0,0,0]`; invert on output |
| MolmoAct 2 SO100/101 | front then wrist | Same legacy degree-frame transform; saved model normalization handles gripper |
| Pi0.5 SO101 | front→desk_view, wrist→wrist_left | Centered degrees / 0–100 unchanged; saved MEAN_STD processors normalize exactly once |
| GR00T SO101 | front + wrist, preserving checkpoint-declared order | Five joints RANGE_M100_100; gripper 0–100. Calibrated HTTP bridge converts centered degrees to/from normalized points |
| Pi0.5 base | Two images and a declared zero third-camera placeholder | Six input values padded to 32 for smoke only; no verified SO101 output mapping |

`embodiment/so_arm10x/mappings/frames.py` owns explicit conversions. GR00T degrees per normalized point are `(range_max-range_min)*360/(4095*200)` per joint; gripper scale is 1. Never infer units from observed numerical magnitude. The original native/gRPC backend preserves its existing controller convention; the new HTTP bridge does not silently change that production path.

Pi0.5 checkpoint `008000` declares six named absolute actions, 50 steps, no relative actions and no empty cameras. Its padded internal representation is not permission to reinterpret the base model's first six outputs as motor targets. G0.5 drains one native 32-action cache per observation; Molmo returns 30 actions; GR00T decodes 16.

## Calibration and hardware boundary

LeRobot 0.6.1 resolves SO101 calibration under `robots/so_follower`, which can differ from older `robots/so101_follower`. Bind to the actual `controller.robot.calibration_fpath`, not an assumed filename. The runner checks the file hash, loaded follower and bus mappings, and motor IDs before and after connect; it never recalibrates during a trial. Calibration endpoint, camera placement and training scene differences still affect behavior after software mappings pass.

`embodiment/so_arm10x/safety.py` owns stop latching, audited low-level SDK dispatch and guarded cleanup. A stop prevents subsequent target/torque-enable dispatch; already transmitted packets cannot be retracted. The retained cleanup preserves hold rather than deliberately disabling torque. It is an observed bounded-trial behavior, not a guarantee for every load/pose/fault.

The reusable runner uses 0.25-degree/point command slew, 3.75-degree/point tracking allowance and a 5-degree/point total excursion envelope, intersected with calibrated limits. It saves raw predictions separately from projected/sent commands. Synthetic diagnostics remain labeled and separate from policy success. A new embodiment needs its own limits and stop implementation; these numbers must not be copied to a different robot.

See [adding an embodiment](EXTENDING-EMBODIMENTS.md) for the implementation and validation sequence.
