# GR00T migration invariants

The LeRobot server loads the unconverted Dum-E GR00T N1.7 SO101 checkpoint. Four fixes must survive future upgrades:

1. **Load BF16 before materialization.** `DumEGrootPolicy` passes `dtype=bfloat16`, `load_bf16=True` and disables the upstream FP32 recast. Casting after load is too late for the validated 12 GiB GPU. Preserve the unused language-head tying behavior.
2. **Specify the actual embodiment and features.** `new_embodiment` decodes 16 steps, despite the generic 40-step config. Camera features are RGB 480×640 and six state/action values. Placeholder 224×224 feature geometry silently changes preprocessing.
3. **Decode a whole relative-action chunk once.** Preprocessing caches the observation anchor used by postprocessing. Decode `(B,T,D)` before converting to timed actions; per-timestep decoding is wrong. Keep the entire pipeline under one exchange lock.
4. **Preserve the serving image recipe.** Force letterbox padding before the checkpoint's crop/resize path. A deterministic 480×640 input produces 256×256 RGB with native-matching historical bytes; the unpatched recipe produces 256×340. A square placeholder resize can have the same final shape and still different content.

`policy/backends/lerobot/models/groot.py` is shared by preflight and post-load validation. Refusal clears policy and both processors; an aborted setup must not leave a usable rejected model. Check the actual decoder, embodiment statistics and singular checkpoint `use_relative_action`; the generic config's identity `normalization_mapping` is normal for this processor and is not by itself a failure.

The native implementation is pinned to `23ace64f17aa5015259b8609d371eb61a357c776`; LeRobot to 0.6.1. Cosmos `9ce19a195e423419c349abfc86fd07178b230561` is a forward reproducibility pin, not recovered training provenance. The cache pins `refs/main` because upstream processor loading cannot accept a revision. `policy/backends/isaac_groot/cache.py` preserves native offline cache resolution without modifying checkpoint blobs.

Native and LeRobot deployed BF16 configurations differ, including attention/kernel paths. The earlier 12-observation, one-seed comparison was accepted as an integration milestone, not a numerical identity claim. A normalized-joint-point difference is converted to degrees using that joint's calibration span; it is not millimeters of end-effector motion. Physical displacement additionally requires kinematics and pose.

Normal observation modes are `off` and `lightweight`, with shared hook cleanup and generation timing. Exhaustive operation interception and approval/attestation workflows were archived because they belonged to migration experiments. `DUME_PARITY_ATTESTATION_PATH` is now explicitly rejected by the normal telemetry launcher; remove that old setting. Historical source and findings remain on the archive branch named in the validation guide.
