# Superseded Phase 10 criteria

Preserved at closeout. These are historical criteria, not a claim of completion.
Operator amendments and the disposition of retained gaps are in 10-VERIFICATION.md.

### Phase 10: Galaxea G0.5 Live on SO101

**Operator amendment (2026-09-13):** Physical trials now emphasize embodiment configuration, live input modalities and inference/execution plumbing across all four models. Grasp accuracy and fine-tuning are not acceptance gates. Preserve correct joint/unit mapping and stop behavior; test supported sync/async modes and explicitly record gaps. Plans live in10-physical-embodiment-integration. Start with one bounded G05 sync trial; wrist-camera content and current operator presence remain prerequisites.

**Async scope clarification (2026-09-13):** Aaron requires async physical validation for Pi0.5 only. Its bounded RTC trial14 passed with operator-confirmed visible smooth motion and safe hold. No additional async trials for the other models are required. The older G0.5-specific criteria below must be reconciled with the amended integration scope during closeout.

**Goal**: G0.5 drives the physical SO-ARM10x coherently toward a spoken target, with joint direction and magnitude verified in air before any task motion.
**Depends on**: Phase 9
**Requirements**: GX-05
**Success Criteria** (what must be TRUE):

  1. **Before any task motion**, each joint's direction and magnitude are confirmed **in air, one joint at a time, at low speed under clamp**, and the per-joint results recorded — Galaxea's own reference client needs `signs = [1,-1,1,1,1,1]` and `offsets = [0,90,90,0,0,0]`, hardcoded constants that must be re-verified against Dum-E's calibration rather than copied on faith. A wrong sign on `shoulder_lift` drives the arm into the table.
  2. Starting from **G0.5's own home/ready pose derived from `g05-so101` dataset statistics** (not GR00T's `[0,-102,96,76,-90,0]`, which sits ~68° off Galaxea's training-distribution centre on `shoulder_lift`), a spoken instruction produces **smooth, coherent, target-directed motion** across repeated attempts. The bar is coherent motion — **explicitly not 10/10**, since `g05-so101` is a generic released checkpoint, not Dum-E's fine-tuned weights, and holding it to the GR00T baseline would be a false gate that fails a working integration.
  3. Camera slots are verified **empirically** (`front`→`exterior`, `wrist`→`wrist_right`) so a mis-mapped camera cannot pass as a silently zero-padded black frame; and exactly one process owns the serial bus — a second would-be owner **fails loudly at startup** rather than corrupting serial frames.
  4. Across a full task run, VRAM does not climb toward OOM and no CoT text reaches TTS unsanitized — CoT is treated as logged/traced diagnostic output this milestone, never a user-facing voice surface.

**Plans**: TBD
**Notes**: Single-requirement phase, kept separate deliberately: 9→10 mirrors 6→7's bisect discipline for the second model family, and GX-05's live-hardware truth must not be entangled with Phase 9's hardware-free build. Its requirement count is low; its verification weight is not.
