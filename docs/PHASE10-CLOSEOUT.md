# Phase 10 closeout

**Status: passed under the operator-amended scope, 2026-09-13.**
The phase covers SO101 configuration, camera/state/action plumbing and bounded
execution across four policies, with async physical validation required only for
Pi0.5. Picking accuracy and fine-tuning are not acceptance gates.

## Completed

- **14 physical command runs**, comprising 11 policy runs and three synthetic
  diagnostics; **1,286 targets** total. Raw policy outputs, bounded targets,
  measured joints, configurations and operator observations are retained.
- **G0.5:** live inputs, native inference and bounded dispatch verified. Its small
  absolute targets produced no visible policy motion; that outcome was accepted
  as an integration result. Synthetic motion is not counted as policy success.
- **MolmoAct2, mapped Pi0.5 and calibrated GR00T:** operator-confirmed visible
  bounded synchronous motion. Pi0.5 base remains a separate input-only profile.
- **Pi0.5 RTC:** local RTX 3060; three overlapping chunks, 100 played targets,
  visible smooth motion and safe hold confirmed. RPCs were **443 / 427 / 430 ms**;
  both replacements arrived after nine ticks. Maximum action interval was
  **50.23 ms**, with no errors or controller clamp warnings.
- **69 focused regression tests passed.** Code review found and fixed one
  deadline gap: a reply could expire during projection/evidence writing.
  A regression test now verifies that expiration stops before dispatch.

The small deadline fix was validated in software after trial14; no additional
physical trial was run. Trial14's original results and source fingerprint were
preserved. The exact executed scheduler source is retained in the closeout
evidence, verified against that fingerprint.

## Retained limitations

The original **GX-05 coherent G0.5 task-motion requirement remains open**.
Dataset-specific ready-pose/all-joint qualification, sustained or physical-fault
RTC qualification, and an explicit second-serial-owner rejection test were not
established by these bounded trials. These have machine-readable `deferred`
results; they are not silently counted as passed.

No additional async trials for G0.5, MolmoAct2 or GR00T are required by this
phase. Completion does not certify unattended operation or close the entire
milestone and its existing backlog.

## Records

- [All-run metrics and configuration](PHASE10-INFERENCE-TRIAL-SUMMARY.md)
- [Pi0.5 mapping](PI05-SO101-CONFIGURATION.md) and [RTC evidence](PI05-RTC-LOCAL.md)
- [Per-run CSV](verification/phase10/physical-runs.csv)
- [Verification](verification/phase10/10-VERIFICATION.md)
- [UAT](verification/phase10/10-UAT.json)
- [Code review](verification/phase10/10-CODE-REVIEW.md)

All three phase plans and their summaries are complete. The roadmap and state
are updated; original criteria remain archived and GX-05 remains unchecked.
Review was performed inline; no independent review is claimed. Portable review
records are included under `docs/verification/phase10/`; raw corpus remains local.
The local Pi0.5 server remains resident without hardware access. No EC2
configuration was changed during closeout.
