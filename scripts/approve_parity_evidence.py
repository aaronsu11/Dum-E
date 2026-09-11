"""Explicit local review of parity evidence; never constructs or connects a robot."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_guard.parity_gate import (  # noqa: E402
    DECISIONS, MILESTONE_LIVE_SCOPE, MILESTONE_TRIAL1_SCOPE, MILESTONE_MODE, DecisionRecord, Evidence, PrerequisiteError,
    decision_evidence, evidence,
    fingerprint_configuration, require, timestamp, utc_now, validate_closeout,
    validate_golden_approval, validate_golden_candidate, validate_live_approval,
    validate_preflight_record, validate_release_evidence, validate_release_review,
    validate_tolerance_agreement, validate_tolerance_proposal, write_evidence,
)
from policy_guard.replay_contract import canonical  # noqa: E402


def _required(prompt, message):
    while True:
        try:
            answer = prompt(message).strip()
        except (EOFError, StopIteration) as exc:
            raise PrerequisiteError("not run: explicit operator decision unavailable") from exc
        if answer:
            return answer


def record_decision(workspace, kind, *, prompt=input, clock=utc_now, test_only=False, trial1_only=False):
    """Canonical writer for tolerance, golden and live review, including rejection.

    The optional prompt/clock/test_only arguments are injection seams for tests
    and explicit checkpoint transcription. The CLI exposes no approval flags.
    Every call displays the full subject before requiring named affirmative input.
    """
    ev = Evidence(workspace, test_only=test_only)
    subject_name, decision_name = DECISIONS[kind]
    require(not (ev.workspace / decision_name).exists(), "immutable decision exists; select an explicit successor session")
    validate = {"tolerances": validate_tolerance_proposal, "golden": validate_golden_candidate,
                "live": validate_release_review}[kind]
    subject = validate(ev)
    print(f"Review {kind}: {ev.reference(subject_name)['sha256']}")
    print(canonical(subject).decode())
    milestone_live = kind == "live" and subject["release_evidence"].get("acceptance_mode") == MILESTONE_MODE
    require(not trial1_only or milestone_live, "trial-1-only scope requires milestone live review")
    if kind == "live":
        print("joint_order:", ", ".join(ev.json("input-lock.json")["joint_order"]))
        if milestone_live:
            print("This decision approves the scoped native reference AND " +
                  ("physical trial 1 only." if trial1_only else "the three-trial physical test together."))
            print("coverage: 12 observations / one seed per observation; exhaustive parity is not established")
            print("native reference:", canonical(ev.reference("milestone-golden-candidate.json")).decode())
            print("fresh native verification:", canonical(ev.reference("milestone-golden-replay.json")).decode())
            accepted = ev.json("milestone-acceptance.json")
            print("physical criteria:", canonical({k: accepted[k] for k in ("limits", "units")}).decode())
            print("measured deviations, bias and per-index trends:", canonical(accepted["metrics"]).decode())
            print("preserved strict result:", ev.json("milestone-report.json")["status"])
            print("caveats:", canonical(accepted["caveats"]).decode())
            print("Operator presence must be confirmed before starting motion; this command accesses no hardware.")
        else:
            # Preserve the exhaustive report and its original decision scope.
            print("coverage: 120 records / 600 cases per comparison")
            print("thresholds:", canonical(ev.json("tolerance-proposal.json")).decode())
            print("deviations, bias, slope, per_index_bias, verdicts and caveats:")
            report = ev.json("offline-report.json")
            for comparison in report["comparisons"]:
                print(canonical({k: comparison[k] for k in ("name", "metrics", "passed")}).decode())
            print("caveats:", canonical(report["caveats"]).decode())
    operator = _required(prompt, "Operator identity (required): ")
    rationale = _required(prompt, "Intentional-change/review rationale (required): ")
    while True:
        question = ("Type approve or reject for this exact scoped native reference AND " +
                    ("physical trial 1 only" if trial1_only else "three-trial physical test") + " (no default): "
                    if milestone_live else "Type approve or reject for this exact subject (no default): ")
        answer = _required(prompt, question).lower()
        if answer in ("approve", "reject"):
            break
    record = {
        **asdict(DecisionRecord(
            subject_digest=ev.reference(subject_name)["sha256"], decision_type=kind,
            decision="approved" if answer == "approve" else "rejected", operator=operator,
            decided_at=clock(), evidence_reviewed=decision_evidence(ev, kind), rationale=rationale,
        )), "schema_version": 1, **ev.identity(),
        "evidence_kind": "test_only" if test_only else "real_model",
    }
    if milestone_live:
        record["approval_scope"] = list(MILESTONE_TRIAL1_SCOPE if trial1_only else MILESTONE_LIVE_SCOPE)
    require(timestamp(subject["ended_at"]) <= timestamp(record["decided_at"]), "decision predates subject")
    if record["decision"] == "approved":
        {"tolerances": validate_tolerance_agreement, "golden": validate_golden_approval,
         "live": validate_live_approval}[kind](ev, decision=record)
    archive = f"decisions/{fingerprint_configuration(record)}.json"
    write_evidence(ev.workspace, archive, record)
    return write_evidence(ev.workspace, decision_name, record)


def prepare_live(workspace, preflight, *, clock=utc_now):
    ev = evidence(workspace)
    release = validate_release_evidence(ev)
    ref = ev.reference(preflight)
    record = validate_preflight_record(ev, ref, expected_stage="review")
    when = clock()
    require(timestamp(record["ended_at"]) <= timestamp(when), "review preflight not yet complete")
    payload = {
        **ev.identity(), "schema_version": 1, "status": "complete",
        "evidence_kind": "test_only" if ev.test_only else "real_model",
        "started_at": when, "ended_at": when, "review_preflight": ref,
        "release_evidence": release,
    }
    return write_evidence(ev.workspace, "release-review.json", payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in (*DECISIONS, "prepare-live", "check"):
        command = commands.add_parser(name)
        command.add_argument("--workspace", type=Path, required=True)
        if name == "prepare-live":
            command.add_argument("--preflight", required=True)
        if name == "check":
            command.add_argument("--stage", choices=(*DECISIONS, "closeout"), required=True)
    args = parser.parse_args(argv)
    try:
        if args.command in DECISIONS:
            if not sys.stdin.isatty():
                raise PrerequisiteError("not run: interactive named operator review required")
            ref = record_decision(args.workspace, args.command)
            decision = Evidence(args.workspace).json(ref)
            print(canonical({"status": decision["decision"], "decision": ref}).decode())
            return 0 if decision["decision"] == "approved" else 1
        if args.command == "prepare-live":
            prepare_live(args.workspace, args.preflight)
        else:
            {"tolerances": validate_tolerance_agreement, "golden": validate_golden_approval,
             "live": validate_live_approval, "closeout": validate_closeout}[args.stage](args.workspace)
        print('{"status":"complete"}')
        return 0
    except (PrerequisiteError, FileNotFoundError) as exc:
        print(canonical({"status": "not_run", "message": f"not run: {exc}"}).decode())
        return 2
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(canonical({"status": "failed", "message": str(exc)}).decode())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
