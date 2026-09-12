"""Close the explicitly amended 12-case milestone across separate physical trials.

Historical records retain their bytes and source identities. This command
validates evidence; it cannot authorize motion or replace a native reference.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json

from policy_guard import parity_gate as gate
from policy_guard.replay_contract import now, sha256_file, write_evidence

ROOT = Path(__file__).resolve().parents[1]


def validate_budget(trial, index):
    gate.require(type(trial.get('index')) is int and trial['index'] == index, 'trial order changed')
    gate.require(type(trial.get('iterations')) is int and trial['iterations'] == 20 and
                 type(trial.get('actions_per_chunk')) is int and trial['actions_per_chunk'] == 16 and
                 trial.get('action_delay') == .05, 'fixed trial budget changed')
    gate.require(trial.get('safety_stop') is False and type(trial.get('clamp_warnings')) is int and
                 trial['clamp_warnings'] == 0, 'trial safety event overrides score')
    for key in ('coherent', 'wrong_target', 'erratic', 'grasp'):
        gate.require(type(trial.get(key)) is bool, 'named boolean judgment required: ' + key)
    gate.text(trial.get('operator'), 'operator identity')
    return int(trial['coherent'] and not trial['wrong_target'] and not trial['erratic'])


def validate_trial(workspace, row, index, identity):
    ev = gate.Evidence(Path(row['workspace']))
    gate.require(ev.identity() == identity, 'trial input/profile/session identity differs')
    gate.require(row['trial'] == index and row['live_run'] == ev.reference('live-run.json'),
                 'trial summary reference changed')
    run = ev.json(row['live_run'])
    gate.require(all(run.get(k) == v for k, v in identity.items()) and
                 run.get('schema_version') == 1 and run.get('evidence_kind') == 'real_model',
                 'real trial identity required')
    gate.require(run.get('status') == 'partial' and run.get('authorized_trials') == 1 and
                 len(run['trials']) == 1, 'one separately authorized completed trial required')
    gate.validate_run_safety_journal(run)
    gate.require(not run.get('error') and run['instruction'] == 'Grab a banana and put it on the plate',
                 'trial failed or instruction changed')
    approval = gate.validate_live_approval(ev)
    gate.require(gate.approved_trial_indices(approval) == (index,), 'wrong single-trial approval')
    gate.require(row['approval'] == run['approval'] == ev.reference('live-approval.json'), 'trial approval changed')
    trial = run['trials'][0]
    success = validate_budget(trial, index)
    gate.require(run['directional_successes'] == success and row['directional_success'] == bool(success) and
                 row['grasp_success'] == trial['grasp'] and row['operator'] == trial['operator'] and
                 row['recorded_end'] == run['ended_at'], 'summary score/chronology changed')
    refs = [run['live_preflight'], run['run_preflight'], trial['preflight']]
    gate.require(run['preflights'] == refs, 'ordered preflight links changed')
    stages = ['live', 'run', f'trial-{index:02d}']
    records = [gate.validate_preflight_record(ev, ref, expected_stage=stage)
               for stage, ref in zip(stages, refs)]
    live, construction, reset = records
    gate.require(construction['previous'] == refs[0] and reset['previous'] == refs[1], 'preflight predecessor changed')
    gate.require(gate.timestamp(live['ended_at']) < gate.timestamp(construction['started_at']) <=
                 gate.timestamp(construction['ended_at']) < gate.timestamp(run['constructed_at']) ==
                 gate.timestamp(run['started_at']) < gate.timestamp(reset['started_at']) <=
                 gate.timestamp(reset['ended_at']) < gate.timestamp(trial['started_at']) <=
                 gate.timestamp(trial['ended_at']) <= gate.timestamp(run['ended_at']),
                 'approval/preflight/construction/reset chronology changed')
    from scripts.run_checkpoint_sanity import _runner_inputs, runtime_identity
    reviewed = _runner_inputs(ev.json(ev.json('release-review.json')['review_preflight']))
    gate.require(all(_runner_inputs(r) == reviewed for r in records), 'reviewed controller/source inputs changed')
    current_calibration = sha256_file(Path(reviewed['calibration_path']))
    gate.require(current_calibration == reviewed['calibration_sha256'], 'current calibration changed')
    identities = []
    for stage, ref, record, at in zip(stages, refs, records,
                                     (construction['started_at'], run['constructed_at'], trial['started_at'])):
        runtime = {k: record[k] for k in ('attestation', 'host', 'request')}
        gate.assert_live_release(ev, ref, expected_stage=stage, runtime=runtime,
                                 current_calibration_sha256=current_calibration, now=at)
        identities.append(runtime_identity(runtime))
    gate.require(identities[0] == identities[1] == identities[2], 'runtime changed inside trial')
    return {'index': index, 'workspace': str(ev.workspace), 'live_run': row['live_run'],
            'approval': row['approval'], 'preflights': refs, 'success': success,
            'grasp': trial['grasp'], 'started_at': run['started_at'], 'ended_at': run['ended_at']}


def validate_physical(workspace):
    ev = gate.evidence(workspace)
    summary = ev.json('physical-trials.json')
    gate.require(summary['schema_version'] == 1 and summary['kind'] == 'physical_trials_summary' and
                 summary['status'] == 'complete' and summary['trial_count'] == 3,
                 'complete three-trial summary required')
    rows = summary['trial_records']
    gate.require(len(rows) == 3 and [r['trial'] for r in rows] == [1, 2, 3] and
                 len({str(Path(r['workspace']).resolve()) for r in rows}) == 3, 'three distinct ordered trials required')
    checked = [validate_trial(ev, row, index, ev.identity()) for index, row in enumerate(rows, 1)]
    gate.require(all(gate.timestamp(a['ended_at']) < gate.timestamp(b['started_at'])
                     for a, b in zip(checked, checked[1:])), 'physical trials overlap or are reordered')
    successes, grasps = sum(t['success'] for t in checked), sum(t['grasp'] for t in checked)
    gate.require(successes >= 2 and summary['directional_successes'] == successes and
                 summary['grasp_successes'] == grasps, 'directional criterion/summary failed')
    return {'summary': ev.reference('physical-trials.json'), 'trials': checked,
            'directional_successes': successes, 'grasp_successes': grasps,
            'ended_at': checked[-1]['ended_at']}


def validate_historical_stock(workspace):
    """Recheck immutable stock outputs against their original approved profiles.

    Do not claim today's changed release-validator bytes produced yesterday's
    capture. The input manifest explicitly selects the historical workspace.
    """
    current = gate.evidence(workspace)
    selected = current.json('closeout-inputs.json')['stock']
    ev = gate.Evidence(Path(selected['workspace']))
    gate.require(selected['result'] == ev.reference('upstream-result.json'), 'historical stock result changed')
    result = ev.record(selected['result'])
    agreement = gate._decision(ev, 'tolerances')
    proposal = ev.json('tolerance-proposal.json')
    gate.require(gate.timestamp(proposal['ended_at']) < gate.timestamp(agreement['decided_at']) <
                 gate.timestamp(result['started_at']), 'stock proposal/agreement/comparison chronology changed')
    gate.require(result['agreement'] == ev.reference('tolerance-agreement.json') and
                 result['harness'] == proposal['harness'] and result['seed'] == 42 and
                 result['tag'] == 'new_embodiment', 'stock agreed scope changed')
    gate.require(result['checkpoint_fingerprint'] == ev.json('input-lock.json')['checkpoint_fingerprint'] ==
                 current.json('input-lock.json')['checkpoint_fingerprint'], 'stock checkpoint changed')
    gate.validate_producer_output(ev.bytes(result['producer_log']['path'], result['producer_log']['sha256']).decode(),
                                 result['producer_exit'])
    gate.require(type(result['consumer_exit']) is int and result['consumer_exit'] == 0, 'stock consumer failed')
    ev.bytes(result['consumer_log']['path'], result['consumer_log']['sha256'])
    gate.require(gate.parse_junit(ev.bytes(result['junit']['path'], result['junit']['sha256'])) == result['tests'],
                 'stock JUnit changed')
    node = gate.CONSUMER + '::' + gate.CASE
    coverage = ev.json(result['coverage'])
    gate.require(coverage['collected'] == [node] and coverage['reports'] == [
        {'nodeid': node, 'when': phase, 'outcome': 'passed', 'wasxfail': None}
        for phase in ('setup', 'call', 'teardown')], 'strict stock coverage failed')
    profiles = {p['backend']: p['observed'] for p in ev.json('profiles.json')['profiles'] if p['purpose'] == 'diagnostic'}
    bounds = proposal['comparisons']['diagnostic']['thresholds']['raw']
    gate.validate_launches(result, profiles, bounds)
    left = gate.validate_observation(ev, result['producer_observation'], profiles['native'])
    right = gate.validate_observation(ev, result['consumer_observation'], profiles['lerobot'])
    gate.require(left['observed']['rng_algorithm'] == right['observed']['rng_algorithm'], 'stock RNG mismatch')
    for field in ('inputs', 'noise'):
        a, b = ev.tensors(left[field]), ev.tensors(right[field])
        gate.require(set(a) == set(b) and bool(a), 'stock tensor fields changed')
        for key in a:
            gate.require(gate.array_comparison(a[key], b[key], {}, exact=True), 'stock exact input/noise mismatch')
    gate.require(result['left'] == left['raw'] and result['right'] == right['raw'], 'stock raw boundary changed')
    gate.compare_raw(ev, result['left'], result['right'], bounds)
    ev.bytes(result['artifact']['path'], result['artifact']['sha256'])
    return selected


def source_files():
    names = ('policy_guard/milestone_closeout.py', 'scripts/replay_milestone_final.py',
             'policy_guard/parity_gate.py', 'scripts/run_checkpoint_sanity.py',
             'scripts/approve_parity_evidence.py', 'docker/lerobot-policy/entrypoint.py')
    return {name: sha256_file(ROOT / name) for name in names}


def validate(workspace):
    ev = gate.evidence(workspace)
    release = gate.validate_release_evidence(ev)
    physical = validate_physical(ev)
    stock = validate_historical_stock(ev)
    from scripts.replay_milestone_final import validate as validate_final
    final = validate_final(ev.workspace)
    final_record = ev.json('final-regression.json')
    gate.require(gate.timestamp(physical['ended_at']) < gate.timestamp(final_record['started_at']),
                 'final native regression must follow the last physical trial')
    security = ev.json('security-review.json')
    gate.require(security['status'] == 'passed' and security['unresolved_high'] == 0 and
                 security['unresolved_critical'] == 0 and security['source_files'] == source_files(),
                 'security review missing, blocking or stale')
    return {**ev.identity(), 'schema_version': 1, 'status': 'complete', 'acceptance_mode': 'milestone_12',
            'source_files': source_files(), 'release_evidence': release, 'physical': physical,
            'historical_stock': stock, 'final_regression': ev.reference('final-regression.json'),
            'security_review': ev.reference('security-review.json'),
            'scope': '12 observations, one seed; retrospective physical-unit acceptance; three separately authorized trials',
            'requirements': ['PAR-01', 'PAR-02', 'PAR-03', 'SAFE-03'], 'grants_approval': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('physical', 'stock', 'check', 'record'))
    parser.add_argument('--workspace', type=Path, required=True)
    args = parser.parse_args()
    if args.command in ('physical', 'stock'):
        result = (validate_physical if args.command == 'physical' else validate_historical_stock)(args.workspace)
    else:
        result = validate(args.workspace)
        ev = gate.Evidence(args.workspace)
        if args.command == 'record':
            result = write_evidence(args.workspace, 'closeout.json', {**result, 'recorded_at': now()})
        else:
            archived = ev.json('closeout.json')
            gate.require({k: v for k, v in archived.items() if k != 'recorded_at'} == result, 'closeout archive changed')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
