"""Descriptive chunk-boundary analysis of an existing physical trace; no hardware."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from policy_guard.contracts import JOINT_ORDER, sha256_file, write_evidence


def analyze(path, output):
    trace = json.loads(path.read_text())
    actions = [e for e in trace['events'] if e['event'] == 'action']
    chunks = [e for e in trace['events'] if e['event'] == 'chunk']
    samples = trace['state_samples']
    if len(samples) != len(actions) or len(samples) < 3:
        raise ValueError('Aligned action and state samples required')
    if [s['step'] for s in samples] != [a['step'] for a in actions]:
        raise ValueError('State/action steps differ')
    times = np.array([a['at'] for a in actions])
    state_times = np.array([s['at'] for s in samples])
    states = np.array([s['state'] for s in samples])
    targets = np.array([[a['target'][j] for j in JOINT_ORDER] for a in actions])
    if states.shape != targets.shape or not np.isfinite(states).all():
        raise ValueError('Finite six-joint samples required')
    if not (np.diff(times) > 0).all() or not (np.diff(state_times) > 0).all():
        raise ValueError('Monotonic sample times required')
    boundaries = sorted(set(int(np.searchsorted(times, c['received_at'])) for c in chunks[1:]))
    boundaries = [b for b in boundaries if 0 < b < len(actions)]
    # Include the transition interval and one following interval to allow lag.
    indices = np.arange(1, len(actions))
    mask = np.isin(indices, boundaries)
    window = mask | np.isin(indices, [b + 1 for b in boundaries])
    velocity = np.diff(states, axis=0) / np.diff(state_times)[:, None]
    target_step = np.diff(targets, axis=0)
    def stats(values):
        return {'count':len(values), 'median':float(np.median(values)),
                'p95':float(np.percentile(values, 95)), 'max':float(np.max(values))}
    if not boundaries or not (~window).any():
        raise ValueError('Boundary and non-boundary samples required')
    result = {'trace_sha256':sha256_file(path), 'actions':len(actions),
              'boundaries':boundaries, 'method':'Finite differences at actual readback times; boundary is first action after chunk acceptance. Window includes that interval and the next.',
              'limitations':'One descriptive trial; no prespecified spike threshold or independent joint timestamps. Finite differences include encoder quantization; no formal continuity pass inferred.',
              'joints':{}}
    for i, joint in enumerate(JOINT_ORDER):
        result['joints'][joint] = {'unit':'percentage points/s' if joint == 'gripper.pos' else 'degrees/s',
            'abs_velocity_boundary':stats(abs(velocity[mask, i])),
            'abs_velocity_other':stats(abs(velocity[~mask, i])),
            'abs_velocity_boundary_window':stats(abs(velocity[window, i])),
            'abs_velocity_outside_window':stats(abs(velocity[~window, i])),
            'abs_target_step_boundary':stats(abs(target_step[mask, i])),
            'abs_target_step_other':stats(abs(target_step[~mask, i]))}
    write_evidence(output, 'joint-continuity.json', result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(6, 1, figsize=(13, 13), sharex=True)
    for i, (ax, joint) in enumerate(zip(axes, JOINT_ORDER)):
        ax.plot(state_times[1:] - times[0], velocity[:, i], linewidth=1, color='#176b9a')
        for b in boundaries:
            ax.axvline(times[b] - times[0], color='#d99000', alpha=.25, linewidth=.8)
        ax.set_ylabel(joint.replace('.pos', '') + '\n' + result['joints'][joint]['unit'])
        ax.grid(alpha=.2)
    axes[-1].set_xlabel('Seconds from first action; amber lines mark chunk acceptance')
    fig.suptitle('Successful async banana trial: measured joint velocities')
    fig.tight_layout()
    fig.savefig(output / 'joint-velocities.png', dpi=160)
    plt.close(fig)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('trace',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();analyze(args.trace,args.output)
