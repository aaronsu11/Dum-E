"""Bounded GPU RPC latency sample for half-chunk scheduling; no robot imports."""
import argparse
import json
import time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from policy.lerobot.serialized_backend import SerializedLeRobotPolicyBackend
from policy_guard import parity_gate as gate
from policy_guard.replay_contract import JOINT_ORDER,load_case,read_json,write_evidence,now,sha256_file
from policy.lerobot.async_chunks import AsyncSettings


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--container',required=True)
    args=p.parse_args();root=Path(__file__).resolve().parents[1]
    if (args.workspace/'latency.json').exists():raise RuntimeError('Preserve existing latency evidence')
    binding=gate.inspect_container_binding(args.container,'127.0.0.1:8080','/checkpoints/model')
    lock=read_json(root/'corpus/phase7-trial3-20260912/input-lock.json')
    case={'record':'record_0005.npz','seed':20265907};arrays,entry=load_case(root/'corpus/frozen_v1_0',lock,case)
    observation=dict(zip(JOINT_ORDER,map(float,arrays['state'])))
    observation.update(front=arrays['video_front'],wrist=arrays['video_wrist'])
    policy=SerializedLeRobotPolicyBackend(host='127.0.0.1',port=8080,language_instruction=entry['instruction'])
    import grpc
    with grpc.insecure_channel('127.0.0.1:8080') as channel:
        grpc.channel_ready_future(channel).result(timeout=60)
    rows=[];started=now()
    try:
        for index in range(101):
            t=time.perf_counter();actions=policy.get_action(observation,entry['instruction']);elapsed=time.perf_counter()-t
            assert len(actions)==16 and np.isfinite(np.array([[a[k] for k in JOINT_ORDER] for a in actions])).all()
            if index:rows.append(elapsed)
    finally:policy.close()
    if binding!=gate.inspect_container_binding(args.container,'127.0.0.1:8080','/checkpoints/model'):
        raise RuntimeError('Serving instance changed')
    p99=float(np.percentile(rows,99,method='higher'))
    settings=AsyncSettings(p99)
    record={'schema_version':1,'kind':'async_request_latency','status':'complete','started_at':started,'ended_at':now(),
            'record':case['record'],'seed_policy':'ambient','samples_s':rows,'warmup_requests':1,'sample_count':100,
            'request_p99_s':p99,'median_s':float(np.median(rows)), 'percentile_method':'higher',
            'settings':vars(settings),'server_binding':binding,'observer_mode':'lightweight','policy_device':'cuda',
            'source_files':{name:sha256_file(root/name) for name in ['scripts/serve_observed_lerobot.py','policy_guard/chunk_observer.py','policy/lerobot/serialized_backend.py','docker/lerobot-policy/server.py']},
            'limitations':'Empirical percentile of 100 sequential calls on one observation; not a population tail estimate or task-quality evaluation.'}
    print(json.dumps(write_evidence(args.workspace,'latency.json',record)))
    print(json.dumps({'median_ms':record['median_s']*1000,'p99_ms':p99*1000,'deadline_ms':settings.request_deadline_s*1000}))


if __name__=='__main__':main()
