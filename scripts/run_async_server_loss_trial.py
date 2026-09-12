"""One explicitly approved bounded async server-loss physical trial, separate from Phase 7.

Retains existing hardware stop/clamp composition. No claim of exhaustive runtime
attestation; exact read-only serving configuration and current controller inputs
are approved here. Old Phase 7 approvals cannot authorize this runner.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time
import threading
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_checkpoint_sanity as safety
from policy_guard import parity_gate as gate
from policy_guard.replay_contract import read_json, write_evidence, sha256_file, load_case, now

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'corpus/phase7-trial3-20260912'
IMAGE = 'sha256:6758186bd24cd0745b7442dafbb6680cbc2a986fe387eb2de5aaf0b75dce9c98'
CONTAINER = 'dume-async-physical-20260912'
ENDPOINT = '127.0.0.1:8080'
PROTOCOL = {'trials': 1, 'chunks': 2, 'actions_per_chunk': 16, 'action_delay': .05,
            'reset': 'initial_then_ready', 'instruction': safety.INSTRUCTION, 'observer_mode': 'lightweight', 'scheduler': 'async', 'trace_states': True, 'server_fault_injection': True, 'kill_after_actions': 4, 'hold_sample_seconds': 2.0}
SOURCES = ('embodiment/so_arm10x/async_pick.py', 'policy/lerobot/async_chunks.py', 'policy/lerobot/serialized_backend.py', 'scripts/run_async_server_loss_trial.py', 'scripts/serve_observed_lerobot.py',
           'policy_guard/chunk_observer.py', 'scripts/run_checkpoint_sanity.py',
           'docker/lerobot-policy/server.py', 'docker/lerobot-policy/entrypoint.py',
           'embodiment/so_arm10x/controller.py', 'embodiment/so_arm10x/skills.py',
           'policy/lerobot/backend.py', 'policy/lerobot/session.py', 'policy/lerobot/features.py')


def serving_binding():
    inspected = json.loads(subprocess.check_output(['docker','inspect','--format','{{json .}}',CONTAINER]))
    binding = gate.container_binding(inspected, ENDPOINT, '/checkpoints/model')
    gate.require(binding['image_digest'] == IMAGE, 'pinned image changed')
    gate.require(inspected['HostConfig']['ReadonlyRootfs'] is True, 'read-only serving root required')
    env = dict(v.split('=',1) for v in inspected['Config']['Env'] if '=' in v)
    gate.require(env.get('DUME_CHUNK_OBSERVER') == 'lightweight' and not env.get('DUME_PARITY_ATTESTATION_PATH')
                 and not env.get('DUME_POLICY_SEED'), 'observer/attestation/RNG mode changed')
    gate.require(inspected['Path'] == 'python3' and inspected['Args'] == [
        '/app/scripts/serve_observed_lerobot.py','--host','0.0.0.0','--port','8080','--cpu-threads','1'],
        'serving command changed')
    for relative, destination in [('scripts','/app/scripts'),('policy_guard','/app/policy_guard'),
                                   ('docker/lerobot-policy','/app/docker/lerobot-policy'),
                                   ('checkpoints/GR00T-N1.7-3B-SO101','/checkpoints/model')]:
        mounts=[m for m in inspected['Mounts'] if m['Destination']==destination]
        gate.require(len(mounts)==1 and mounts[0]['RW'] is False and
                     Path(mounts[0]['Source']).resolve()==ROOT/relative, 'source/checkpoint mount changed')
    return binding


def sources():
    return {name:sha256_file(ROOT/name) for name in SOURCES}


def devices(settings):
    paths=[settings['robot_port'],f"/dev/video{settings['wrist_cam_idx']}",f"/dev/video{settings['front_cam_idx']}"]
    return {p:{'rdev':os.stat(p).st_rdev,'inode':os.stat(p).st_ino,
               'sysfs':str((Path('/sys/class/tty' if 'tty' in p else '/sys/class/video4linux')/Path(p).name/'device').resolve(strict=True))}
            for p in paths}


def validate_approval(approval, current):
    gate.require(approval.get('kind')=='async_server_loss_physical_trial' and approval.get('approved') is True,
                 'new explicit async lightweight approval required')
    gate.require(approval.get('operator')=='Aaron' and approval.get('operator_present') is True,
                 'named present operator required')
    gate.text(approval.get('user_authorization'),'actual user authorization')
    gate.require(approval.get('protocol')==PROTOCOL and approval.get('snapshot')==current,
                 'approved trial scope or current inputs changed')
    gate.require(0 <= (gate.timestamp(now())-gate.timestamp(approval['approved_at'])).total_seconds() <= 1800,
                 'operator presence approval is stale')


def checkpoint_metadata():
    root=ROOT/'checkpoints/GR00T-N1.7-3B-SO101'
    return {str(p.relative_to(root)):[p.stat().st_ino,p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ctime_ns]
            for p in sorted(root.rglob('*')) if p.is_file()}


def snapshot():
    safety.audit_dispatch_paths()
    lock=safety.load_input_lock(ROOT/'corpus/frozen_v1_0',ROOT/'checkpoints/GR00T-N1.7-3B-SO101')
    gate.require(lock==read_json(BASE/'input-lock.json'), 'current checkpoint or frozen inputs changed')
    controller=safety.controller_inputs(BASE)
    return {'controller_inputs':controller, 'devices':devices(controller['controller']),
            'serving':serving_binding(), 'sources':sources(), 'checkpoint_metadata':checkpoint_metadata(),
            'input_fingerprint':lock['fingerprint'], 'measurement_sha256':sha256_file(ROOT/'corpus/phase8-latency-20260912/latency.json')}


def probe_cameras(settings):
    import cv2
    result={}
    for key in ('wrist_cam_idx','front_cam_idx'):
        cap=cv2.VideoCapture(settings[key],cv2.CAP_V4L2)
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,640);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
            for _ in range(30):
                ok,frame=cap.read()
                gate.require(ok and frame.shape==(480,640,3), 'camera unavailable or wrong geometry: '+key)
            result[key]={'device':settings[key],'frames':30,'shape':list(frame.shape)}
        finally:cap.release()
    return result


def run(workspace):
    from policy.lerobot.serialized_backend import SerializedLeRobotPolicyBackend
    from embodiment.so_arm10x.async_pick import AsyncPickSkill, load_settings
    from policy.lerobot.async_chunks import InferenceStopped
    from policy.lerobot.session import LeRobotPolicySession
    gate.require(sys.stdin.isatty(), 'interactive stop channel required')
    gate.require(not (workspace/'live-run.json').exists() and not (workspace/'preflight.json').exists(),
                 'immutable attempt already exists')
    approval=read_json(workspace/'approval.json');current=snapshot();validate_approval(approval,current)
    async_settings=load_settings(ROOT/'corpus/phase8-latency-20260912/latency.json')
    settings=current['controller_inputs']['controller']
    cameras=probe_cameras(settings)
    stop=safety.StopLatch();controller=None;policy=None;checked=None
    trigger=threading.Event();kill_thread=None;injection={};physical_actions=[]
    failures=[]
    def inject_loss():
        if not trigger.wait(15):
            injection['error']='No injection point reached';return
        try:
            gate.require(serving_binding()==current['serving'], 'Serving identity changed before injection')
            injection['requested_at']=time.monotonic()
            killed=subprocess.run(['docker','kill',CONTAINER],capture_output=True,text=True,timeout=3)
            injection['completed_at']=time.monotonic();injection['returncode']=killed.returncode
        except Exception as exc:
            injection['error']=str(exc);stop.trip('Server-loss injector failed')
    class TrialController(safety.StopGuardedController):
        def set_target_state(self,target):
            result=super().set_target_state(target)
            if checked is not None and checked.active:
                physical_actions.append({'at':time.monotonic(),'target':dict(target)})
                if len(physical_actions)==PROTOCOL['kill_after_actions']:trigger.set()
            return result
    result={'schema_version':1,'kind':'async_server_loss_physical_trial','status':'failed',
            'protocol':PROTOCOL,'approval':gate.Evidence(workspace).reference('approval.json'),
            'started_at':now(),'operator':'Aaron','exhaustive_attestation':False,'timings':[]}
    def check_current():
        stop.check()
        gate.require(serving_binding()==current['serving'] and sources()==current['sources'] and
                     checkpoint_metadata()==current['checkpoint_metadata'],
                     'serving instance or source changed')
        gate.require(sha256_file(Path(current['controller_inputs']['calibration_path']))==
                     current['controller_inputs']['calibration_sha256'], 'calibration changed')
        gate.require(devices(settings)==current['devices'], 'hardware device reconnected or changed')
        stop.check()
    with safety.armed_stop(stop):
        try:
            policy=SerializedLeRobotPolicyBackend(host='127.0.0.1',port=8080,camera_keys=['front','wrist'],
                                       robot_state_keys=list(safety.JOINT_ORDER),language_instruction=safety.INSTRUCTION)
            gate.require(policy._policy_type=='groot' and policy._checkpoint_path=='/checkpoints/model' and
                         policy._actions_per_chunk==16 and policy._device in ('cuda','cuda:0'), 'client serving options changed')
            policy._session.close()
            policy._session=LeRobotPolicySession(ENDPOINT,deadline_s=900,max_attempts=1,
                                                handshake_deadline_s=900,handshake_max_attempts=1)
            lock=read_json(BASE/'input-lock.json')
            arrays,entry=load_case(ROOT/'corpus/frozen_v1_0',lock,{'record':'record_0005.npz','seed':20265907})
            observation=dict(zip(safety.JOINT_ORDER,map(float,arrays['state'])))
            observation.update(front=arrays['video_front'],wrist=arrays['video_wrist'])
            # Load/handshake and one full validated request before controller construction.
            safety.CheckedPolicy(policy,stop).get_action(observation,entry['instruction'])
            check_current();validate_approval(approval,snapshot())
            logs=subprocess.check_output(['docker','logs',CONTAINER],stderr=subprocess.STDOUT,text=True)
            gate.require('SAFE-01 guard: PASS' in logs and '"observer_mode": "lightweight"' in logs,
                         'guarded load and async lightweight request not proven')
            write_evidence(workspace,'preflight.json',{'status':'complete','checked_at':now(),
                           'snapshot':current,'cameras':cameras,'warmup_requests':1,'exhaustive_attestation':False})
            check_current()
            result['constructed_at']=now()
            controller=TrialController(stop=stop,**settings)
            safety.validate_loaded_calibration(controller,current['controller_inputs'])
            controller.connect(calibrate=False)
            check_current()
            checked=AsyncPickSkill(controller,policy,async_settings,trace_directory=workspace/'traces',
                                   on_failure=lambda reason: failures.append({'at':time.monotonic(),'reason':reason}))
            result['motion_started_at']=now()
            print('ONE SERVER-LOSS TRIAL: reset then kill policy server after four actions. STOP ARMED.',flush=True)
            kill_thread=threading.Thread(target=inject_loss,daemon=True)
            kill_thread.start()
            try:
                checked.run(pose='initial',actions_to_execute=2,action_horizon=16,
                            language_instruction=safety.INSTRUCTION)
            except InferenceStopped as exc:
                detected=time.monotonic()
                kill_thread.join(timeout=4)
                gate.require(injection.get('returncode')==0 and 'error' not in injection,
                             'Expected server injection did not complete')
                bound=async_settings.prefetch_remaining*async_settings.period_s+async_settings.request_deadline_s+async_settings.period_s
                last=physical_actions[-1]['at']
                gate.require(failures and physical_actions, 'No physical dispatch or failure callback recorded')
                gate.require(0<=detected-injection['requested_at']<=bound,
                             'Measured detection exceeded configured bound')
                gate.require(last-injection['requested_at']<=bound, 'Late action after server loss')
                gate.require(stop.stopped and stop.clamp_warnings==0, 'Unexpected clamp or missing stop latch')
                before=len(stop.journal)
                try:
                    checked.other_motion(controller.move_to_initial_pose)
                except InferenceStopped:
                    pass
                else:
                    raise RuntimeError('Reset was not refused after fault')
                gate.require(len(stop.journal)==before, 'Reset refusal dispatched hardware')
                # Read only: do not clear the latch, reset, disable, or re-enable torque.
                hold=[]
                for _ in range(40):
                    hold.append({'at':time.monotonic(),'state':list(map(float,controller.get_current_state()))})
                    time.sleep(.05)
                result.update(status='awaiting_operator_hold_observation',expected_fault=str(exc),
                              failure_detected_at=detected,declared_bound_s=bound,
                              detection_after_kill_s=detected-injection['requested_at'],
                              last_command_after_kill_s=last-injection['requested_at'],
                              post_fault_reset_refused=True,hold_samples=hold,
                              voice_status='not_connected_callback_only')
                print('Expected watchdog stop recorded. Read-only hold samples collected; no second trial.',flush=True)
            else:
                raise RuntimeError('Server-loss trial ended without the expected watchdog stop')
            result['motion_ended_at']=now()
        except (Exception,KeyboardInterrupt) as exc:
            stop.trip(str(exc) or type(exc).__name__)
            result['error']=type(exc).__name__+': '+str(exc)
        finally:
            for resource in (controller,policy):
                if resource is not None:
                    try:
                        resource.disconnect() if resource is controller else resource.close()
                    except Exception as exc:stop.trip('cleanup failed: '+str(exc))
            if stop.stopped and result['status']!='awaiting_operator_hold_observation':result['status']='failed'
            result.update(injection=injection,physical_actions=physical_actions,failure_callbacks=failures)
            first_stop=next((i for i,e in enumerate(stop.journal) if e['kind']=='stop'),None)
            result['dispatches_after_stop']=sum(e['kind']=='dispatch' for e in stop.journal[first_stop+1:]) if first_stop is not None else None
            if result['dispatches_after_stop'] not in (None,0):result['status']='failed'
            result.update(ended_at=now(),iterations=sum(e.get('event')=='action' for e in checked.last_events) if checked else 0,
                          safety_stop=stop.stopped,clamp_warnings=stop.clamp_warnings,
                          stop_reason=stop.reason,events=stop.journal)
            write_evidence(workspace,'live-run.json',result)
    print(json.dumps({k:result[k] for k in ('status','iterations','safety_stop','clamp_warnings')}),flush=True)
    return 0 if result['status']=='awaiting_operator_hold_observation' else 1


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['snapshot','run'])
    p.add_argument('--workspace',type=Path,required=True);args=p.parse_args()
    if args.command=='snapshot':
        print(json.dumps(snapshot(),sort_keys=True));return 0
    return run(args.workspace)


if __name__=='__main__':raise SystemExit(main())
