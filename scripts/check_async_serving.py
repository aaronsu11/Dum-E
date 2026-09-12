"""Real GPU + real gRPC with simulated controller; retarget and server-loss check."""
import argparse
from pathlib import Path
import sys,json,time,threading,subprocess
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from embodiment.so_arm10x.async_pick import AsyncPickSkill,load_settings
from policy.lerobot.serialized_backend import SerializedLeRobotPolicyBackend
from policy.lerobot.async_chunks import InferenceStopped
from policy_guard import parity_gate as gate
from policy_guard.replay_contract import JOINT_ORDER,read_json,load_case,write_evidence,now


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--container',required=True);p.add_argument('--measurement',type=Path);args=p.parse_args()
    root=Path(__file__).resolve().parents[1];workspace=args.workspace
    if (workspace/'async-serving-check.json').exists():raise RuntimeError('Existing attempt must be preserved')
    measurement_path=args.measurement or workspace/'latency.json'
    measurement=read_json(measurement_path);settings=load_settings(measurement_path)
    binding=gate.inspect_container_binding(args.container,'127.0.0.1:8080','/checkpoints/model')
    assert binding==measurement['server_binding']
    arrays,entry=load_case(root/'corpus/frozen_v1_0',read_json(root/'corpus/phase7-trial3-20260912/input-lock.json'),
                          {'record':'record_0005.npz','seed':20265907})
    observation=dict(zip(JOINT_ORDER,map(float,arrays['state'])))
    observation.update(front=arrays['video_front'],wrist=arrays['video_wrist'])
    class Controller:
        def __init__(self,trigger):self.actions=[];self.trigger=trigger;self.event=threading.Event();self.resets=[]
        def get_observation(self):return observation
        def get_current_images(self):return {'front':arrays['video_front']}
        def move_to_initial_pose(self):self.resets.append('initial')
        def move_to_ready_pose(self):self.resets.append('ready')
        def set_target_state(self,action):
            self.actions.append({'at':time.monotonic(),'target':action})
            if len(self.actions)==self.trigger:self.event.set()
            return dict(action)
    class Policy(SerializedLeRobotPolicyBackend):
        def __init__(self):
            super().__init__(host='127.0.0.1',port=8080,language_instruction=entry['instruction'])
            self.instructions=[]
        def get_action(self,observation,lang=None):
            self.instructions.append(lang)
            return super().get_action(observation,lang)
    policy=Policy()
    # This is the exact service just loaded and measured above. Ready flushes
    # transport queues without starting a second model load.
    policy._session.probe_ready_or_raise();policy._handshaken=True
    result={'schema_version':1,'status':'failed','started_at':now(),'hardware':'simulated_controller_no_robot_access',
            'server_binding':binding,'measurement':{'path':str(measurement_path.resolve()),'sha256':__import__('hashlib').sha256(measurement_path.read_bytes()).hexdigest()}}
    try:
        controller=Controller(8);skill=AsyncPickSkill(controller,policy,settings)
        errors=[]
        def retarget():
            try:
                if not controller.event.wait(10):raise RuntimeError('Retarget point not reached')
                skill.set_task('Grab an apple and put it on the plate')
            except Exception as exc:errors.append(str(exc))
        thread=threading.Thread(target=retarget,daemon=True);thread.start()
        skill.run(actions_to_execute=3);thread.join(timeout=1)
        assert not errors and not thread.is_alive() and len(controller.actions)==48
        assert 'Grab an apple and put it on the plate' in policy.instructions
        assert gate.inspect_container_binding(args.container,'127.0.0.1:8080','/checkpoints/model')==binding
        result['retarget']={'status':'passed','actions':controller.actions,'instructions':policy.instructions,
                            'events':skill.last_events,'same_loaded_instance':True}
        controller=Controller(4);failures=[];skill=AsyncPickSkill(controller,policy,settings,on_failure=failures.append)
        kill={}
        def kill_server():
            if controller.event.wait(10):
                kill['requested_at']=time.monotonic()
                kill['returncode']=subprocess.run(['docker','kill',args.container],capture_output=True,timeout=10).returncode
                kill['completed_at']=time.monotonic()
        thread=threading.Thread(target=kill_server,daemon=True);thread.start()
        try:skill.run(actions_to_execute=3)
        except InferenceStopped as exc:failure=str(exc);detected=time.monotonic()
        else:raise AssertionError('Server death did not latch a stop')
        thread.join(timeout=12)
        assert kill.get('returncode')==0 and failures and 0<len(controller.actions)<48
        bound=settings.prefetch_remaining*settings.period_s+settings.request_deadline_s+settings.period_s
        assert detected-kill['requested_at']<=bound+.1
        count=len(controller.actions)
        try:skill.other_motion(controller.move_to_initial_pose)
        except InferenceStopped:pass
        else:raise AssertionError('Post-fault reset allowed')
        assert len(controller.actions)==count and controller.resets==['initial','ready']
        result['server_loss']={'status':'passed','failure':failure,'kill':kill,'detected_at':detected,
            'detected_after_request_s':detected-kill['requested_at'],'declared_bound_s':bound,
            'last_command_after_request_s':max(0,controller.actions[-1]['at']-kill['requested_at']),
            'actions':controller.actions,'events':skill.last_events,'failure_callback':failures,'post_fault_reset_refused':True}
        result['status']='complete'
    except Exception as exc:
        result['error']=type(exc).__name__+': '+str(exc)
        raise
    finally:
        policy.close();result['ended_at']=now();write_evidence(workspace,'async-serving-check.json',result)
    print(json.dumps({'status':result['status'],'retarget':'passed','server_loss':result['server_loss']['detected_after_request_s']}))


if __name__=='__main__':main()
