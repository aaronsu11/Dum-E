import asyncio
from types import SimpleNamespace
import threading
import time
import pytest
from embodiment.so_arm10x.async_pick import AsyncPickSkill
from policy.lerobot.async_chunks import AsyncSettings,InferenceStopped
from policy_guard.contracts import JOINT_ORDER


class Controller:
    def __init__(self):self.sent=[];self.resets=[]
    def get_observation(self):return {'n':len(self.sent)}
    def set_target_state(self,target):self.sent.append((time.monotonic(),target));return dict(target)
    def get_current_images(self):return {'front':'image'}
    def move_to_initial_pose(self):self.resets.append('initial')
    def move_to_ready_pose(self):self.resets.append('ready')


class Policy:
    def __init__(self):
        self._handshaken=True;self._session=SimpleNamespace();self.language_instruction='banana'
        self.calls=[];self.closed=False;self.fail=False
    def prepare_execution(self, observation, instruction, *, deadline_s):
        if not self._handshaken:
            self._handshake()
        self.get_action(observation, instruction)
    def get_action(self,obs,task):
        self.calls.append((threading.current_thread().name,obs,task))
        if self.fail and len(self.calls)>2:raise RuntimeError('server killed')
        return [dict.fromkeys(JOINT_ORDER,1.) for _ in range(16)]
    def set_task(self,task):self.language_instruction=task
    def close(self):self.closed=True


@pytest.mark.asyncio
async def test_async_pick_offloads_inference_and_keeps_event_loop_responsive():
    controller=Controller();policy=Policy();skill=AsyncPickSkill(controller,policy,AsyncSettings(.15))
    task=asyncio.create_task(asyncio.to_thread(skill.run,actions_to_execute=1))
    ticks=0
    while not task.done():ticks+=1;await asyncio.sleep(.01)
    assert await task=={'front':'image'}
    assert len(controller.sent)==16 and controller.resets==['initial','ready']
    assert ticks>20
    assert any(name.startswith('policy-inference') for name,_,_ in policy.calls)
    intervals=[b[0]-a[0] for a,b in zip(controller.sent,controller.sent[1:])]
    assert min(intervals)>.025  # no catch-up burst


def test_server_failure_latches_and_prevents_reset_or_more_motion():
    controller=Controller();policy=Policy();policy.fail=True;errors=[]
    skill=AsyncPickSkill(controller,policy,AsyncSettings(.15),on_failure=errors.append)
    with pytest.raises(InferenceStopped,match='server killed'):skill.run(actions_to_execute=2)
    count=len(controller.sent);assert 0<count<32 and errors and policy.closed
    with pytest.raises(InferenceStopped):skill.run(actions_to_execute=1)
    with pytest.raises(InferenceStopped):skill.other_motion(controller.move_to_initial_pose)
    assert len(controller.sent)==count and controller.resets==['initial','ready']


@pytest.mark.asyncio
async def test_retarget_changes_future_requests_without_reconnecting():
    controller=Controller();policy=Policy();skill=AsyncPickSkill(controller,policy,AsyncSettings(.15))
    task=asyncio.create_task(asyncio.to_thread(skill.run,actions_to_execute=2))
    while not skill.active:await asyncio.sleep(.005)
    await asyncio.to_thread(skill.set_task,'apple')
    await task
    assert any(instruction=='apple' for _,_,instruction in policy.calls)
    assert not policy.closed and len(controller.sent)==32


def test_opt_in_trace_records_actual_readback(tmp_path):
    import json
    controller=Controller();controller.get_current_state=lambda: [7.]*6
    skill=AsyncPickSkill(controller,Policy(),AsyncSettings(.15),trace_directory=tmp_path)
    skill.run(actions_to_execute=1)
    files=list(tmp_path.glob('async-trace-*.json'));assert len(files)==1
    trace=json.loads(files[0].read_text())
    assert trace['status']=='complete' and len(trace['state_samples'])==16
    assert trace['state_samples'][0]['state']==[7.]*6
    assert trace['events'][1]['event'] in ('action','chunk')


def test_successive_tasks_reuse_handshake_but_get_fresh_actions():
    class PersistentPolicy(Policy):
        def __init__(self):
            super().__init__();self._handshaken=False;self.handshakes=0
        def _handshake(self):
            self.handshakes+=1;self._handshaken=True
        def get_action(self,obs,task):
            self.calls.append((threading.current_thread().name,obs,task))
            return [dict.fromkeys(JOINT_ORDER,2. if task=='apple' else 1.) for _ in range(16)]
    controller=Controller();policy=PersistentPolicy();skill=AsyncPickSkill(controller,policy,AsyncSettings(.15))
    original_session=policy._session
    skill.run(actions_to_execute=1,language_instruction='banana')
    skill.run(actions_to_execute=1,language_instruction='apple')
    assert policy.handshakes==1 and policy._session is original_session and not policy.closed
    assert len(controller.sent)==32
    assert all(target[JOINT_ORDER[0]]==1. for _,target in controller.sent[:16])
    assert all(target[JOINT_ORDER[0]]==2. for _,target in controller.sent[16:])
    assert controller.resets==['initial','ready','initial','ready']
    assert not skill.active
