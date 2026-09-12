from concurrent.futures import Future
from pathlib import Path
import pytest
from policy_guard.replay_contract import write_evidence,read_json,now
from scripts.voice_trial_control import ControlRelay,canonical_instruction


@pytest.mark.parametrize('text',[None,'pear','banana and apple','do not pick apple','stop banana'])
def test_only_unambiguous_reviewed_targets_allowed(text):
    with pytest.raises(ValueError):canonical_instruction(text)


def test_retarget_ack_is_after_application_and_not_replayed(tmp_path):
    class Skill:
        active=True
        def __init__(self):self.tasks=[]
        def set_task(self,text):self.tasks.append(text)
    skill=Skill();relay=ControlRelay(skill,tmp_path)
    write_evidence(tmp_path/'controls','0001.json',{'instruction':'pick an apple','requested_at':now()})
    relay.poll();relay.poll()
    assert skill.tasks==['Grab apple and put it on the plate']
    assert read_json(tmp_path/'runtime'/'ack-0001.json')['status']=='applied'
    assert (tmp_path/'runtime'/'active.json').exists()


def test_inactive_or_faulted_pick_is_not_acknowledged_as_applied(tmp_path):
    class Skill:
        active=False
        def set_task(self,text):raise RuntimeError('Pick stopped')
    write_evidence(tmp_path/'controls','0001.json',{'instruction':'apple','requested_at':now()})
    ControlRelay(Skill(),tmp_path).poll()
    assert read_json(tmp_path/'runtime'/'ack-0001.json')['status']=='rejected'
    assert not (tmp_path/'runtime'/'active.json').exists()


def test_closeout_scope_does_not_mutate_short_trial():
    from scripts import run_voice_closeout_trial as closeout,run_voice_physical_trial as short
    assert closeout.physical.PROTOCOL['chunks']==160
    assert closeout.physical.PROTOCOL['nominal_action_duration_s']==128
    assert short.physical.PROTOCOL['chunks']==20
    assert short.physical.PROTOCOL['retarget_allowed'] is False


@pytest.mark.asyncio
async def test_file_retarget_changes_real_scheduler_epoch_and_simulated_targets(tmp_path):
    import asyncio
    from types import SimpleNamespace
    from embodiment.so_arm10x.async_pick import AsyncPickSkill
    from policy.lerobot.async_chunks import AsyncSettings
    from policy_guard.replay_contract import JOINT_ORDER
    class Controller:
        def __init__(self):self.actions=[]
        def get_observation(self):return {}
        def get_current_images(self):return {}
        def move_to_initial_pose(self):pass
        def move_to_ready_pose(self):pass
        def set_target_state(self,target):self.actions.append(dict(target));return dict(target)
    class Policy:
        _handshaken=True;language_instruction='banana';_session=SimpleNamespace()
        def get_action(self,obs,text):return [dict.fromkeys(JOINT_ORDER,2. if 'apple' in text else 1.) for _ in range(16)]
        def set_task(self,text):self.language_instruction=text
        def close(self):pass
    controller=Controller();skill=AsyncPickSkill(controller,Policy(),AsyncSettings(.15))
    relay=ControlRelay(skill,tmp_path);relay.start()
    task=__import__('asyncio').create_task(asyncio.to_thread(skill.run,actions_to_execute=2))
    try:
        async with asyncio.timeout(4):
            while len(controller.actions)<4:await asyncio.sleep(.005)
            write_evidence(tmp_path/'controls','0001.json',{'instruction':'pick apple','requested_at':now()})
            await task
        assert len(controller.actions)==32
        assert any(e['event']=='retarget' and e['epoch']==1 for e in skill.last_events)
        assert any(a[JOINT_ORDER[0]]==2. for a in controller.actions)
        assert read_json(tmp_path/'runtime'/'ack-0001.json')['status']=='applied'
    finally:
        relay.close()
        if not task.done():skill.stop('Test cleanup');await task


def test_active_time_limit_stops_instead_of_extending_run(tmp_path):
    class Skill:
        active=True
        def __init__(self):self.reason=None
        def stop(self,reason):self.reason=reason
    t=[0.];skill=Skill();relay=ControlRelay(skill,tmp_path,clock=lambda:t[0])
    relay.poll();t[0]=140;relay.poll()
    assert skill.reason=='Closeout active-time limit reached' and relay.done.is_set()
