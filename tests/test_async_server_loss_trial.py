import copy
import json
import sys
import threading
from types import SimpleNamespace
import pytest
from scripts import run_async_server_loss_trial as api


def test_server_loss_needs_its_own_current_approval():
    approval={'kind':'async_server_loss_physical_trial','approved':True,'operator':'Aaron',
              'operator_present':True,'user_authorization':'approved server-loss trial',
              'approved_at':api.now(),'protocol':copy.deepcopy(api.PROTOCOL),'snapshot':{}}
    api.validate_approval(approval,{})
    approval['kind']='async_single_physical_trial'
    with pytest.raises(Exception):api.validate_approval(approval,{})
    assert api.PROTOCOL['kill_after_actions']==4
    assert api.PROTOCOL['chunks']==2


def test_fault_runner_records_bounded_stop_read_only_hold_and_no_reset(tmp_path,monkeypatch):
    import numpy as np
    import policy.lerobot.serialized_backend as backend
    import policy.lerobot.session as session
    from embodiment.so_arm10x.async_pick import AsyncPickSkill
    from policy.lerobot.async_chunks import AsyncSettings
    import embodiment.so_arm10x.async_pick as driver
    killed=threading.Event();controllers=[]
    current={'controller_inputs':{'controller':{},'calibration_path':str(tmp_path/'cal'),
                'calibration_sha256':'cal'},'serving':{},'sources':{},'checkpoint_metadata':{},'devices':{}}
    class Session:
        def __init__(self,*args,**kwargs):pass
        def close(self):pass
    class Policy:
        _policy_type='groot';_checkpoint_path='/checkpoints/model';_actions_per_chunk=16;_device='cuda'
        def __init__(self,**kwargs):
            self._session=Session();self._handshaken=True;self.language_instruction=api.safety.INSTRUCTION
        def get_action(self,*args):
            if killed.is_set():raise RuntimeError('injected server loss')
            return [dict.fromkeys(api.safety.JOINT_ORDER,1.) for _ in range(16)]
        def close(self):pass
    class Controller:
        def __init__(self,stop,**kwargs):self.stop=stop;self.resets=[];self.targets=[];self.reads=0;controllers.append(self)
        def connect(self,**kwargs):pass
        def disconnect(self):pass
        def move_to_initial_pose(self):self.resets.append('initial')
        def move_to_ready_pose(self):self.resets.append('ready')
        def get_observation(self):return {}
        def get_current_images(self):return {}
        def get_current_state(self):self.reads+=1;return [1.]*6
        def set_target_state(self,target):
            with self.stop.dispatch('fake_target'):self.targets.append(target)
            return dict(target)
    monkeypatch.setattr(backend,'SerializedLeRobotPolicyBackend',Policy)
    monkeypatch.setattr(session,'LeRobotPolicySession',Session)
    monkeypatch.setattr(api.safety,'StopGuardedController',Controller)
    monkeypatch.setattr(api.safety,'validate_loaded_calibration',lambda *a:None)
    monkeypatch.setattr(api,'snapshot',lambda:current)
    monkeypatch.setattr(api,'validate_approval',lambda *a:None)
    monkeypatch.setattr(api,'probe_cameras',lambda s:{})
    monkeypatch.setattr(api,'serving_binding',lambda:{})
    monkeypatch.setattr(api,'sources',lambda:{})
    monkeypatch.setattr(api,'devices',lambda s:{})
    monkeypatch.setattr(api,'checkpoint_metadata',lambda:{})
    monkeypatch.setattr(api,'sha256_file',lambda p:'cal')
    monkeypatch.setattr(driver,'load_settings',lambda p:AsyncSettings(.15))
    monkeypatch.setattr(api,'read_json',lambda p:{})
    monkeypatch.setattr(api,'load_case',lambda *a:({'state':np.ones(6),'video_front':np.zeros((2,2,3)),'video_wrist':np.zeros((2,2,3))},{'instruction':'banana'}))
    monkeypatch.setattr(sys,'stdin',SimpleNamespace(isatty=lambda:True))
    monkeypatch.setattr(api.subprocess,'check_output',lambda *a,**kw:'SAFE-01 guard: PASS "observer_mode": "lightweight"')
    def kill(cmd,**kw):
        assert cmd==['docker','kill',api.CONTAINER]
        killed.set();return SimpleNamespace(returncode=0)
    monkeypatch.setattr(api.subprocess,'run',kill)
    (tmp_path/'approval.json').write_text('{}')
    assert api.run(tmp_path)==0
    result=json.loads((tmp_path/'live-run.json').read_text())
    assert result['status']=='awaiting_operator_hold_observation'
    assert result['dispatches_after_stop']==0 and result['post_fault_reset_refused']
    assert result['detection_after_kill_s']<=result['declared_bound_s']
    assert len(result['hold_samples'])==40
    assert controllers[0].resets==['initial','ready']
    assert 4<=len(controllers[0].targets)<32
    assert result['voice_status']=='not_connected_callback_only'
