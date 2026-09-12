from concurrent.futures import ThreadPoolExecutor
import time
from policy.lerobot.serialized_backend import SerializedLeRobotPolicyBackend
from policy.lerobot.backend import LeRobotPolicyBackend


def test_two_overlapping_calls_keep_their_own_observation(monkeypatch):
    def init(self,**kwargs):
        self._device='cuda';self._actions_per_chunk=16;self._language_instruction='banana'
    def infer(self,observation,lang):
        self.anchor=observation
        time.sleep(.01)
        return self.anchor,lang
    monkeypatch.setattr(LeRobotPolicyBackend,'__init__',init)
    monkeypatch.setattr(LeRobotPolicyBackend,'get_action',infer)
    policy=SerializedLeRobotPolicyBackend(host='127.0.0.1')
    with ThreadPoolExecutor(max_workers=2) as pool:
        a=pool.submit(policy.get_action,'state A','banana')
        b=pool.submit(policy.get_action,'state B','apple')
        assert a.result()==('state A','banana')
        assert b.result()==('state B','apple')
