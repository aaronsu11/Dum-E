import copy
import pytest
from scripts import run_voice_physical_trial as voice


def test_voice_trial_scope_and_approval_are_distinct():
    p=voice.physical
    assert p.PROTOCOL['chunks']*p.PROTOCOL['actions_per_chunk']==320
    assert p.PROTOCOL['retarget_allowed'] is False
    assert p.PROTOCOL['server_fault_injection'] is False
    approval={'kind':'async_single_physical_trial','approved':True,'operator':'Aaron','operator_present':True,
              'user_authorization':'one voice-triggered physical banana pick','approved_at':p.now(),
              'protocol':copy.deepcopy(p.PROTOCOL),'snapshot':{}}
    p.validate_approval(approval,{})
    approval['protocol'].pop('trigger')
    with pytest.raises(Exception):p.validate_approval(approval,{})


@pytest.mark.parametrize('text,expected',[('Pick up the banana and deliver it',True),('grab banana',True),
                                        ('do not pick the banana',False),('cancel banana pick',False),('pick an apple',False),('run failure notification test',False),('',False)])
def test_only_banana_pick_can_release_runner(text,expected):
    assert voice.accepts_instruction(text)==expected
