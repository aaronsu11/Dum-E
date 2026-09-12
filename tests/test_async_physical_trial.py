import copy
import pytest
from scripts import run_async_physical_trial as api


def approved():
    return {'kind':'async_single_physical_trial','approved':True,'operator':'Aaron',
            'operator_present':True,'user_authorization':'Physical trial; I’m beside the arm',
            'protocol':copy.deepcopy(api.PROTOCOL),'snapshot':{'test':'snapshot'},'approved_at':api.now()}


def test_explicit_single_trial_scope_passes():
    api.validate_approval(approved(),{'test':'snapshot'})


@pytest.mark.parametrize('field,value',[('operator_present',False),('operator',''),('approved',False),
                                       ('approved_at','2026-01-01T00:00:00Z')])
def test_missing_presence_authority_or_freshness_refused(field,value):
    record=approved();record[field]=value
    with pytest.raises(Exception):api.validate_approval(record,{'test':'snapshot'})


def test_changed_sources_or_trial_budget_refused():
    with pytest.raises(Exception):api.validate_approval(approved(),{'test':'changed'})
    record=approved();record['protocol']['trials']=2
    with pytest.raises(Exception):api.validate_approval(record,{'test':'snapshot'})


def test_old_observer_approval_cannot_release_async():
    record=approved();record['kind']='observer_off_single_physical_trial'
    with pytest.raises(Exception):api.validate_approval(record,{'test':'snapshot'})


def test_bounded_async_protocol():
    assert api.PROTOCOL['chunks']*api.PROTOCOL['actions_per_chunk']==320
    assert api.PROTOCOL['scheduler']=='async'
    assert api.PROTOCOL['observer_mode']=='lightweight'
    assert api.PROTOCOL['trace_states'] is True
    assert api.PROTOCOL['server_fault_injection'] is False
