"""Physical closeout budget/judgment checks; no model or hardware."""
import pytest
from policy_guard.milestone_closeout import validate_budget


def trial():
    return dict(index=2, iterations=20, actions_per_chunk=16, action_delay=.05,
                safety_stop=False, clamp_warnings=0, coherent=True,
                wrong_target=False, erratic=False, grasp=True, operator='Aaron')


def test_grasp_is_optional_for_directional_success():
    value=trial();value['grasp']=False
    assert validate_budget(value,2)==1


@pytest.mark.parametrize('changes', [
    {'index':1}, {'iterations':19}, {'iterations':21}, {'actions_per_chunk':15},
    {'action_delay':.1}, {'safety_stop':True}, {'clamp_warnings':1},
    {'clamp_warnings':False}, {'coherent':'yes'}, {'operator':''},
])
def test_invalid_trial_cannot_close(changes):
    value=trial();value.update(changes)
    with pytest.raises(ValueError):validate_budget(value,2)


@pytest.mark.parametrize('changes',[{'coherent':False},{'wrong_target':True},{'erratic':True}])
def test_directional_failures_do_not_count_as_success(changes):
    value=trial();value.update(changes)
    assert validate_budget(value,2)==0
