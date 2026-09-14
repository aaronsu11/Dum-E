from concurrent.futures import Future
import pytest
from policy.execution.asynchronous import AsyncChunks,AsyncSettings,InferenceStopped,ChunkPending


class Executor:
    def __init__(self):self.calls=[]
    def submit(self,fn,*args):
        future=Future();self.calls.append((future,fn,args));return future


def fixture():
    clock=[0.];executor=Executor()
    queue=AsyncChunks(lambda obs,task: None,['joint'],AsyncSettings(.15),'banana',executor=executor,clock=lambda:clock[0])
    return queue,executor,clock


def actions(value=1.):return [{'joint':value} for _ in range(16)]


def test_prefetch_and_indexed_blending_drop_elapsed_actions():
    q,e,c=fixture();q.request('obs0',0);e.calls[0][0].set_result(actions());q.poll(0)
    for step in range(8):q.dispatch(q.next_action(step),lambda a:a)
    assert q.wants_observation(8)
    c[0]=.4;q.request('obs8',8);c[0]=.5
    e.calls[1][0].set_result(actions(3.));q.poll(10)
    assert q.next_action(10).target['joint']==pytest.approx(2.4)
    assert q.next_action(16).target['joint']==3.


def test_deadline_stops_even_with_actions_left_and_latches():
    q,e,c=fixture();q.request({},0);e.calls[0][0].set_result(actions());q.poll(0)
    command=q.next_action(0);q.request({},8);c[0]=.251;sent=[]
    with pytest.raises(InferenceStopped,match='deadline'):q.dispatch(command,sent.append)
    assert not sent
    with pytest.raises(InferenceStopped):q.set_task('apple')


def test_retarget_discards_old_future_and_uses_new_instruction():
    q,e,c=fixture();q.request('old',0);q.set_task('apple')
    e.calls[0][0].set_result(actions());q.poll(0)
    assert q.wants_observation(0)
    q.request('new',0);assert e.calls[1][2]==('new','apple')
    with pytest.raises(ChunkPending):q.next_action(0)
    e.calls[1][0].set_result(actions(2.));assert q.next_action(0).target=={'joint':2.}


def test_retarget_between_dequeue_and_dispatch_cannot_send_old_action():
    q,e,c=fixture();q.request({},0);e.calls[0][0].set_result(actions());command=q.next_action(0)
    q.set_task('apple');sent=[]
    with pytest.raises(ChunkPending):q.dispatch(command,sent.append)
    assert sent==[]
    assert q.wants_observation(0)


def test_server_failure_stops_queued_actions():
    q,e,c=fixture();q.request({},0);e.calls[0][0].set_result(actions());q.poll(0)
    command=q.next_action(0);q.request({},8);e.calls[1][0].set_exception(RuntimeError('server gone'))
    with pytest.raises(InferenceStopped,match='server gone'):q.dispatch(command,lambda a:pytest.fail('sent'))


def test_stale_and_nonfinite_actions_are_refused():
    q,e,c=fixture();q.request({},0);e.calls[0][0].set_result(actions());q.poll(0);c[0]=.801
    with pytest.raises(InferenceStopped,match='stale'):q.next_action(0)
    q,e,c=fixture();q.request({},0);e.calls[0][0].set_result(actions(float('nan')))
    with pytest.raises(InferenceStopped,match='nonfinite'):q.poll(0)


def test_wrong_horizon_and_latency_without_overlap_refused():
    with pytest.raises(ValueError):AsyncSettings(.35)
    with pytest.raises(ValueError):AsyncSettings(.15,actions_per_chunk=40)
    q,e,c=fixture();q.request({},0);e.calls[0][0].set_result(actions()[:15])
    with pytest.raises(InferenceStopped,match='16'):q.poll(0)


def test_second_request_cannot_overlap():
    q,e,c=fixture();q.request({},0)
    with pytest.raises(RuntimeError,match='one request'):q.request({},1)
    assert len(e.calls)==1


def test_expiring_old_blend_is_dropped_at_dispatch_not_fresh_chunk():
    q,e,c=fixture();q.request({},0);e.calls[0][0].set_result(actions(1.));q.poll(0)
    c[0]=.65;q.request({},8);c[0]=.75;e.calls[1][0].set_result(actions(3.));q.poll(10)
    command=q.next_action(10);assert command.target['joint']==pytest.approx(2.4)
    c[0]=.81;sent=[]
    q.dispatch(command,lambda target: sent.append(target) or target)
    assert sent==[{'joint':3.}]


def test_changed_target_latches_before_next_dispatch():
    q,e,c=fixture();q.request({},0);e.calls[0][0].set_result(actions())
    command=q.next_action(0)
    with pytest.raises(InferenceStopped,match='clamped'):q.dispatch(command,lambda target:{'joint':.5})
    with pytest.raises(InferenceStopped):q.next_action(1)


def test_unclipped_upstream_roundtrip_is_not_a_clamp():
    from lerobot.robots.utils import ensure_safe_goal_position
    q,e,c=fixture();q.request({},0);e.calls[0][0].set_result(actions(.1))
    def send(target):
        result=ensure_safe_goal_position({'joint':(target['joint'],10.)},160.)
        assert result['joint'] != target['joint']  # IEEE rounding, no clipping
        return result
    q.dispatch(q.next_action(0),send)
    q.next_action(1)


@pytest.mark.parametrize('returned',[float('nan'),float('inf'),True,None,'1',1.0002])
def test_invalid_or_materially_changed_return_stops(returned):
    q,e,c=fixture();q.request({},0);e.calls[0][0].set_result(actions())
    with pytest.raises(InferenceStopped):q.dispatch(q.next_action(0),lambda target:{'joint':returned})
