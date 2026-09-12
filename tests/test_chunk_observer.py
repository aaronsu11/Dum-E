import pytest
import torch
from types import SimpleNamespace
from policy_guard.chunk_observer import ChunkObserver


def model():
    return SimpleNamespace(backbone=torch.nn.Identity(), action_head=SimpleNamespace(action_encoder=torch.nn.Identity()))


def predict(m, value):
    value=m.backbone(value)
    for _ in range(4):
        value=m.action_head.action_encoder(value)
    return value


@pytest.mark.parametrize('mode', ['off', 'lightweight'])
def test_identity_rng_and_hook_cleanup(mode):
    m=model();value=torch.ones(1, 2);rng=torch.get_rng_state().clone()
    observer=ChunkObserver(mode)
    assert observer.run(m, predict, m, value) is value
    assert torch.equal(rng, torch.get_rng_state())
    assert not m.backbone._forward_hooks and not m.backbone._forward_pre_hooks
    assert not m.action_head.action_encoder._forward_pre_hooks
    assert observer.last['exhaustive_attestation'] is False
    assert observer.last['status']=='complete'
    if mode=='lightweight':
        assert observer.last['flow_steps']==4
        assert observer.last['backbone_outputs']['shape']==[1, 2]
    else:
        assert 'flow_steps' not in observer.last


def test_exception_cleans_hooks_and_is_not_success():
    m=model();o=ChunkObserver()
    def fail():raise RuntimeError('prediction failed')
    with pytest.raises(RuntimeError, match='prediction failed'):o.run(m, fail)
    assert o.last['status']=='failed'
    assert not m.backbone._forward_hooks and not m.backbone._forward_pre_hooks
    assert not m.action_head.action_encoder._forward_pre_hooks


def test_invalid_mode_refused():
    with pytest.raises(ValueError):ChunkObserver('typo')


def test_keyword_inputs_and_mapping_outputs_are_observed():
    from collections import UserDict
    class Backbone(torch.nn.Module):
        def forward(self, *, pixel_values):
            return UserDict(features=pixel_values)
    m=model();m.backbone=Backbone();value=torch.ones(1, 3)
    o=ChunkObserver();result=o.run(m, lambda: m.backbone(pixel_values=value))
    assert result['features'] is value
    assert o.last['backbone_inputs']['kwargs']['pixel_values']['shape']==[1, 3]
    assert o.last['backbone_outputs']['features']['dtype']=='torch.float32'


def test_native_duck_typed_batch_feature():
    from policy_guard.chunk_observer import tensor_metadata
    class Batch:
        def items(self):return {'pixel_values': torch.ones(1, 3), 'label': 'ignored'}.items()
    assert tensor_metadata((Batch(),))[0]['pixel_values']['shape']==[1, 3]


@pytest.mark.parametrize('mode', ['off', 'lightweight'])
def test_lerobot_entrypoint_routes_prediction_through_shared_observer(monkeypatch, mode):
    import sys
    from scripts import serve_observed_lerobot as script
    m=model();value=torch.ones(1);logs=[]
    class Server:
        def __init__(self):
            self.policy=SimpleNamespace(_groot_model=m)
            self.logger=SimpleNamespace(info=lambda *args: logs.append(args))
        def _predict_action_chunk_impl(self, observation):
            return predict(m, observation)
    entry=SimpleNamespace(DumEGrootPolicyServer=Server)
    def serve():
        instance=entry.DumEGrootPolicyServer()
        assert instance._predict_action_chunk(value) is value
        instance._parity_attestor=object()
        with pytest.raises(RuntimeError, match='cannot be bypassed'):
            instance._predict_action_chunk(value)
        return 0
    entry.main=serve
    monkeypatch.setitem(sys.modules, 'entrypoint', entry)
    monkeypatch.setenv('DUME_CHUNK_OBSERVER', mode)
    monkeypatch.delenv('DUME_PARITY_ATTESTATION_PATH', raising=False)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    assert script.main()==0
    assert len(logs)==1 and f'"observer_mode": "{mode}"' in logs[0][-1]


def test_lerobot_refuses_attestation_downgrade(monkeypatch):
    from scripts import serve_observed_lerobot as script
    monkeypatch.setenv('DUME_PARITY_ATTESTATION_PATH', '/evidence/runtime.json')
    with pytest.raises(RuntimeError, match='cannot replace exhaustive'):
        script.main()


@pytest.mark.parametrize('mode', ['off', 'lightweight'])
def test_native_endpoint_uses_shared_observer(monkeypatch, tmp_path, mode):
    import sys
    from contextlib import nullcontext
    from scripts import serve_observed_native as script
    from scripts import replay_groot_native
    m=model();value=torch.ones(1);calls=[]
    class Policy:
        def __init__(self, *a, **kw):
            assert kw['device']=='cuda:0' and kw['strict'] is True
            self.model=m
        def get_action(self, observation, options=None):
            calls.append(options)
            return predict(m, observation), {'original': True}
    class Server:
        def __init__(self, policy, **kw):self.policy=policy
        def register_endpoint(self, name, endpoint):
            assert name=='get_action';self.endpoint=endpoint
        def run(self):
            output, info=self.endpoint(observation=value, options={'test': 1})
            assert output is value and info=={'original': True}
    for name, module in {
        'gr00t': SimpleNamespace(), 'gr00t.data': SimpleNamespace(),
        'gr00t.data.embodiment_tags': SimpleNamespace(EmbodimentTag=SimpleNamespace(NEW_EMBODIMENT='new_embodiment')),
        'gr00t.policy': SimpleNamespace(),
        'gr00t.policy.gr00t_policy': SimpleNamespace(Gr00tPolicy=Policy),
        'gr00t.policy.server_client': SimpleNamespace(PolicyServer=Server),
    }.items():monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(replay_groot_native, 'pinned_native_cache', nullcontext)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(torch, 'set_num_threads', lambda n: None)
    monkeypatch.setattr(sys, 'argv', ['serve', '--model-path', str(tmp_path), '--observer-mode', mode])
    script.main()
    assert calls==[{'test': 1}]
