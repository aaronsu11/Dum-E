"""Exercise the wrapper's fail-closed AR boundary without loading a checkpoint."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def runtime(monkeypatch, generated):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "policy"))
    spec = importlib.util.spec_from_file_location(
        "galaxea_server_budget_test", root / "policy/backends/galaxea/server.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    def infer(*args, **kwargs):
        calls.append(kwargs["max_new_tokens"])
        return {"generated_ids": np.array([generated], dtype=np.int64)}

    helper = SimpleNamespace(infer=infer, max_new_tokens=1000, eov_token_id=99)
    model = SimpleNamespace(ar_helper=helper, cfg=SimpleNamespace(eos_token_id=99))
    obj = module.Runtime.__new__(module.Runtime)
    obj.policy = SimpleNamespace(model=model)
    obj.tokens_generated = 0
    obj._bound_ar_generation()
    return obj, model, calls


def test_token_limit_refuses_incomplete_generation(monkeypatch):
    obj, model, calls = runtime(monkeypatch, [1] * 300)
    with pytest.raises(TimeoutError, match="Incomplete"):
        model.ar_helper.infer(model)
    assert calls == [300]


def test_budget_is_shared_across_generation_stages(monkeypatch):
    obj, model, calls = runtime(monkeypatch, [1] * 149 + [99])
    model.ar_helper.infer(model)
    model.ar_helper.infer(model)
    with pytest.raises(TimeoutError, match="exhausted"):
        model.ar_helper.infer(model)
    assert calls == [300, 150]
    assert obj.tokens_generated == 300
