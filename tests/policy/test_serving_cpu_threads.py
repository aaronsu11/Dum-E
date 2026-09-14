"""Startup thread configuration; no model, server, or hardware construction."""
import sys
from types import SimpleNamespace

import pytest
from tests.policy.test_container_contract import _load_container_entrypoint


@pytest.mark.parametrize("env,argv,expected", [
    (None, [], 1), ("2", [], 2), ("20", ["--cpu-threads", "1"], 1),
])
def test_threads_applied_before_preflight(monkeypatch, capsys, env, argv, expected):
    entry = _load_container_entrypoint()
    if env is None:
        monkeypatch.delenv("DUME_POLICY_CPU_THREADS", raising=False)
    else:
        monkeypatch.setenv("DUME_POLICY_CPU_THREADS", env)
    calls = []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        set_num_threads=lambda n: calls.append(n), get_num_threads=lambda: calls[-1],
        get_num_interop_threads=lambda: 20))
    def preflight(*args):
        assert calls == [expected]
        return 1
    monkeypatch.setattr(entry, "run_preflight", preflight)
    monkeypatch.setattr(entry, "DumEGrootPolicyServer", lambda *args: pytest.fail("server must not start"))
    monkeypatch.setattr(sys, "argv", ["entrypoint.py", *argv])
    assert entry.main() == 1
    assert f"intra_op={expected}" in capsys.readouterr().out


@pytest.mark.parametrize("value", ["0", "-1", "many", "1.5"])
def test_invalid_threads_refused_before_preflight(monkeypatch, value):
    entry = _load_container_entrypoint()
    monkeypatch.setenv("DUME_POLICY_CPU_THREADS", value)
    monkeypatch.setattr(entry, "run_preflight", lambda *args: pytest.fail("invalid setting reached preflight"))
    monkeypatch.setattr(sys, "argv", ["entrypoint.py"])
    with pytest.raises(SystemExit) as exc:
        entry.main()
    assert exc.value.code == 2
