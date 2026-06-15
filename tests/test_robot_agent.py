"""
Mocked-wiring async tests for the SO10x robot agent (Phase 03, GR-06 / D-13).

These tests are keyless and hardware-free. They pin the three client surfaces
hardened in plan 03-02:

1. Robot ``@tool`` adapters offload the blocking ``Skill`` via
   ``asyncio.to_thread`` (the event-loop-responsiveness fix, threat T-03-04).
2. ``get_available_tools()`` returns a real ``List[ToolDefinition]`` (GR-05).
3. ``get_status()`` does NOT await the synchronous ``is_connected()`` (D-12 /
   threat T-03-06).

Import note: ``embodiment.so_arm10x.controller`` imports
``policy.gr00t.service.ExternalRobotInferenceClient`` which transitively imports
the external ``gr00t`` package (only present on the policy-server host) and a
symbol owned by the parallel plan 03-01. To keep this suite runnable in CI we
stub those modules in ``sys.modules`` BEFORE importing the agent. This is
mocked-wiring, not a behavioral change to production code.
"""

import sys
import types
from unittest.mock import Mock

import pytest


# ---------------------------------------------------------------------------
# Stub the unavailable external/parallel-plan import chain before importing the
# agent. `controller.py` does `from policy.gr00t.service import
# ExternalRobotInferenceClient`, which imports `gr00t.*`. Neither is available
# in a keyless/no-hardware CI env.
# ---------------------------------------------------------------------------
def _install_import_stubs():
    # Stub the gr00t package tree used by policy.gr00t.service.
    for mod_name in ("gr00t", "gr00t.data", "gr00t.data.types", "gr00t.data.utils"):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)
    sys.modules["gr00t.data.types"].ModalityConfig = object
    sys.modules["gr00t.data.utils"].to_json_serializable = lambda x: x

    # Ensure policy.gr00t.service exposes ExternalRobotInferenceClient even if
    # the real module fails to import (gr00t missing) or the symbol is owned by
    # the parallel plan 03-01 and not yet present in this worktree.
    try:
        import policy.gr00t.service as svc  # noqa: F401

        if not hasattr(svc, "ExternalRobotInferenceClient"):
            svc.ExternalRobotInferenceClient = Mock
    except Exception:
        svc = types.ModuleType("policy.gr00t.service")
        svc.ExternalRobotInferenceClient = Mock
        # Make sure parent packages exist for the dotted import to resolve.
        for parent in ("policy", "policy.gr00t"):
            if parent not in sys.modules:
                sys.modules[parent] = types.ModuleType(parent)
        sys.modules["policy.gr00t.service"] = svc


_install_import_stubs()

# Now the agent (and its controller/skills imports) resolve cleanly.
from embodiment.so_arm10x.agent import (  # noqa: E402
    SO10xRobotAgent,
    create_robot_tools,
)
from embodiment.so_arm10x.skills import PickSkill  # noqa: E402
from shared import ToolDefinition  # noqa: E402


def _make_controller_mock():
    """A controller mock whose camera images are JPEG-encodable numpy arrays."""
    import numpy as np

    controller = Mock()
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    controller.get_current_images.return_value = {"front": img, "wrist": img}
    controller.id = "test_arm"
    return controller


@pytest.mark.asyncio
async def test_start_pick_offloads_to_thread(mocker):
    """The start_pick @tool must run the blocking Skill via asyncio.to_thread."""
    import asyncio

    import numpy as np

    controller = _make_controller_mock()
    gr00t_client = Mock()

    tools = create_robot_tools(controller, gr00t_client)
    tools_by_name = {t.tool_name: t for t in tools}
    start_pick = tools_by_name["start_pick"]

    # Spy on asyncio.to_thread so we can assert the offload happened, while
    # forcing the Skill.run to return a known raw-images dict.
    known_images = {"front": np.ones((4, 4, 3), dtype=np.uint8)}

    async def fake_to_thread(func, *args, **kwargs):
        # Record + short-circuit the real (blocking) Skill body.
        return known_images

    spy = mocker.patch("asyncio.to_thread", side_effect=fake_to_thread)

    # Invoke the underlying async tool function directly.
    result = await start_pick._tool_func(item="a banana")

    assert spy.await_count == 1 or spy.call_count == 1
    # The Skill bound in the offloaded call is the PickSkill.
    offloaded_callable = spy.call_args.args[0]
    assert offloaded_callable.__self__.__class__ is PickSkill
    # The adapter still builds the preserved response shape from the raw images.
    assert result["status"] == "success"
    assert any("image" in c for c in result["content"])
    # The language instruction was set on the client before offloading.
    gr00t_client.set_lang_instruction.assert_called_once()


@pytest.mark.asyncio
async def test_get_available_tools_returns_tooldefinitions(mocker):
    """get_available_tools returns a non-empty List[ToolDefinition] (GR-05)."""
    controller = _make_controller_mock()
    gr00t_client = Mock()

    agent = SO10xRobotAgent(
        robot_controller=controller,
        gr00t_client_instance=gr00t_client,
        task_manager=None,
        message_broker=None,
    )

    tools = await agent.get_available_tools()

    assert isinstance(tools, list) and len(tools) > 0
    assert all(isinstance(t, ToolDefinition) for t in tools)
    names = {t.name for t in tools}
    assert {"start_pick", "resume_pick", "place"}.issubset(names)
    for t in tools:
        assert t.category == "robot"
        assert t.requires_hardware is True
        assert callable(t.function)
        assert isinstance(t.parameters_schema, dict)


@pytest.mark.asyncio
async def test_get_status_does_not_await_is_connected():
    """get_status must call the SYNC is_connected() without awaiting it (D-12)."""
    from unittest.mock import AsyncMock

    controller = _make_controller_mock()
    # SYNC mock: if get_status awaited this, awaiting a bool would raise.
    controller.is_connected = Mock(return_value=True)
    gr00t_client = Mock()

    task_manager = Mock()
    task_manager.list_tasks = AsyncMock(return_value=[])
    message_broker = Mock()
    message_broker.get_message_history = AsyncMock(return_value=[])

    agent = SO10xRobotAgent(
        robot_controller=controller,
        gr00t_client_instance=gr00t_client,
        task_manager=task_manager,
        message_broker=message_broker,
    )

    status = await agent.get_status()

    assert status["status"] == "healthy"
    assert status["controller_connected"] is True
    controller.is_connected.assert_called_once_with()
    # The genuinely-async deps were awaited.
    task_manager.list_tasks.assert_awaited_once()
    message_broker.get_message_history.assert_awaited_once()
