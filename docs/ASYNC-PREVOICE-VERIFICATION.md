# Pre-voice verification — 2026-09-12

The focused pre-voice suite passed **149 tests in 23.57 seconds**, with one existing
audioop deprecation warning. Aaron explicitly deferred live voice integration to
a separate session. No physical motion was performed for these checks.

Three integration scenarios use a real loopback MCP HTTP server, shared-memory
task/message storage, the agent worker, agent streaming wrapper, decorated pick
tool and async scheduler. Hardware and policy responses are simulated; a scripted
model selects the pick tool. This checks orchestration, not live LLM tool selection
or speech recognition/synthesis.

- Inference failure reaches task status, broker events and the MCP result. No
  successful completion or post-fault reset occurs.
- Retargeting reaches the running policy and changes its targets without creating
  a second task or closing the policy connection.
- Cancellation stops dispatch and refuses reset. The test exposed cancellation
  being overwritten as failure; the agent now preserves CANCELLED and MCP reports
  cancellation explicitly.
- MCP status requests complete while inference is active; blocking controller
  calls run off the main thread.
- Empty, missing-task and terminal-task retarget requests are rejected.

The remaining focused tests cover staleness/deadlines, action validation, anchor
serialization, observer compatibility, controller stop harnesses, agent tools,
MCP interfaces and existing voice-adapter unit contracts. Existing physical pick,
server-loss hold and GPU results remain separately recorded evidence.

Evidence: `corpus/phase8-prevoice-integration-20260912/result.json` binds the tested
source hashes and `pytest.log`. Full audible failure notification, spoken retargeting,
live voice responsiveness, and formal Phase 8 verification remain pending.
