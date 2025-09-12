## Tests

This suite validates the shared interfaces, the MCP server tools, and the orchestrated server/agent flow without requiring robot hardware. All tests are robot‑agnostic unless explicitly using the mock worker via the orchestrator helpers.

### Test layout and scope

- `test_shm_broker.py`
  - Scope: `SharedMemoryMessageBroker` publish/subscribe/history semantics
  - What it does: allocates SHM ring buffer; subscribes to `TASK_PROGRESS`; publishes messages; asserts filtered receipt and bounded history

- `test_shm_task_manager.py`
  - Scope: `SharedMemoryTaskManager` lifecycle and atomic claim
  - What it does: creates a task; claims it; updates progress; completes it; verifies via `get_task` and `list_tasks`

- `test_agent_interfaces.py`
  - Scope: `IRobotAgent` integration using `MockRobotAgent`
  - What it does: runs `MockRobotAgent.astream` with SHM backends; subscribes to broker; asserts warmup and assistant messages; verifies task completion state

- `test_orchestrator.py`
  - Scope: Orchestrator process spawning and HTTP MCP integration with mock worker
  - What it does: validates env injection in spawned processes; spins up MCP server, mock worker, and Pipecat; uses `fastmcp.Client` over HTTP to call `execute_robot_instruction`; captures and asserts progress events

- `test_mcp_interfaces.py`
  - Scope: MCP tools exposed by `mcp_server.py` using SHM backends only (no worker)
  - What it does: starts only the MCP HTTP server; seeds tasks/messages/fleet directly via SHM TaskManager/Broker/FleetManager; uses `fastmcp.Client` to call tools and verifies responses:
    - `list_namespaces` returns the active namespace
    - `list_tasks` supports empty and filtered views
    - `get_task_details` optionally includes recent messages
    - `cancel_task` updates task state and acknowledges success
    - `list_last_messages` filters by `task_id` and `message_types`
    - `register_robot` upserts a robot
    - `list_robots` lists robots
    - `get_robot` gets the robot details by ID
    - `set_robot_enabled` enables or disables a robot by ID

### Running tests

```bash
pytest -q
```

### Shared memory setup

For unit tests that directly attach to SHM (not via orchestrator), the tests create `ShareableList` buffers using `SharedMemoryManager` and pass their names via env vars:

- `DUME_BROKER_BUF`, `DUME_BROKER_META` – broker buffer and write index
- `DUME_TASKS_BUF` – task storage buffer
- `DUME_FLEET_BUF` – fleet registry buffer
- `DUME_NAMESPACE` – namespace for lock files
- `DUME_IPC=shm` – signals SHM backend selection

### Notes

- The mock agent avoids any model or hardware dependencies while exercising the same interfaces and message flows.
- `mcp_server.py` logs task lifecycle and progress forwarding for easier debugging.