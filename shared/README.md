## Shared interfaces and backends

The shared module defines transport-agnostic interfaces for Dum‑E components and their implementations for efficient cross‑process communication.

### Interfaces (`shared/__init__.py`)

- `IRobotAgent`: async agent contract with `arun`, `astream`, `get_available_tools`, `get_status`.
- `ITaskManager`: task lifecycle with `create_task`, `get_task`, `update_task`, `list_tasks`, `cancel_task`, `claim_task`.
- `IMessageBroker`: pub/sub for streaming events with `publish`, `subscribe`, `get_message_history`.
- Data models: `TaskInfo`, `TaskStatus`, `Message`, `MessageType`, `ToolDefinition`, `BackendConfig`.

These abstractions decouple the Dum‑E MCP server and agent so they can run in different processes or machines while sharing the same contract.

### Shared Memory backends

Implemented for Python 3.12 using `multiprocessing.shared_memory.ShareableList` and lightweight file locks.

- `shared/message_broker/shm.py`
  - `SharedMemoryMessageBroker(namespace, buffer_name, meta_name, poll_interval_s)`
    - Ring buffer of JSON strings in a `ShareableList` (buffer) and a single integer `write_index` (meta).
    - `publish` appends with a file lock; `subscribe` polls `write_index` and yields new entries; `get_message_history` scans backwards up to a limit.
  - `get_shared_memory_broker_from_env()` attaches from env.

- `shared/task_manager/shm.py`
  - `SharedMemoryTaskManager(namespace, tasks_name)`
    - Fixed‑size `ShareableList` storing JSON‑encoded `TaskInfo` records.
    - Implements the full `ITaskManager` API including atomic `claim_task` guarded by a file lock.
  - `get_shared_memory_task_manager_from_env()` attaches from env.

#### Environment variables

These are set by the `dum_e.py` when `DUME_IPC=shm`:

- `DUME_NAMESPACE`: logical namespace (included in lock filenames).
- `DUME_BROKER_BUF`, `DUME_BROKER_META`: buffer and meta SHM names for broker.
- `DUME_TASKS_BUF`: tasks SHM name for task manager.
- Optional capacity/slot sizing are internal to the dum_e; processes only need SHM names.

To attach inside a process:

```python
from shared.message_broker.shm import get_shared_memory_broker_from_env
from shared.task_manager.shm import get_shared_memory_task_manager_from_env

MESSAGE_BROKER = get_shared_memory_broker_from_env()
TASK_MANAGER = get_shared_memory_task_manager_from_env()
```

### Server and agent usage

- The MCP server (`mcp_server.py`) creates a task via `ITaskManager`, publishes `TASK_STARTED` via `IMessageBroker`, and streams progress to MCP clients through `ctx.report_progress` as broker events arrive.
- The worker loop (real agent or mock) subscribes to `TASK_STARTED`, atomically claims a task via `claim_task`, executes `IRobotAgent.astream`, and publishes `TASK_PROGRESS`/`TASK_COMPLETED` or `TASK_FAILED`.

### Performance & safety notes

- File locks serialize writers while subscribers poll without locks for simplicity and robustness.
- Ring buffers are bounded; history is truncated by capacity. Choose capacities in the dum_e to match workload.
- SHM backends are process‑local; for OTA, plug in networked implementations (e.g., MQTT/DynamoDB) behind the same interfaces.

### Roadmap

Add new factories (e.g., `get_mqtt_message_broker_from_config`, `get_dynamodb_task_manager_from_config`) and select them by config/env. The server and agent code remain unchanged because they depend only on `IMessageBroker` and `ITaskManager`.


