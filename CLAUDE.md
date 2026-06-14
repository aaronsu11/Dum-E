<!-- GSD:project-start source:PROJECT.md -->

## Project

**Dum-E — Latest Pipecat / Deepgram / GR00T Upgrade**

Dum-E is a voice- and vision-enabled embodied AI agent that drives an SO-ARM10x robot
arm through real-time conversation, long-horizon task planning, and hybrid robot control
(VLA policies + classical control). This milestone upgrades the three external pillars the
system depends on — the **Pipecat** voice pipeline, the **Deepgram** speech models, and the
**Isaac GR00T** VLA policy — to their latest stable releases and best practices, and packages
the GR00T inference server into a multi-arch (x86 + ARM/Jetson Orin) Docker container that is
verified end-to-end on real hardware.

**Core Value:** After the upgrade, a user can speak to Dum-E and have it pick up an object using the GR00T
N1.7 3B policy running inside the containerized inference server — the full voice → agent →
policy → robot loop works on the latest stack, on both x86 and Jetson Orin.

### Constraints

- **Tech stack**: Python 3.12, `uv` package manager, `pyproject.toml` + `uv.lock` — keep reproducible.
- **Compatibility**: Pipecat version and `pipecat-ai-small-webrtc-prebuilt` must stay matched (`/start` endpoint).
- **Hardware (verification)**: x86 + NVIDIA GPU available AND a physical Jetson Orin available — Docker
  work must be verified live on both, not build-only.

- **Profiles**: `aws` profile and existing multi-language behavior must keep working through the upgrade.
- **GR00T**: follow official GR00T n1.7-release guidance/recommendations wherever it exists.
- **Deepgram SageMaker**: support the backend path but do not deploy the endpoint this milestone.
- **Process**: prefer smaller-model subagents for research/verification where it doesn't cost quality.

<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->

## Technology Stack

## Languages

- Python 3.12 - Full stack robotics, voice, and agent implementation
- YAML - Configuration files (`config.example.yaml`, `my-dum-e.yaml`)
- ZMQ (via pyzmq) - Inter-process communication for inference clients
- MessagePack - Binary serialization for complex data (numpy arrays, structured messages)

## Runtime

- Python 3.12 (specified in `.python-version`)
- Virtual environment: `.venv/` (pip/uv-managed)
- uv 0.10.4+ - Fast Python package manager
- Lockfile: `uv.lock` (978KB, contains full dependency graph)

## Frameworks

- Pipecat 0.0.104 - Voice assistant pipeline orchestration
- FastAPI 0.129.0+ - HTTP server for voice pipeline (`pipecat_server.py`)
- FastMCP 2.14.5+ - Model Context Protocol server for tool management (`mcp_server.py`)
- LeRobot 0.3.3 - Robot control and imitation learning (SO10x arms)
- Strands 1.12.0 - Reasoning agent framework for robot task execution
- pytest 9.0.2 - Test runner
- pytest-asyncio 1.3.0 - Async test support
- pytest-cov 7.0.0 - Coverage reporting
- pytest-mock 3.15.1 - Mocking utilities
- httpx 0.28.1 - HTTP client for integration tests
- Pydantic 2.0+ - Data validation and configuration
- python-dotenv 1.2.1+ - Environment variable loading
- Loguru 0.7.3+ - Structured logging
- Uvicorn 0.41.0+ - ASGI server for FastAPI

## Key Dependencies

- `pipecat-ai[anthropic,aws,aws-nova-sonic,deepgram,elevenlabs,silero,webrtc]==0.0.104` - Core voice pipeline with multi-service support
- `anthropic` (via pipecat) - Claude LLM integration
- `deepgram` (via pipecat) - Speech-to-text and text-to-speech
- `elevenlabs` (via pipecat) - Premium TTS voice synthesis
- `strands-agents[anthropic,otel]==1.12.0` - Agent framework with telemetry
- `fastmcp>=2.14.5` - MCP server for tool management
- `pyzmq>=27.1.0` - ZMQ bindings for inference client communication
- `msgpack>=1.0` - Binary serialization
- `langfuse>=3.14.3` - OpenTelemetry tracing and observability
- `opentelemetry.exporter.otlp` (via pipecat) - OTEL span export
- `lerobot[feetech]==0.3.3` - Robot control library
- `opencv-python` (implicit via lerobot) - Computer vision
- `numpy` - Array operations

## Configuration

- `.env` file (not committed) - Runtime secrets (API keys, AWS credentials)
- `.env.example` - Template documenting required vars
- `config.example.yaml` - Configuration template with sensible defaults
- `my-dum-e.yaml` - Instance-specific deployment config
- `pyproject.toml` - Project metadata and dependency groups
- `uv.lock` - Frozen dependency versions for reproducibility
- `.python-version` - Specifies Python 3.12

## Platform Requirements

- Linux environment (`.ttyACM0` serial port for robot)
- Python 3.12
- uv package manager
- Virtual environment support
- Linux/POSIX (robot controllers via `/dev/ttyACM0`)
- AWS region configuration for Bedrock/Polly/Transcribe optionally
- WebRTC capability (Twilio or direct WebRTC transport)
- MCP server reachability (HTTP `http://localhost:8000/mcp`)
- Local shared-memory backends (default, single-machine)
- Cloud backends: AWS IoT (MQTT) + DynamoDB (optional)

## Service Integration Points

- Default: Deepgram (STT) + Anthropic Claude (LLM) + ElevenLabs (TTS)
- AWS Profile: AWS Transcribe (STT) + AWS Bedrock Claude (LLM) + AWS Polly (TTS)
- Speech-to-Speech: AWS Nova Sonic (direct LLM audio)
- ZMQ-based inference client for policy server (port 5555)
- Gr00t model inference via BaseInferenceClient

<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->

## Conventions

## Naming Patterns

- Module files use `snake_case.py` (e.g., `pipecat_server.py`, `mcp_server.py`, `dum_e.py`)
- Test files follow pattern `test_<module>.py` (e.g., `test_utils.py`, `test_pipecat_server.py`)
- Package directories use `snake_case` (e.g., `embodiment/so_arm10x`, `policy/gr00t`, `shared/fleet_manager`)
- All functions use `snake_case` (e.g., `build_backend_config`, `load_config_file`, `image_to_jpeg_bytes`)
- Private helper functions prefix with `_` (e.g., `_spawn_mcp_server`, `_spawn_pipecat_server`, `_filter_verbose_content`)
- Async functions use `a` prefix for operations (e.g., `arun`, `astream`, `aload` - inherited from `IRobotAgent` interface)
- Factory functions use `create_` prefix (e.g., `create_robot_tools`, `create_clean_callback_handler`)
- Local variables and attributes use `snake_case` (e.g., `task_id`, `message_type`, `robot_controller`)
- Configuration dictionaries use `snake_case` keys (e.g., `voice_config`, `agent_args`)
- Constants use `UPPER_SNAKE_CASE` (e.g., `LANGUAGE_PRESETS`, `TaskStatus` enum variants like `PENDING`, `RUNNING`)
- Classes use `PascalCase` (e.g., `RobotCallbackHandler`, `BackendConfig`, `AsyncMCPClient`, `IRobotAgent`)
- Interfaces/Abstract base classes use `I` prefix (e.g., `IRobotAgent`, `ITaskManager`, `IMessageBroker`, `IFleetManager`)
- Dataclasses use `PascalCase` (e.g., `TaskInfo`, `RobotInfo`, `Message`, `ToolDefinition`)
- Enum classes use `PascalCase` with `UPPER_SNAKE_CASE` members (e.g., `TaskStatus.PENDING`, `MessageType.TASK_PROGRESS`)

## Code Style

- No explicit formatter configured; code follows PEP 8 conventions
- Line length: No strict enforcer found; typical ranges 80-120 characters
- Docstrings: Module-level docstrings at top of file; function docstrings use standard format with Args, Returns, Yields sections
- Indentation: 4 spaces
- No linter configuration found in repository
- Code organization follows logical grouping: imports, constants, classes, functions, main block

## Import Organization

- No path aliases configured in `pyproject.toml` or `setup.cfg`
- Imports use absolute paths from project root (e.g., `from shared import BackendConfig`, `from embodiment.so_arm10x.controller import`)

## Error Handling

- `ValueError` for invalid inputs (e.g., missing required parameters like `--port`)
- `FileNotFoundError` for missing config files (raised in `load_config_file`)
- `RuntimeError` for dependency issues (e.g., PyYAML not available when loading YAML configs)
- Generic `Exception` in broad try-catch blocks for robustness (e.g., in `dum_e.py` fleet registration fallback)
- Logging via `logger.error()` or `logger.warning()` before raising or continuing

## Logging

- Log at module level: `logger = logging.getLogger(__name__)` for stdlib components; `from loguru import logger` for primary code
- Use structured logging with format string and values: `logger.info("Message with {}", value)`
- Log levels used: DEBUG, INFO, WARNING, ERROR
- Special emoji prefixes for domain-specific logging:

## Comments

- Comments are sparse; code is generally self-documenting through clear naming
- Docstrings provide rationale and usage examples (e.g., class docstrings explain purpose and responsibilities)
- Inline comments used for non-obvious logic or temporary workarounds
- Not applicable (Python project)
- Docstrings use Python conventions with Args, Returns, Yields sections:

## Function Design

- Functions are generally 20-50 lines
- Longer functions (100+ lines) exist for complex orchestration (e.g., `build_backend_config`, `run_jarvis` in `pipecat_server.py`)
- Private helper functions are smaller, focused on single responsibility
- Use type hints for all parameters (e.g., `task_id: Optional[str] = None`)
- Functions use keyword-only arguments for clarity on intent
- Configuration passed as `Dict[str, Any]` when flexible structure needed
- Optional parameters have sensible defaults
- Async functions return `Dict[str, Any]` or `AsyncIterator[Dict[str, Any]]` for streaming
- Functions return `None` for side-effect operations
- Configuration-building functions return Pydantic models (`BackendConfig`)

## Module Design

- Modules export primary classes and factory functions
- Helper functions prefixed with `_` are not intended for external use
- `__init__.py` in packages re-exports key interfaces and implementations
- `shared/__init__.py` is a barrel file that exports core interfaces, enums, and dataclasses
- Allows: `from shared import BackendConfig, IRobotAgent, TaskStatus`

## Type Hints

- Type hints present on all public functions and methods
- Return type hints included (e.g., `-> Dict[str, Any]`, `-> AsyncIterator[Message]`)
- Optional types use `Optional[Type]` notation
- Union types use `Literal["option1", "option2"]` for mode/profile validation

## Configuration Management

- Configuration loaded from JSON/YAML files via `load_config_file(path: Optional[str])`
- Command-line args override config file, which overrides environment variables
- Pydantic `BaseModel` used for validation and structured access (`BackendConfig`)
- Environment variable names prefixed with `DUME_` for project-specific settings

<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->

## Architecture

## System Overview

```text

```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| **Voice Interface** | Real-time voice I/O (STT/TTS), Pipecat cascaded pipeline | `pipecat_server.py` |
| **MCP Server** | HTTP REST endpoints for task/message/fleet operations | `mcp_server.py` |
| **Task Manager** | Task lifecycle: create, track, claim, complete | `shared/task_manager/` |
| **Message Broker** | Async event streaming, progress updates, tool execution | `shared/message_broker/` |
| **Fleet Manager** | Robot registry, registration, health status | `shared/fleet_manager/` |
| **Robot Agent** | Multi-modal orchestration, tool invocation, streaming | `embodiment/so_arm10x/agent.py` |
| **Robot Controller** | LeRobot interface, camera I/O, joint commands | `embodiment/so_arm10x/controller.py` |
| **Policy Service** | Isaac GR00T VLA inference via ZMQ | `policy/gr00t/service.py` |
| **Orchestrator** | Process spawning, component coordination | `dum_e.py` |

## Pattern Overview

- **Interface-driven design**: High-level modules depend on abstract interfaces (`IRobotAgent`, `ITaskManager`, `IMessageBroker`, `IFleetManager`) not concrete implementations
- **Shared memory coordination**: Components communicate via shared-memory backends (with optional cloud backends for distributed deployments)
- **Streaming-first architecture**: All long-running operations (agent execution, tool calls) yield streaming events rather than blocking until completion
- **Multi-process isolation**: Pipecat server, MCP server, and agent worker run as separate processes, coordinating through backends
- **Modular embodiments**: Robot implementations (`SO10xRobotAgent`, `SO10xArmController`) inherit from interfaces, enabling swappable implementations

## Layers

- Purpose: Handle real-time voice I/O, STT→LLM→TTS cascaded pipeline, WebRTC bidirectional streaming
- Location: `pipecat_server.py`, configuration in `config.example.yaml` (voice section)
- Contains: Pipecat pipeline, service instantiation (LLM, STT, TTS), VAD, transport handling
- Depends on: Message broker (for streaming tool results back to voice), MCP service (for tool definitions)
- Used by: Frontend clients connecting via WebRTC
- Purpose: Expose task management, message streaming, and fleet operations via HTTP + MCP
- Location: `mcp_server.py` (FastMCP), shared backends in `shared/`
- Contains: MCP tool definitions, async message forwarding, task lifecycle handlers, fleet endpoints
- Depends on: Task Manager, Message Broker, Fleet Manager (interfaces + implementations)
- Used by: Voice interface, external MCP clients, agent orchestrator
- Purpose: Provide local IPC for task coordination, event streaming, and fleet state
- Location: `shared/task_manager/shm.py`, `shared/message_broker/shm.py`, `shared/fleet_manager/shm.py`
- Contains: File-backed shared memory with atomic operations, event circular buffers, task queues
- Depends on: `shared/__init__.py` (interface definitions), `multiprocessing.managers.SharedMemoryManager`
- Used by: Agent worker, MCP server, pipecat server
- Purpose: Orchestrate multi-modal task execution via LLM reasoning + tool invocation
- Location: `embodiment/so_arm10x/agent.py` (SO10xRobotAgent)
- Contains: Strands Agent integration, tool registry, streaming iteration, error recovery
- Depends on: Robot controller, policy service, message broker, task manager
- Used by: MCP server tasks, agent worker loop, orchestrator
- Purpose: Manage robot hardware I/O, joint state, camera observations, gripper control
- Location: `embodiment/so_arm10x/controller.py` (SO10xArmController + Gr00tRobotInferenceClient)
- Contains: LeRobot Robot API wrapper, camera frame capture, gripper state, policy client
- Depends on: LeRobot, OpenCV, ZMQ/msgpack for policy inference
- Used by: Agent layer for robot operations
- Purpose: Execute pre-trained VLA (vision-language-action) policies, return dense action sequences
- Location: `policy/gr00t/service.py` (ExternalRobotInferenceClient base, Gr00tRobotInferenceClient wrapper)
- Contains: ZMQ REQ-REPLY protocol, msgpack serialization, numpy array marshalling
- Depends on: External Isaac-GR00T server (runs separately on GPU server)
- Used by: Robot controller, agent tools (e.g., `pick`, `place`)

## Data Flow

### Primary Request Path (Voice → Task Execution → Response)

### Task Lifecycle Flow

### Multi-modal Observation Flow

- Task state: File-backed shared memory (atomic with file locks) in `shared/task_manager/shm.py`
- Event history: Circular buffer in shared memory in `shared/message_broker/shm.py`
- Robot state: Transient in-process state in SO10xArmController (queried from LeRobot on-demand)
- Fleet state: Optional shared memory or cloud backends (DynamoDB, MQTT)

## Key Abstractions

- Purpose: Abstract interface for robot task executors; enables mock agents, different embodiments, and policy frameworks
- Examples: `SO10xRobotAgent` (`embodiment/so_arm10x/agent.py:210`)
- Pattern: Async generators for streaming, structured result dicts for completion
- Purpose: Decouple task lifecycle from storage backend; support both local and distributed deployments
- Examples: `SharedMemoryTaskManager` (`shared/task_manager/shm.py:29`)
- Pattern: Atomic operations (claim_task uses Compare-And-Swap semantics), status enums
- Purpose: Enable real-time event streaming without tight coupling between publishers and subscribers
- Examples: `SharedMemoryMessageBroker` (`shared/message_broker/shm.py:30`)
- Pattern: Async iterators for subscription, circular buffer for history
- Purpose: Abstract robot registry; support hybrid local + cloud deployments
- Examples: `SharedMemoryFleetManager` (`shared/fleet_manager/shm.py`)
- Pattern: Basic CRUD operations with enabled/disabled state
- Purpose: Low-level hardware abstraction for robot operations (joints, gripper, cameras)
- Examples: `SO10xArmController` (`embodiment/so_arm10x/controller.py`)
- Pattern: Context manager for connection lifecycle, observation dicts

## Entry Points

- Location: `pipecat_server.py`
- Triggers: `python pipecat_server.py` or subprocess from `dum_e.py`
- Responsibilities: Establish WebRTC transport, create Pipecat pipeline, run voice loop, handle tool calls via AsyncMCPClient
- Location: `mcp_server.py`
- Triggers: `python mcp_server.py` or subprocess from `dum_e.py`
- Responsibilities: Initialize shared-memory backends, create FastMCP instance, expose task/message/fleet tools, run HTTP server
- Location: `embodiment/so_arm10x/agent.py` (when run with `--worker` flag)
- Triggers: `python -m embodiment.so_arm10x.agent --worker --config config.yaml` or subprocess from `dum_e.py`
- Responsibilities: Load robot configuration, initialize SO10xRobotAgent, enter task claim loop, handle interrupts
- Location: `dum_e.py:163` (`main()` function)
- Triggers: `python dum_e.py --config my-dum-e.yaml --node all`
- Responsibilities: Parse config, spawn subprocesses (Pipecat, MCP, Agent), coordinate startup/shutdown, manage shared-memory backends
- Location: `embodiment/so_arm10x/agent.py` (when run with `--instruction` flag, not `--worker`)
- Triggers: `python -m embodiment.so_arm10x.agent --config config.yaml --instruction "pick apple"`
- Responsibilities: Create agent, execute single instruction, return result, exit
- Location: External (Isaac-GR00T)
- Triggers: Manual setup per README; runs on dedicated GPU server
- Responsibilities: Load VLA model, expose ZMQ endpoint at port 5555, handle observation→action inference

## Architectural Constraints

- **Single-threaded event loop per process**: Python async/await means each process (Pipecat, MCP, Agent) is single-threaded; concurrency via multiprocessing, not multithreading
- **Shared memory namespace isolation**: Multiple deployments can coexist on same machine via different `DUME_NAMESPACE` values; prevents cross-talk
- **ZMQ REQ-REPLY pattern**: Policy service client blocks until server responds; timeouts at 15s (configurable) to prevent deadlocks
- **No circular dependencies**: Task Manager doesn't call Message Broker; Message Broker doesn't call Task Manager; prevents deadlock in shared-memory operations
- **Async message publishing**: All long-running operations must yield control; blocking operations (e.g., robot movement) run in executor threads to avoid starving event loop
- **Global state per module**: Strands Agent created once per worker and cached; re-entrant calls reuse same agent instance
- **File-based locking**: Shared memory backends use POSIX file locks (`fcntl.flock`) for atomicity; requires shared filesystem if scaling to multiple machines

## Anti-Patterns

### Blocking Robot Operations in Event Loop

```python

```

### Synchronous Policy Calls

```python

```

### Message Publishing Without Error Handling

```python

```

## Error Handling

- **Policy inference failures**: If ZMQ timeout or connection refused, agent falls back to classical control (MoveIt) if available, or returns error to user
- **Camera capture failures**: If wrist or front camera unavailable, agent continues with cached image or retry loop (up to 3x)
- **Tool execution errors**: Caught in Strands Agent, returned to LLM as error context for recovery planning
- **Shared memory backend errors**: If task manager write fails, error logged but does not crash process; future retries possible
- **Network timeouts**: 15s timeout on policy service calls; exceeded → user error message + task marked FAILED

## Cross-Cutting Concerns

- Centralized via `loguru` in all modules; configured in `utils.py`
- Robot operations logged at DEBUG level to reduce noise
- Agent streaming events logged at TRACE for inspection
- No credentials logged (redacted by string matching)
- Configuration via Pydantic models (`BackendConfig`, YAML schemas)
- Task instruction strings validated for length (max 1000 chars)
- Robot parameters validated at startup (camera indices, ports) via LeRobot
- API keys stored in `.env` (never committed)
- MCP server currently unauthenticated (local-only); intended for same-machine clients
- Policy server ZMQ connection is local-only (no auth layer)

<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->

## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->

## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:

- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->

## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
