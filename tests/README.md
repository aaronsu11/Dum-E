## Tests

Comprehensive test suite for Dum-E that validates shared interfaces, MCP server tools, configuration management, utility functions, and orchestrated server/agent flows **without requiring robot hardware**. All tests are hardware-independent unless explicitly using the mock worker.

### 🎯 Test Design Philosophy

- **Hardware Independence**: All tests run without robot hardware using mocks and shared memory backends
- **Comprehensive Coverage**: 50%+ overall coverage with focus on testable components
- **Environment Isolation**: Each test runs in isolated environment to prevent interference
- **Integration Testing**: Real process spawning with shared memory for end-to-end validation
- **Mock-Based Testing**: Extensive use of mocks to simulate hardware and external dependencies

### 📁 Test Files

#### **test_dum_e.py** (24 tests)
**Comprehensive tests for main application orchestration**

- **TestBuildBackendConfig** (7 tests)
  - Configuration resolution with priority: CLI args > config file > environment
  - Default values, invalid types, logging verification
  - Example:
    ```python
    def test_build_backend_config_priority_order(self):
        # Tests that args override config which overrides env
        args = Namespace(namespace="args_namespace")
        cfg = {"backend": {"namespace": "config_namespace"}}
        with mock.patch.dict(os.environ, {"DUME_NAMESPACE": "env"}):
            result = build_backend_config(args, cfg)
            assert result.namespace == "args_namespace"  # Highest priority
    ```

- **TestSpawnMcpServer** (3 tests)
  - MCP server process spawning with environment injection
  - Command construction and environment variable handling

- **TestSpawnPipecatServer** (2 tests)
  - Pipecat server process spawning and environment inheritance

- **TestSpawnAgentWorker** (8 tests)
  - Mock mode: Uses `tests.mocks` for hardware-free testing
  - Real mode: Tests with config file or individual args
  - Default values and error handling (missing port, etc.)
  - Example:
    ```python
    def test_spawn_agent_worker_mock_mode(self):
        # Tests spawning mock agent without hardware
        config = BackendConfig(namespace="test")
        agent_args = {"use_mock": True, "id": "mock_robot"}
        _spawn_agent_worker(config, agent_args)
        # Validates command uses tests.mocks module
    ```

- **TestEnvironmentInjection** (3 tests)
  - Environment variable injection for all spawned processes
  - Validates namespace and custom environment variables

- **TestEndToEndIntegration** (1 test)
  - Full system integration with real processes and shared memory
  - HTTP MCP client connection and progress event validation
  - Proper process cleanup

#### **test_utils.py** (22 tests)
**Comprehensive utility function testing**

- **TestLoadConfigFile** (8 tests)
  - JSON and YAML config file loading
  - Error handling: file not found, unsupported formats, import errors
  - Example:
    ```python
    def test_load_config_file_priority(self):
        # Tests loading from JSON and YAML
        with tempfile.NamedTemporaryFile(suffix='.json') as f:
            json.dump({"key": "value"}, f)
            result = load_config_file(f.name)
            assert result["key"] == "value"
    ```

- **TestCleanFormatter** (3 tests)
  - Log formatter initialization and formatting
  - Long message truncation

- **TestRobotCallbackHandler** (6 tests)
  - Callback handler initialization and configuration
  - Data output and streaming data handling
  - Content filtering (normal, non-string, binary detection)

- **TestSetupRobotLogging** (2 tests)
  - Logging setup and configuration
  - Already configured detection

- **TestCreateCleanCallbackHandler** (2 tests)
  - Callback handler factory function

#### **test_shm_broker.py** (1 test)
**Shared memory message broker testing**

- Scope: `SharedMemoryMessageBroker` publish/subscribe/history
- What it does: Allocates SHM ring buffer, subscribes to `TASK_PROGRESS`, publishes messages, asserts filtered receipt and bounded history
- Example:
  ```python
  async def test_shared_memory_broker():
      # Creates broker with SHM backend
      # Publishes progress messages
      # Verifies subscription filtering and history
  ```

#### **test_shm_task_manager.py** (1 test)
**Shared memory task manager testing**

- Scope: `SharedMemoryTaskManager` lifecycle and atomic claim
- What it does: Creates task, claims it, updates progress, completes it, verifies via `get_task` and `list_tasks`
- Example:
  ```python
  async def test_shared_memory_task_manager():
      # Creates and claims tasks atomically
      # Updates task status and metadata
      # Lists tasks with filtering
  ```

#### **test_agent_interfaces.py** (1 test)
**Robot agent interface testing**

- Scope: `IRobotAgent` integration using `MockRobotAgent`
- What it does: Runs `MockRobotAgent.astream` with SHM backends, subscribes to broker, asserts warmup and assistant messages, verifies task completion
- Example:
  ```python
  async def test_agent_with_shared_memory():
      # Uses MockRobotAgent (no hardware required)
      # Validates agent streaming and progress updates
      # Checks task lifecycle completion
  ```

#### **test_mcp_interfaces.py** (8 tests)
**MCP server tool testing**

- Scope: MCP tools exposed by `mcp_server.py` using SHM backends (no worker)
- What it does: Starts MCP HTTP server, seeds tasks/messages/fleet via SHM, uses `fastmcp.Client` to call tools
- Tests:
  - `list_namespaces` - Returns active namespace
  - `list_tasks` - Empty and filtered views
  - `get_task_details` - With optional recent messages
  - `cancel_task` - Updates task state
  - `list_last_messages` - Filters by task_id and message_types
  - `register_robot` - Upserts robot in fleet
  - `list_robots` - Lists all robots
  - `get_robot` - Gets robot details by ID
  - `set_robot_enabled` - Enables/disables robot
- Example:
  ```python
  async def test_task_lifecycle():
      # Creates task via SHM
      # Calls MCP tools via HTTP client
      # Verifies task state transitions
  ```

### 🚀 Running Tests

#### Run all tests
```bash
pytest tests/
```

#### Run with coverage report
```bash
pytest tests/ --cov=. --cov-report=term-missing --cov-report=html
```

#### Run specific test file
```bash
pytest tests/test_dum_e.py -v
```

#### Run specific test class
```bash
pytest tests/test_dum_e.py::TestBuildBackendConfig -v
```

#### Run specific test
```bash
pytest tests/test_dum_e.py::TestBuildBackendConfig::test_build_backend_config_defaults -v
```

#### Run with output
```bash
pytest tests/ -v -s
```

### 🔧 Shared Memory Setup

Tests that directly attach to SHM (not via dum_e) create `ShareableList` buffers using `SharedMemoryManager` and pass names via environment variables:

#### Environment Variables
- `DUME_BROKER_BUF` - Broker buffer name for messages
- `DUME_BROKER_META` - Broker metadata (write index)
- `DUME_TASKS_BUF` - Task storage buffer name
- `DUME_FLEET_BUF` - Fleet registry buffer name
- `DUME_NAMESPACE` - Namespace for lock files (default: "default")
- `DUME_IPC` - Backend selection (set to "shm" for shared memory)

#### Example Setup
```python
from multiprocessing.managers import SharedMemoryManager

with SharedMemoryManager() as smm:
    # Create shared memory buffers
    broker_buf = smm.ShareableList([" " * 4096] * 32)
    broker_meta = smm.ShareableList([0])
    tasks_buf = smm.ShareableList([" " * 4096] * 16)
    
    # Set environment variables
    os.environ["DUME_BROKER_BUF"] = broker_buf.shm.name
    os.environ["DUME_BROKER_META"] = broker_meta.shm.name
    os.environ["DUME_TASKS_BUF"] = tasks_buf.shm.name
    os.environ["DUME_NAMESPACE"] = "test"
    os.environ["DUME_IPC"] = "shm"
```

### 🧪 Test Patterns

#### 1. Mock-Based Unit Testing
```python
@mock.patch("subprocess.Popen")
def test_spawn_process(self, mock_popen):
    # Test process spawning without actually spawning
    _spawn_mcp_server(config)
    
    # Inspect mock calls
    call_args = mock_popen.call_args
    cmd = call_args[0][0]
    env = call_args[1]["env"]
    
    assert sys.executable in cmd
    assert env["DUME_NAMESPACE"] == "test"
```

#### 2. Environment Isolation
```python
def test_with_clean_env(self):
    # Clear environment to ensure true defaults
    with mock.patch.dict(os.environ, {}, clear=True):
        result = build_backend_config(args, cfg)
        assert result.namespace == "default"
```

#### 3. Custom Process Capture
```python
def test_env_injection(self):
    popen_calls = []
    
    class DummyPopen:
        def __init__(self, cmd, env=None):
            popen_calls.append((cmd, env))
    
    with mock.patch("subprocess.Popen", DummyPopen):
        _spawn_mcp_server(config, extra_env)
    
    cmd, env = popen_calls[0]
    assert env["CUSTOM_VAR"] == "custom_value"
```

#### 4. Async Integration Testing
```python
@pytest.mark.asyncio
async def test_end_to_end(self):
    # Spawn real processes with shared memory
    server_proc = _spawn_mcp_server(config, shm_env)
    worker_proc = _spawn_agent_worker(config, agent_args, shm_env)
    
    try:
        # Connect via HTTP and test
        async with Client("http://localhost:8010/mcp") as client:
            result = await client.call_tool("execute_robot_instruction", {...})
            assert result.data["status"] == "completed"
    finally:
        # Clean up processes
        server_proc.terminate()
        worker_proc.terminate()
```

### 📝 Writing New Tests

#### Guidelines
1. **Isolate Environment**: Use `mock.patch.dict(os.environ, ...)` to prevent test interference
2. **Mock Hardware**: Use `tests.mocks.MockRobotAgent` for hardware-free testing
3. **Mock Processes**: Use `@mock.patch("subprocess.Popen")` to test process spawning
4. **Clean Up**: Always terminate spawned processes in `finally` blocks
5. **Async Tests**: Use `@pytest.mark.asyncio` for async test functions
6. **Descriptive Names**: Use clear test names that describe what's being tested
7. **Comprehensive**: Test happy path, error cases, and edge cases

#### Example New Test
```python
class TestNewFeature:
    """Test new feature functionality."""
    
    def test_new_feature_happy_path(self):
        """Test new feature works correctly with valid input."""
        # Arrange
        config = BackendConfig(namespace="test")
        
        # Act
        result = new_feature(config)
        
        # Assert
        assert result.status == "success"
    
    def test_new_feature_error_handling(self):
        """Test new feature handles errors gracefully."""
        with pytest.raises(ValueError, match="Invalid input"):
            new_feature(None)
```

### 🐛 Debugging Tests

#### View test output
```bash
pytest tests/ -v -s
```

#### Run single test with debugging
```bash
pytest tests/test_dum_e.py::TestBuildBackendConfig::test_build_backend_config_defaults -v -s
```

#### Check coverage for specific file
```bash
pytest tests/ --cov=dum_e --cov-report=term-missing
```

#### Generate HTML coverage report
```bash
pytest tests/ --cov=. --cov-report=html
# Open htmlcov/index.html in browser
```

### 📚 Notes

- **Mock Agent**: `MockRobotAgent` in `tests/mocks.py` avoids model and hardware dependencies while exercising the same interfaces and message flows
- **Logging**: `mcp_server.py` logs task lifecycle and progress forwarding for easier debugging
- **Coverage**: Focus is on testing business logic and configuration management; hardware-specific code (agent, controller) has 0% coverage by design
- **Shared Memory**: All shared memory backends are tested with real `ShareableList` buffers managed by `SharedMemoryManager`
- **Process Management**: Integration tests spawn real processes but use HTTP connections to avoid stdio complexity

### 🎯 Future Test Improvements

1. **Hardware Mocking**: Add comprehensive mocks for robot hardware interfaces
2. **Performance Tests**: Add tests for process spawning performance and shared memory efficiency
3. **Error Recovery**: Test error handling and recovery scenarios
4. **Concurrency**: Test concurrent task execution and shared memory race conditions
5. **Coverage**: Increase coverage of main application logic (currently 37%)
