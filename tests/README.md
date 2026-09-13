# Tests

Run `uv sync --locked`, then `uv run pytest -q` from the repository root.
Default tests use simulated hardware, local mock HTTP/gRPC/ZMQ services and
isolated shared memory. They require permission to bind loopback sockets.
Tracked GR00T metadata fixtures avoid a dependency on downloaded checkpoints.

GPU model serving and paid speech services are opt-in through
`DUME_RUN_LIVE_LEROBOT_TESTS=1` and `DUME_RUN_LIVE_SPEECH_TESTS=1` respectively.
Do not enable these for ordinary regression runs. No default test moves a robot.

See [the validation guide](../docs/POLICY-VALIDATION.md) for the coverage map,
recording format, benchmark commands, attended trial procedure and limitations.
New embodiments should add mapping, invalid-input and stop/cancellation tests
before GPU or physical qualification; see [extension points](../docs/EXTENDING-EMBODIMENTS.md).
