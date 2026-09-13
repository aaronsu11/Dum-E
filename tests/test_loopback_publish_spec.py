"""Check executable serving documentation for accidental public pickle transport."""
import re
import shlex
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def documented_run(text, name):
    blocks = re.findall(r"^[ \t]*```(?:bash|sh)?[ \t]*\n(.*?)\n[ \t]*```[ \t]*$", text, re.S | re.M)
    matches = [b for b in blocks if "docker run" in b and f"--name {name}" in b]
    if len(matches) != 1:
        raise ValueError(f"Expected one Docker run block for {name}")
    return matches[0]


def assert_loopback(command, port):
    tokens = shlex.split(command.replace("\\\n", " "), comments=True)
    publishes = []
    for i, token in enumerate(tokens):
        if token in ("--network=host", "--net=host") or (token in ("--network", "--net") and tokens[i+1:i+2] == ["host"]):
            raise ValueError("Pickle server cannot use host networking")
        if token in ("-p", "--publish"):
            publishes.append(tokens[i+1] if i+1 < len(tokens) else "")
        elif token.startswith("--publish="):
            publishes.append(token.split("=", 1)[1])
    if publishes != [f"127.0.0.1:{port}:{port}"]:
        raise ValueError("Expected an explicit loopback publish")


def test_current_lerobot_and_native_commands_publish_only_loopback():
    lerobot = documented_run((ROOT / "docs/POLICY-SERVING.md").read_text(), "dume-lerobot")
    native = documented_run((ROOT / "README.md").read_text(), "gr00t-server")
    assert_loopback(lerobot, 8080)
    assert_loopback(native, 5555)


@pytest.mark.parametrize("command", [
    "docker run -p 8080:8080 model", "docker run -p 0.0.0.0:8080:8080 model",
    "docker run --network host model", "docker run --net=host model",
    "docker run model", "docker run -p model"])
def test_unsafe_or_missing_publication_is_rejected(command):
    with pytest.raises(ValueError):
        assert_loopback(command, 8080)


def test_documentation_extraction_cannot_pass_on_an_absent_command():
    with pytest.raises(ValueError):
        documented_run("No command", "dume-lerobot")
