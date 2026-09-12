"""LRG-06's loopback guarantee, asserted against the documented run command.

**The guarantee lives in the RUN COMMAND, not in the code**, so this module reads
the command `README.md` documents. Testing the bind address would test the wrong
artifact: the container binds all interfaces *inside* itself on purpose, because
a container-internal loopback bind is unreachable from the host — which makes
``PolicyServerConfig``'s ``host="localhost"`` default (``configs.py:56``) simply
wrong here, and it is overridden deliberately. The loopback guarantee therefore
lives entirely in the host-side ``-p 127.0.0.1:8080:8080`` publish spec.

**Why it is asserted rather than assumed.** LeRobot's async gRPC transport
``pickle.loads`` peer bytes in **both** directions by upstream design, and the
service has no authentication of any kind, so anyone who can reach the port can
execute code in the server process. The publish spec's host-IP prefix is the
**sole** mitigation for that accepted risk (ASY-06), and D-12 states plainly that
a sole mitigation cannot rest on a default. ``--network host`` was considered and
**rejected** for exactly this reason. Widening the spec to make something work
would silently convert an accepted, mitigated risk into an unmitigated one.

**Scoped deliberately to the ``lerobot-policy`` block.** The incumbent
Isaac-GR00T runbook publishes ``5555`` with no host-IP prefix at all, so a test
that globbed every ``docker run`` fence in `README.md` would FAIL on that line.
Narrowing the GR00T-native container's reachability would change the fallback
path Phase 7's live parity gate runs on — a different failure axis and a
different decision. ``test_incumbent_gr00t_block_is_out_of_scope...`` records
that scoping as a test rather than leaving it as prose.

Both helpers are pure over text, so the negative tests can drive them with
SYNTHETIC input and no test ever mutates `README.md`. Both **raise** rather than
returning an empty value on absent input: a guard that silently passes on zero
blocks is worse than no guard, because it reads as coverage. **Nothing here may
skip** — a skipped loopback test is a silent pass on the only access control this
service has.
"""

import re
import shlex
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

README = REPO_ROOT / "README.md"
ENTRYPOINT = REPO_ROOT / "docker" / "lerobot-policy" / "entrypoint.py"

#: The stable anchor comment plan 06-06 authored immediately above the fenced
#: run block. It sits OUTSIDE the fence so the fence stays a pure command.
ANCHOR = "# lerobot-policy container (LRG-06: loopback publish only)"

#: The exact publish value LRG-06 requires. Full-string equality, not a prefix
#: match, so a typo in the port mapping is caught too.
EXPECTED_PUBLISH = "127.0.0.1:8080:8080"

#: Ways a command can opt into the host network namespace instead of publishing
#: a port. D-12 rejected this alternative explicitly.
HOST_NETWORK_TOKENS = ("--network=host", "--net=host")
HOST_NETWORK_FLAGS = ("--network", "--net")

#: How the incumbent Isaac-GR00T run block is located — by its own container
#: name, never by an unanchored "next fence" search.
INCUMBENT_LOCATOR = "gr00t-server"

#: A fenced code block at any indentation, with an optional language tag.
_FENCE = re.compile(
    r"^[ \t]*```[A-Za-z0-9_+.-]*[ \t]*\n(.*?)\n[ \t]*```[ \t]*$",
    re.DOTALL | re.MULTILINE,
)


def fenced_blocks(text: str) -> list[str]:
    """Every fenced code block body in ``text``, in document order."""
    return [match.group(1) for match in _FENCE.finditer(text)]


def extract_run_block(readme_text: str, anchor: str = ANCHOR) -> str:
    """The body of the FIRST fenced block following ``anchor``.

    Raises ``AssertionError`` — never returns ``None`` and never returns ``""``
    — when the anchor is absent or when no fenced block follows it. A helper that
    returned empty on a missing anchor would make every assertion in this module
    vacuous the moment someone reorganized the README, which is precisely the
    tampering path T-06-27 describes.

    The anchoring is load-bearing in both directions: a second fence (the
    ``--preflight-only`` command) follows further down, so an unanchored "next
    fence" search would eventually assert against the wrong block.
    """
    index = readme_text.find(anchor)
    if index < 0:
        raise AssertionError(
            f"anchor not found: {anchor!r} — LRG-06's publish spec cannot be "
            "asserted, and passing here would be a vacuous guard rather than a "
            "missing one"
        )
    match = _FENCE.search(readme_text, index + len(anchor))
    if match is None:
        raise AssertionError(
            f"no fenced code block follows the anchor {anchor!r} — the documented "
            "run command is gone or the fence was reshaped"
        )
    return match.group(1)


def _tokens(command_text: str) -> list[str]:
    """Shell tokens of a possibly backslash-continued, multi-line command."""
    return shlex.split(command_text)


def publish_flags(command_text: str) -> list[str]:
    """The value following each ``-p`` / ``--publish`` flag, in order.

    Raises ``AssertionError`` when the command declares none, for the same
    non-vacuity reason as :func:`extract_run_block`: an empty list would make
    ``all(...)``-shaped assertions pass trivially.
    """
    tokens = _tokens(command_text)
    values: list[str] = []
    for position, token in enumerate(tokens):
        if token in ("-p", "--publish"):
            if position + 1 >= len(tokens):
                raise AssertionError(
                    f"{token} appears with no value in: {command_text!r}"
                )
            values.append(tokens[position + 1])
        elif token.startswith("--publish="):
            values.append(token.split("=", 1)[1])
    if not values:
        raise AssertionError(
            f"no -p/--publish flag found in: {command_text!r} — a command that "
            "publishes nothing must not satisfy a publish-spec assertion by "
            "vacuity"
        )
    return values


def shares_host_network_namespace(command_text: str) -> bool:
    """True when the command opts into the host network namespace."""
    tokens = _tokens(command_text)
    for position, token in enumerate(tokens):
        if token in HOST_NETWORK_TOKENS:
            return True
        if token in HOST_NETWORK_FLAGS and tokens[position + 1 : position + 2] == [
            "host"
        ]:
            return True
    return False


def assert_loopback_only(command_text: str) -> None:
    """The LRG-06 guard itself: raise unless the command is loopback-only.

    One function so the positive case and all three synthetic negatives exercise
    the SAME logic — a negative test against a second copy of the check would
    prove nothing about the check that actually runs.
    """
    if shares_host_network_namespace(command_text):
        raise AssertionError(
            "the command shares the host network namespace, which D-12 rejected: "
            "it places an unauthenticated pickle-wire service on every interface "
            "with no publish spec left to constrain it"
        )
    values = publish_flags(command_text)
    if values != [EXPECTED_PUBLISH]:
        raise AssertionError(
            f"publish spec is {values!r}, expected exactly [{EXPECTED_PUBLISH!r}] — "
            "the host-IP prefix is the sole mitigation for a pickle-in-both-"
            "directions wire on an unauthenticated service"
        )


# --- The documented command ---------------------------------------------------


def test_documented_lerobot_policy_run_block_publishes_with_the_host_ip_prefix():
    """The documented run command publishes exactly 127.0.0.1:8080:8080.

    Exactly one publish flag, and its value asserted by full-string equality
    rather than a prefix match, so a typo in the port mapping is caught as well
    as a missing host IP.
    """
    block = extract_run_block(README.read_text(encoding="utf-8"))
    values = publish_flags(block)
    assert len(values) == 1, f"expected exactly one publish flag, got {values!r}"
    assert values[0] == EXPECTED_PUBLISH
    # The guard the three synthetic negatives below drive, run against the real
    # documented command.
    assert_loopback_only(block)


def test_documented_run_block_binds_all_interfaces_inside_the_container_deliberately():
    """The run command must NOT tighten the container's INTERNAL bind.

    This is the half that stops a well-meaning "tighten the bind too" change from
    making the container unreachable from the host. The entrypoint's ``--host``
    default is the all-interfaces address on purpose, overriding
    ``PolicyServerConfig``'s ``host="localhost"``, and the documented command
    does not override it back to a loopback address.
    """
    block = extract_run_block(README.read_text(encoding="utf-8"))
    tokens = _tokens(block)
    for position, token in enumerate(tokens):
        if token == "--host":
            following = tokens[position + 1 : position + 2]
            assert following and following[0] not in ("127.0.0.1", "localhost"), (
                "the documented command overrides the container's internal bind to "
                f"{following!r}; a container-internal loopback bind is unreachable "
                "from the host, so this would break the container while adding no "
                "security — the loopback guarantee is the publish spec's job"
            )

    entrypoint = ENTRYPOINT.read_text(encoding="utf-8")
    all_interfaces = ".".join(["0", "0", "0", "0"])
    assert f'default="{all_interfaces}"' in entrypoint, (
        f"{ENTRYPOINT} no longer defaults --host to the all-interfaces address; "
        "PolicyServerConfig's localhost default is wrong for a published container"
    )
    # The explanation is present, not just the value: an unexplained 0.0.0.0 is
    # what a future reader "fixes".
    assert "loopback is enforced by the publish spec" in entrypoint
    assert "unreachable" in entrypoint


# --- Negative tests: the guard has teeth --------------------------------------
#
# Each drives a pure helper with SYNTHETIC text. No test mutates README.md, and
# no test asserts anything about the incumbent Isaac-GR00T block's publish flags.


def test_extractor_fails_loudly_when_the_anchor_is_absent():
    """A README without the anchor must RAISE, not yield an empty block.

    Non-vacuity for the extractor itself. Without this, a README reorganization
    would silently turn every assertion above into a tautology over ``""``.
    """
    synthetic = "# A README\n\nSome prose.\n\n```bash\ndocker run --rm hello\n```\n"
    with pytest.raises(AssertionError, match="anchor not found"):
        extract_run_block(synthetic)

    # And the complementary branch: anchor present, no fence after it.
    with pytest.raises(AssertionError, match="no fenced code block follows"):
        extract_run_block(f"# A README\n\n{ANCHOR}\n\nno fence here at all\n")


def test_guard_rejects_a_synthetic_command_without_the_host_ip_prefix():
    """A bare ``port:port`` publish value must FAIL the guard.

    PATTERNS.md's required fail-first proof: without it, a doc rewrite that
    dropped the host IP would leave the positive assertion above passing for the
    wrong reason.
    """
    synthetic = (
        "docker run -d \\\n"
        "    --gpus all \\\n"
        "    -p 8080:8080 \\\n"
        "    --name lerobot-policy-server \\\n"
        "    lerobot-policy\n"
    )
    # The helper still extracts the value — it is the CHECK that must reject it.
    assert publish_flags(synthetic) == ["8080:8080"]
    with pytest.raises(AssertionError, match="publish spec is"):
        assert_loopback_only(synthetic)


def test_guard_rejects_a_synthetic_command_that_shares_the_host_network_namespace():
    """Opting into the host network namespace must FAIL the guard.

    D-12 rejected ``--network host`` explicitly: it hands the container every
    interface and leaves no publish spec to constrain it. This test makes that
    rejection mechanical rather than a paragraph someone can disagree with.
    """
    for variant in (
        "docker run -d --network host --name lerobot-policy-server lerobot-policy\n",
        "docker run -d --network=host --name lerobot-policy-server lerobot-policy\n",
        "docker run -d --net host -p 127.0.0.1:8080:8080 lerobot-policy\n",
    ):
        assert shares_host_network_namespace(variant)
        with pytest.raises(AssertionError, match="host network namespace"):
            assert_loopback_only(variant)

    # A command with no publish flag at all must not pass by vacuity either.
    with pytest.raises(AssertionError, match="no -p/--publish flag found"):
        publish_flags("docker run -d --name lerobot-policy-server lerobot-policy\n")


# --- The deliberate scoping ---------------------------------------------------


def test_incumbent_gr00t_block_is_out_of_scope_and_the_scoping_is_deliberate():
    """The incumbent Isaac-GR00T block is left alone, and that is the decision.

    That block publishes ``5555`` with **no** host-IP prefix, so it would FAIL
    this module's guard — which is demonstrated below rather than asserted about
    it. A test that globbed every ``docker run`` fence in `README.md` would
    therefore go red on a line this phase deliberately does not touch: narrowing
    the GR00T-native container's reachability would change the fallback path
    Phase 7's live parity gate runs on, which is a different failure axis and
    deserves its own change with its own verification.

    So this module asserts three things about scoping, and nothing about what the
    incumbent block's publish spec *ought* to be: the incumbent block still
    exists, an unanchored glob would collide with it, and the anchored extractor
    does not reach it.
    """
    text = README.read_text(encoding="utf-8")

    incumbent = [
        block for block in fenced_blocks(text) if INCUMBENT_LOCATOR in block
    ]
    assert incumbent, (
        f"the incumbent Isaac-GR00T run block (located by {INCUMBENT_LOCATOR!r}) is "
        "gone from README.md; this phase must not delete it"
    )

    # The collision is real, which is WHY the extractor is anchored. This is a
    # statement about the guard's scoping, not a verdict on the incumbent line.
    with pytest.raises(AssertionError):
        assert_loopback_only(incumbent[0])

    # And the anchored extractor lands on the lerobot-policy block, not that one.
    block = extract_run_block(text)
    assert "lerobot-policy" in block
    assert INCUMBENT_LOCATOR not in block
