"""Policy-backend selector.

One entry point — ``make_policy_backend()`` — turns the ``DUME_POLICY_BACKEND``
environment variable into a concrete :class:`shared.IPolicyBackend`. This is the
ONLY way production code reaches inference; there is no remaining hardcoded path
to a specific policy stack.

Selection is an explicit allowlist plus explicit branches. A backend
name is never used to build an import path, a module attribute lookup or a class
name, and is never ``eval``'d — the allowlist IS the whole validation surface
(ASVS V5).

Each branch lazy-imports its own dependencies so a selection never pays for a
stack it does not use.
"""

import os
from typing import Any

from shared import IPolicyBackend

#: Name of the environment variable that selects the backend.
POLICY_BACKEND_ENV_VAR = "DUME_POLICY_BACKEND"

#: The allowlist. Anything not in here is rejected — never defaulted.
POLICY_BACKENDS = ("lerobot", "groot-native", "galaxea", "pi05-so101")

#: Code-level default when the environment variable is unset. Deliberately the
#: target backend rather than the one that works today, so an unset variable
#: raises the "not wired" error below instead of silently selecting a fallback.
DEFAULT_POLICY_BACKEND = "lerobot"


def make_policy_backend(**kwargs: Any) -> IPolicyBackend:
    """Build the policy backend selected by ``DUME_POLICY_BACKEND``.

    Args:
        **kwargs: Forwarded verbatim to the selected backend's constructor
            (``host``, ``port``, ``camera_keys``, ``robot_state_keys``,
            ``show_images``, ``language_instruction`` — the SAME keyword set for
            all backends), so every constructor default stays reachable.

    Returns:
        A concrete :class:`shared.IPolicyBackend`.

    Raises:
        ValueError: on an unknown value.
    """
    backend = os.getenv(POLICY_BACKEND_ENV_VAR, DEFAULT_POLICY_BACKEND)

    if backend not in POLICY_BACKENDS:
        # DELIBERATE DIVERGENCE from the DUME_DEEPGRAM_BACKEND analog, which
        # warns and falls back to its default on an unknown value. Do NOT
        # "restore parity" here: silently swapping which neural network drives a
        # physical arm is worse than a crash, so a typo must stop the process
        # rather than quietly select a different policy.
        raise ValueError(
            f"{POLICY_BACKEND_ENV_VAR}={backend!r} is not a known policy backend; "
            f"allowed values are {', '.join(POLICY_BACKENDS)}. "
            "Refusing to fall back to a default — an unintended policy must never "
            "silently command the arm."
        )

    if backend == "lerobot":
        # Lazy-import INSIDE the branch, and this is the branch that proves why the
        # discipline exists: this import DOES pull the LeRobot policy stack (torch
        # and grpc, transitively), so at module scope it would make every
        # groot-native process pay for a stack it never uses. Asserted by
        # tests/test_policy_backend.py::test_factory_module_import_pulls_no_torch_or_lerobot,
        # which runs in a subprocess so the rest of the suite cannot mask it.
        if os.getenv("DUME_ASYNC_INFERENCE", "0") == "1":
            from policy.lerobot.serialized_backend import SerializedLeRobotPolicyBackend
            return SerializedLeRobotPolicyBackend(**kwargs)
        from policy.lerobot.backend import LeRobotPolicyBackend

        return LeRobotPolicyBackend(**kwargs)

    if backend == "pi05-so101":
        from policy.pi05_backend import Pi05SO101PolicyBackend
        return Pi05SO101PolicyBackend(**kwargs)

    if backend == "galaxea":
        if os.getenv("DUME_ASYNC_INFERENCE", "0") == "1":
            raise ValueError("Galaxea uses its native chunk protocol; LeRobot async mode is not supported")
        from policy.galaxea.backend import GalaxeaPolicyBackend
        return GalaxeaPolicyBackend(**kwargs)

    if backend == "groot-native":
        if os.getenv("DUME_ASYNC_INFERENCE", "0") == "1":
            raise ValueError("Async picking currently requires the LeRobot backend")
        # Lazy-import inside the branch, for the same reason as above: the GR00T
        # transport (policy/gr00t/service.py) depends only on msgpack/numpy/zmq and
        # must not be dragged in by a lerobot selection either.
        from embodiment.so_arm10x.controller import Gr00tRobotInferenceClient

        return Gr00tRobotInferenceClient(**kwargs)

    # Unreachable while every POLICY_BACKENDS member has a branch above. Kept so
    # that adding an allowlist entry without a branch fails loudly instead of
    # returning None into the robot action loop.
    raise NotImplementedError(
        f"{POLICY_BACKEND_ENV_VAR}={backend!r} is in POLICY_BACKENDS but has no "
        "branch in make_policy_backend(). This is a bug in policy/factory.py, "
        "not a configuration error."
    )
