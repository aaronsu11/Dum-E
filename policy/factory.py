"""Policy-backend selector (BACK-02/03/04/06).

One entry point — ``make_policy_backend()`` — turns the ``DUME_POLICY_BACKEND``
environment variable into a concrete :class:`shared.IPolicyBackend`. This is the
ONLY way production code reaches inference; there is no remaining hardcoded path
to a specific policy stack.

Selection is an explicit two-member allowlist plus explicit branches. A backend
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
POLICY_BACKENDS = ("lerobot", "groot-native")

#: Code-level default when the environment variable is unset (BACK-02).
DEFAULT_POLICY_BACKEND = "lerobot"


def make_policy_backend(**kwargs: Any) -> IPolicyBackend:
    """Build the policy backend selected by ``DUME_POLICY_BACKEND``.

    Args:
        **kwargs: Forwarded verbatim to the selected backend's constructor
            (``host``, ``port``, ``camera_keys``, ``robot_state_keys``,
            ``show_images``, ``language_instruction`` for ``groot-native``), so
            every constructor default stays reachable.

    Returns:
        A concrete :class:`shared.IPolicyBackend`.

    Raises:
        ValueError: on an unknown value, or on a known-but-unwired backend.
    """
    backend = os.getenv(POLICY_BACKEND_ENV_VAR, DEFAULT_POLICY_BACKEND)

    if backend == "groot-native":
        # Lazy-import inside the branch: the 'lerobot' branch will pull the
        # LeRobot policy stack (torch/GPU, and gRPC for async inference) in
        # Phase 6, and this torch-free path must not pay for it. The GR00T
        # transport (policy/gr00t/service.py) depends only on msgpack/numpy/zmq.
        from embodiment.so_arm10x.controller import Gr00tRobotInferenceClient

        return Gr00tRobotInferenceClient(**kwargs)

    # Never fall through: an unhandled value must raise, never return None.
    # The remaining dispositions (unknown value, and the known-but-unwired
    # 'lerobot' backend) are completed in the next task.
    raise NotImplementedError(
        f"{POLICY_BACKEND_ENV_VAR} selector is incomplete: no branch handled the "
        "selection. This is a bug in policy/factory.py, not a configuration error."
    )
