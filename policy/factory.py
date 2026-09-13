'Policy-backend selector.'

import os
from typing import Any

from shared import IPolicyBackend
from policy.configuration import PolicyDeployment

#: Name of the environment variable that selects the backend.
POLICY_BACKEND_ENV_VAR = "DUME_POLICY_BACKEND"

#: The allowlist. Anything not in here is rejected — never defaulted.
POLICY_BACKENDS = ("lerobot", "groot-native", "galaxea", "pi05-so101")

# Match dum_e.py and config.example.yaml; explicit selection overrides this.
DEFAULT_POLICY_BACKEND = "groot-native"


def make_policy_backend(*, deployment: PolicyDeployment | None = None, **kwargs: Any) -> IPolicyBackend:
    'Build the policy backend selected by ``DUME_POLICY_BACKEND``.'
    if deployment is not None:
        deployment.validate()
        # The explicit deployment cannot be silently changed by ambient flags.
        if os.getenv("DUME_ASYNC_INFERENCE", "0") == "1" and deployment.execution != "async":
            raise ValueError("Explicit execution conflicts with async DUME_ASYNC_INFERENCE")
        if deployment.backend == "isaac_groot":
            from policy.backends.isaac_groot.client import Gr00tRobotInferenceClient
            return Gr00tRobotInferenceClient(**kwargs)
        if deployment.backend == "galaxea":
            from policy.backends.galaxea.backend import GalaxeaPolicyBackend
            from embodiment.so_arm10x.mappings import galaxea as mapping
            return GalaxeaPolicyBackend(mapping=mapping, **kwargs)
        if deployment.backend == "lerobot" and deployment.transport == "grpc":
            if deployment.execution == "async":
                from policy.backends.lerobot.serialized_backend import SerializedLeRobotPolicyBackend
                return SerializedLeRobotPolicyBackend(**kwargs)
            from policy.backends.lerobot.backend import LeRobotPolicyBackend
            return LeRobotPolicyBackend(**kwargs)
        from embodiment.so_arm10x.mappings.lerobot import SO101Mapping
        mapping = SO101Mapping(deployment.checkpoint, calibration_path=kwargs.pop("calibration_path", None))
        if deployment.policy == "pi05":
            from policy.backends.lerobot.pi05_client import Pi05SO101PolicyBackend
            return Pi05SO101PolicyBackend(mapping=mapping, **kwargs)
        from policy.backends.lerobot.http_client import HTTPPolicyBackend
        if deployment.policy == "groot":
            kwargs.setdefault("port", 8081)
        return HTTPPolicyBackend(profile=deployment.checkpoint, mapping=mapping, **kwargs)
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
        if os.getenv("DUME_ASYNC_INFERENCE", "0") == "1":
            from policy.backends.lerobot.serialized_backend import SerializedLeRobotPolicyBackend
            return SerializedLeRobotPolicyBackend(**kwargs)
        from policy.backends.lerobot.backend import LeRobotPolicyBackend

        return LeRobotPolicyBackend(**kwargs)

    if backend == "pi05-so101":
        return make_policy_backend(deployment=PolicyDeployment(
            "so_arm10x", "lerobot", "pi05", "pi05-so101", transport="http"), **kwargs)

    if backend == "galaxea":
        return make_policy_backend(deployment=PolicyDeployment(
            "so_arm10x", "galaxea", "g05", "g05-so101"), **kwargs)

    if backend == "groot-native":
        if os.getenv("DUME_ASYNC_INFERENCE", "0") == "1":
            raise ValueError("Async picking currently requires the LeRobot backend")
        # Lazy-import inside the branch, for the same reason as above: the GR00T
        # transport (policy/backends/isaac_groot/service.py) depends only on msgpack/numpy/zmq and
        # must not be dragged in by a lerobot selection either.
        from policy.backends.isaac_groot.client import Gr00tRobotInferenceClient

        return Gr00tRobotInferenceClient(**kwargs)

    # Unreachable while every POLICY_BACKENDS member has a branch above. Kept so
    # that adding an allowlist entry without a branch fails loudly instead of
    # returning None into the robot action loop.
    raise NotImplementedError(
        f"{POLICY_BACKEND_ENV_VAR}={backend!r} is in POLICY_BACKENDS but has no "
        "branch in make_policy_backend(). This is a bug in policy/factory.py, "
        "not a configuration error."
    )
