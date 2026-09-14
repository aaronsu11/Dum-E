"""Explicit deployment selection, independent of process environment."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PolicyDeployment:
    embodiment: str
    backend: str
    policy: str
    checkpoint: str
    execution: str = "sync"
    transport: str = "native"

    def validate(self):
        supported = {
            ("isaac_groot", "groot", "groot-so101", "native"): {"sync"},
            ("lerobot", "groot", "groot-so101", "grpc"): {"sync", "async"},
            ("lerobot", "groot", "groot-so101", "http"): {"sync"},
            ("lerobot", "pi05", "pi05-so101", "http"): {"sync", "rtc"},
            ("lerobot", "molmoact2", "molmoact2-so101", "http"): {"sync"},
            ("galaxea", "g05", "g05-so101", "native"): {"sync"},
        }
        if self.embodiment != "so_arm10x":
            raise ValueError(f"Unimplemented embodiment: {self.embodiment}")
        modes = supported.get((self.backend, self.policy, self.checkpoint, self.transport), set())
        if self.execution not in modes:
            raise ValueError(f"Unsupported policy deployment: {self}")
        return self


def deployment_for_profile(profile, *, execution="sync"):
    choices = {
        "g05-so101": ("galaxea", "g05", "g05-so101", "native"),
        "groot-so101": ("lerobot", "groot", "groot-so101", "http"),
        "lerobot-gr00t": ("lerobot", "groot", "groot-so101", "grpc"),
        "pi05-so101": ("lerobot", "pi05", "pi05-so101", "http"),
        "molmoact2-so101": ("lerobot", "molmoact2", "molmoact2-so101", "http"),
    }
    if profile not in choices:
        raise ValueError(f"No arm deployment for profile: {profile}")
    backend, family, checkpoint, transport = choices[profile]
    return PolicyDeployment("so_arm10x", backend, family, checkpoint, execution, transport).validate()


def load_deployment(path):
    """Read a strict deployment document; reject unknown or missing fields."""
    import yaml
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("Deployment must be a mapping")
    try:
        return PolicyDeployment(**data).validate()
    except TypeError as exc:
        raise ValueError(f"Invalid deployment fields: {exc}") from exc


def profile_for_deployment(deployment):
    deployment.validate()
    if deployment.backend == "isaac_groot":
        raise ValueError("Native GR00T uses its existing native validation command")
    return "lerobot-gr00t" if deployment.transport == "grpc" else deployment.checkpoint
