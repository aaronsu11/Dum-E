"""Pinned offline native GR00T processor cache."""
import os, tempfile
from contextlib import contextmanager
from pathlib import Path
from policy.evidence import PrerequisiteError
BACKBONE_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"

@contextmanager
def pinned_native_cache(hub_root=None):
    """Resolve the unchanged model ID as a local path in an owned temporary cwd.

    Transformers 4.57.3 skips tokenizer Hub metadata discovery for a local
    directory. The literal ID must stay unchanged for native backbone dispatch.
    The cached snapshot and its blob links remain read-only and unmodified.
    """
    if hub_root is None:
        from huggingface_hub.constants import HF_HUB_CACHE
        hub_root = HF_HUB_CACHE
    model = Path(hub_root).resolve() / "models--nvidia--Cosmos-Reason2-2B"
    snapshot = model / "snapshots" / BACKBONE_REVISION
    if not snapshot.is_dir() or not (model / "refs/main").is_file():
        raise PrerequisiteError("pinned native backbone snapshot is absent")
    if (model / "refs/main").read_text().strip() != BACKBONE_REVISION:
        raise PrerequisiteError("native backbone default revision differs from pin")
    if not snapshot.resolve().is_relative_to(model):
        raise ValueError("native snapshot escapes model cache")
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        if not (snapshot / name).is_file():
            raise PrerequisiteError(f"pinned snapshot missing {name}")
    for path in snapshot.rglob("*"):
        if path.is_symlink() and (not path.exists() or not path.resolve().is_relative_to(model)):
            raise ValueError(f"native snapshot blob escapes model cache: {path.name}")
    previous = os.open(".", os.O_RDONLY)
    try:
        with tempfile.TemporaryDirectory(prefix="dume-native-cache-", dir="/tmp") as temporary:
            root = Path(temporary)
            (root / "nvidia").mkdir()
            (root / "nvidia/Cosmos-Reason2-2B").symlink_to(snapshot, target_is_directory=True)
            try:
                os.chdir(root)
                yield snapshot
            finally:
                os.fchdir(previous)
    finally:
        os.close(previous)
