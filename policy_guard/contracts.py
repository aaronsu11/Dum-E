"""Bounded numeric evidence IO and checkpoint identity. No hardware or model imports."""
from __future__ import annotations
import hashlib, io, json, math, os, struct, tempfile, zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np

JOINT_ORDER = (
    "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
    "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
)


CAMERA_ORDER = ("front", "wrist")


MAX_JSON = 16 * 1024**2


MAX_ARRAY_BYTES = 128 * 1024**2


BACKBONE_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"


class PrerequisiteError(RuntimeError):
    """A real prerequisite was unavailable; the requested stage was not run."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path | str | bytes) -> dict:
    data = path if isinstance(path, bytes) else capture_bytes(Path(path), MAX_JSON)
    if len(data) > MAX_JSON:
        raise ValueError("excessive JSON size")
    result = json.loads(
        data, object_pairs_hook=_unique_object,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    if not isinstance(result, dict):
        raise ValueError(f"expected JSON object: {path}")
    return result


def canonical(value: Any) -> bytes:
    if hasattr(value, "__dataclass_fields__"):
        value = asdict(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def fingerprint_configuration(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()




def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def contained(root: Path | str, relative: str) -> Path:
    root = Path(root).resolve()
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or relative in ("", "."):
        raise ValueError(f"path escapes root: {relative}")
    result = root / rel
    if not result.resolve().is_relative_to(root):
        raise ValueError(f"symlink escapes root: {relative}")
    return result


def capture_bytes(path: Path, limit: int = MAX_ARRAY_BYTES) -> bytes:
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"excessive captured file size: {path}")
    return data




def _publish(path: Path, data: bytes) -> None:
    """fsync then link(2): the final name is never partial and never replaced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".replay-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            if os.geteuid() == 0 and "DUME_REPLAY_UID" in os.environ:
                os.fchown(stream.fileno(), int(os.environ["DUME_REPLAY_UID"]), int(os.environ["DUME_REPLAY_GID"]))
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temp)


def write_evidence(workspace: Path | str, relative: str, payload: Any) -> dict:
    data = canonical(payload) + b"\n"
    path = contained(workspace, relative)
    _publish(path, data)
    return {"path": relative, "sha256": hashlib.sha256(data).hexdigest()}


def load_numeric(source: Path | bytes, *, expected_keys: set[str] | None = None, frozen_instruction: bool = False) -> dict:
    data = source if isinstance(source, bytes) else capture_bytes(source)
    if len(data) > MAX_ARRAY_BYTES:
        raise ValueError("excessive tensor archive size")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or len(names) > 256:
            raise ValueError("duplicate/excessive tensor keys")
        if sum(item.file_size for item in archive.infolist()) > MAX_ARRAY_BYTES:
            raise ValueError("excessive declared tensor size")
        # Read NPY headers BEFORE numpy allocates based on declared shapes.
        for item in archive.infolist():
            with archive.open(item) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError("unsupported NPY version")
                text = frozen_instruction and item.filename == "instruction.npy" and dtype.kind == "U" and shape == ()
                if dtype.hasobject or (dtype.kind not in "biuf" and not text):
                    raise ValueError("only numeric non-object tensors are allowed")
                if math.prod(shape) * dtype.itemsize > MAX_ARRAY_BYTES:
                    raise ValueError("excessive declared array size")
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        if expected_keys is not None and set(archive.files) != expected_keys:
            raise ValueError("unexpected tensor keys")
        result = {key: archive[key] for key in archive.files}
    for key, value in result.items():
        if value.dtype.kind != "U" and not np.isfinite(value).all():
            raise ValueError(f"nonfinite tensor: {key}")
    return result






def checkpoint_inventory(checkpoint: Path) -> tuple[list, int]:
    index = read_json(checkpoint / "model.safetensors.index.json")
    shards = sorted(set(index["weight_map"].values()))
    files = set(shards)
    files.update(
        str(path.relative_to(checkpoint)) for path in checkpoint.rglob("*")
        if path.is_file() and ".cache" not in path.parts and
        path.suffix in (".json", ".yaml", ".yml")
    )
    for name in ("config.json", "processor_config.json", "statistics.json", "embodiment_id.json"):
        if name not in files:
            raise FileNotFoundError(checkpoint / name)
    refs = [
        {"path": name, "sha256": sha256_file(contained(checkpoint, name)),
         "bytes": contained(checkpoint, name).stat().st_size}
        for name in sorted(files)
    ]
    fp32_bytes = 0
    for name in shards:
        with contained(checkpoint, name).open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            if length > MAX_JSON:
                raise ValueError("excessive safetensors header")
            header = json.loads(stream.read(length), object_pairs_hook=_unique_object)
        for key, value in header.items():
            if key != "__metadata__" and value["dtype"] in ("F32", "F16", "BF16"):
                fp32_bytes += math.prod(value["shape"]) * 4
    return refs, fp32_bytes


def require(condition, message):
    if not condition:
        raise ValueError(message)
