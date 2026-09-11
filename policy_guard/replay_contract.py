"""Immutable offline replay contracts. Model imports are local to worker helpers.

This module never compares backends, approves evidence, or imports robot code.
TorchFunctionMode and module hooks observe operations without replacing upstream
functions. A recorded float32 output alone is deliberately insufficient evidence.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import struct
import tempfile
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

JOINT_ORDER = (
    "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
    "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
)
CAMERA_ORDER = ("front", "wrist")
RAW_SHAPE = (1, 40, 132)
DECODED_SHAPE = (16, 6)
MAX_JSON = 16 * 1024**2
MAX_ARRAY_BYTES = 128 * 1024**2
NATIVE_PIN = "23ace64f17aa5015259b8609d371eb61a357c776"
NATIVE_SOURCE_DIGEST = "b18c8578c077cddf02705b80da815e5d838752cab9f391c19e4edca7a6e74a40"
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


def configuration_value(value):
    """Serialize actual configuration without unstable repr/object addresses."""
    if hasattr(value, "__dataclass_fields__"):
        return configuration_value(asdict(value))
    if isinstance(value, Enum):
        return configuration_value(value.value)
    if isinstance(value, dict):
        return {str(key): configuration_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [configuration_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path) or (type(value).__module__ == "torch" and type(value).__name__ == "dtype"):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported effective configuration type: {type(value).__name__}")


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


def verify_file(root: Path | str, reference: dict) -> bytes:
    path = contained(root, reference["path"])
    data = capture_bytes(path)
    if hashlib.sha256(data).hexdigest() != reference["sha256"]:
        raise ValueError(f"input/result digest changed: {reference['path']}")
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


def write_tensors(workspace: Path | str, arrays: dict[str, np.ndarray]) -> dict:
    if not arrays:
        raise ValueError("missing tensors")
    for key, value in arrays.items():
        if not re.fullmatch(r"[a-zA-Z0-9_.-]+", key):
            raise ValueError(f"invalid tensor key: {key}")
        if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
            raise ValueError(f"non-numeric/nonfinite tensor: {key}")
    if sum(value.nbytes for value in arrays.values()) > MAX_ARRAY_BYTES:
        raise ValueError("excessive tensor payload")
    stream = io.BytesIO()
    np.savez(stream, **arrays)
    data = stream.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    relative = f"tensors/{digest}.npz"
    path = contained(workspace, relative)
    try:
        _publish(path, data)
    except FileExistsError:
        if sha256_file(path) != digest:
            raise ValueError("content-addressed tensor conflict") from None
    return {
        "path": relative, "sha256": digest,
        "arrays": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in arrays.items()
        },
    }


def read_tensors(workspace: Path | str, reference: dict) -> dict:
    arrays = load_numeric(
        verify_file(workspace, reference), expected_keys=set(reference["arrays"])
    )
    for key, value in arrays.items():
        if reference["arrays"][key] != {"shape": list(value.shape), "dtype": str(value.dtype)}:
            raise ValueError(f"tensor shape/dtype mismatch: {key}")
    return arrays


@dataclass(frozen=True)
class ReplayProfile:
    backend: str
    purpose: str
    observations: dict

    def to_dict(self):
        return {"backend": self.backend, "purpose": self.purpose, **self.observations}


@dataclass
class ReplayManifest:
    session: str
    stage: str
    input_fingerprint: str
    expected_cases: list
    started_at: str = field(default_factory=now)
    schema_version: int = 1
    status: str = "not_run"
    ended_at: str | None = None
    evidence_kind: str = "real_model"
    profile_fingerprint: str | None = None
    configuration_fingerprint: str | None = None
    profile: dict = field(default_factory=dict)
    executed_cases: list = field(default_factory=list)
    cases: list = field(default_factory=list)
    prerequisite_errors: list = field(default_factory=list)
    resources: dict = field(default_factory=dict)
    failure_ledger: list = field(default_factory=list)


def base_case(key: dict) -> dict:
    return {"record": key["record"], "seed": key["seed"]}


def validate_schedule(lock: dict, cases: list, kind: str) -> None:
    if lock["joint_order"] != list(JOINT_ORDER) or lock["camera_order"] != list(CAMERA_ORDER):
        raise ValueError("joint/camera permutation")
    records = lock["records"]
    if len(records) != 120:
        raise ValueError("exactly 120 records required")
    expected = []
    for index, entry in enumerate(records):
        if entry["file"] != f"record_{index:04d}.npz":
            raise ValueError("record membership/order changed")
        seeds = entry["seeds"]
        if len(seeds) != 5 or len(set(seeds)) != 5 or any(type(seed) is not int for seed in seeds):
            raise ValueError("five distinct integer seeds required")
        expected.extend({"record": entry["file"], "seed": seed} for seed in seeds)
    if lock["schedule"] != expected or len(expected) != 600:
        raise ValueError("locked schedule differs from exact 600 pairs")
    if kind == "replay":
        valid = cases == expected
    elif kind == "tracer":
        valid = len(cases) == 1 and cases[0] in expected
    elif kind == "repeatability":
        valid = any(cases == group["cases"] for group in repeatability_schedule(lock)["groups"])
    else:
        raise ValueError(f"unknown schedule kind: {kind}")
    if not valid:
        raise ValueError(f"incomplete or altered {kind} schedule")


def repeatability_schedule(lock: dict) -> dict:
    validate_schedule(lock, lock["schedule"], "replay")
    warm, cold = [], []
    for index in (0, 60, 80, 90, 100, 119):
        record = lock["records"][index]
        first, second = record["seeds"][:2]
        warm.extend({"record": record["file"], "seed": first, "mode": "warm", "repeat": i} for i in range(5))
        warm.append({"record": record["file"], "seed": second, "mode": "changed", "repeat": 0})
        cold.extend({"id": f"cold-{index:04d}-{i}", "cases": [{"record": record["file"], "seed": first, "mode": "cold", "repeat": i}]} for i in range(2))
    return {"kind": "repeatability", "groups": [{"id": "warm", "cases": warm}, *cold]}


def profile_configuration(profile: dict) -> dict:
    # Operation counts can vary with the input. Every individual observation is
    # still retained and validated; it is not part of semantic configuration.
    return {key: value for key, value in profile.items() if key not in (
        "sdpa_calls", "floating_operation_count", "non_fp32_operations",
    )}


def execute_cases(report, workspace, lock, cases, trace_case, label):
    """Serialize each complete prediction/decode, publishing before the next case."""
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", label) or not cases:
        raise ValueError("invalid case stream label or empty schedule")
    for index, key in enumerate(cases):
        try:
            profile, tensors, entry = trace_case(key)
            reference = write_tensors(workspace, tensors)
            case = {"key": key, "record_sha256": entry["sha256"],
                    "instruction": entry["instruction"], "tensors": reference,
                    "observer_inert": profile["observer_inert"],
                    "profile_fingerprint": fingerprint_configuration(profile)}
            current = fingerprint_configuration(profile_configuration(profile))
            issue = None
            try:
                validate_profile(profile)
            except ValueError as exc:
                issue = PrerequisiteError(f"observed required profile unsupported: {exc}")
            if report.configuration_fingerprint and current != report.configuration_fingerprint:
                issue = ValueError("mixed effective configuration in case stream")
            if not report.profile:
                report.profile = profile
                report.profile_fingerprint = case["profile_fingerprint"]
                report.configuration_fingerprint = current
            payload = {**case, "schema_version": 1, "session": report.session,
                       "stage": report.stage, "input_fingerprint": report.input_fingerprint,
                       "evidence_kind": report.evidence_kind, "profile": profile,
                       "status": "complete" if issue is None else "not_run",
                       "error": None if issue is None else str(issue)}
            case["evidence"] = write_evidence(workspace, f"workers/cases/{label}-{index:04d}.json", payload)
            report.cases.append(case)
            report.executed_cases.append(key)
            if issue is not None:
                raise issue
        except Exception as exc:
            failure = write_evidence(workspace, f"workers/failures/{label}-{index:04d}.json", {
                "schema_version": 1, "session": report.session, "key": key,
                "evidence_kind": report.evidence_kind, "input_fingerprint": report.input_fingerprint,
                "status": "not_run" if prerequisite_exception(exc) else "failed",
                "type": type(exc).__name__, "message": str(exc), "recorded_at": now(),
            })
            report.failure_ledger.append(failure)
            raise


@dataclass(frozen=True)
class DecisionRecord:
    subject_digest: str
    decision_type: str
    decision: str
    operator: str
    decided_at: str
    evidence_reviewed: list
    rationale: str


def validate_profile(profile: dict) -> None:
    if profile.get("backend") not in ("native", "lerobot"):
        raise ValueError("unknown backend")
    if profile.get("purpose") not in ("diagnostic", "operational"):
        raise ValueError("unknown profile purpose")
    for key in ("source", "packages", "image_digest", "checkpoint_fingerprint",
                "backbone_fingerprint", "instrumentation_fingerprint", "device", "rng_algorithm"):
        if not profile.get(key):
            raise ValueError(f"missing observed identity: {key}")
    for key, expected in (
        ("raw_shape", list(RAW_SHAPE)), ("noise_shape", list(RAW_SHAPE)),
        ("decoded_shape", list(DECODED_SHAPE)), ("flow_steps", 4), ("noise_draws", 1),
        ("eval", True), ("seed_at_sampling_boundary", True), ("observer_inert", True),
        ("joint_order", list(JOINT_ORDER)), ("camera_order", list(CAMERA_ORDER)),
    ):
        if profile.get(key) != expected:
            raise ValueError(f"invalid observed {key}: {profile.get(key)!r}")
    if profile["purpose"] == "diagnostic":
        for key in ("parameter_dtypes", "input_dtypes", "backbone_dtypes", "compute_dtypes"):
            if profile.get(key) != ["torch.float32"]:
                raise ValueError(f"diagnostic {key} is not actual fp32: {profile.get(key)!r}")
        if any(dtype != "torch.float32" for dtype in profile.get("buffer_dtypes", [])):
            raise ValueError("diagnostic buffer_dtypes are not fp32")
        for key, expected in (
            ("noise_dtype", "torch.float32"), ("raw_dtype", "torch.float32"),
            ("autocast", False), ("tf32", False), ("attention", ["sdpa"]),
        ):
            if profile.get(key) != expected:
                raise ValueError(f"diagnostic {key} mismatch: {profile.get(key)!r}")
        if profile.get("sdpa_calls", 0) < 1:
            raise ValueError("no actual SDPA calls observed")


def validate_replay_manifest(
    report: dict, workspace: Path | str, expected_cases: list, *,
    expected_session: str, input_lock: dict, backend: str, purpose: str,
    checkpoint_fingerprint: str, image_digest: str, source_files: dict, device: str,
) -> None:
    if report.get("evidence_kind") != "real_model":
        raise ValueError("only real_model evidence may complete a replay")
    if report.get("status") != "complete":
        raise ValueError("replay did not complete")
    if not expected_cases or report.get("expected_cases") != expected_cases:
        raise ValueError("expected coverage mismatch")
    if report.get("executed_cases") != expected_cases or len(report.get("cases", [])) != len(expected_cases):
        raise ValueError("executed coverage is incomplete")
    if report.get("prerequisite_errors"):
        raise ValueError("unresolved worker errors")
    if report.get("session") != expected_session or report.get("stage") != purpose:
        raise ValueError("worker session/stage differs from launcher")
    if report.get("input_fingerprint") != input_lock["fingerprint"]:
        raise ValueError("worker input fingerprint differs from locked input")
    if checkpoint_fingerprint != input_lock["checkpoint_fingerprint"]:
        raise ValueError("launcher checkpoint differs from input lock")
    observed = report["profile"]
    for key, expected in (
        ("backend", backend), ("purpose", purpose), ("checkpoint_fingerprint", checkpoint_fingerprint),
        ("image_digest", image_digest), ("owned_source_files", source_files),
        ("owned_source_fingerprint", fingerprint_configuration(source_files)), ("device", device),
        ("joint_order", input_lock["joint_order"]), ("camera_order", input_lock["camera_order"]),
    ):
        if observed.get(key) != expected:
            raise ValueError(f"worker profile {key} differs from launcher")
    for key in ("input_fingerprint", "session", "started_at", "ended_at"):
        if not report.get(key):
            raise ValueError(f"missing manifest {key}")
    if datetime.fromisoformat(report["ended_at"]) < datetime.fromisoformat(report["started_at"]):
        raise ValueError("invalid replay chronology")
    validate_profile(report["profile"])
    if fingerprint_configuration(report["profile"]) != report["profile_fingerprint"]:
        raise ValueError("profile fingerprint mismatch")
    if report.get("configuration_fingerprint") and fingerprint_configuration(profile_configuration(report["profile"])) != report["configuration_fingerprint"]:
        raise ValueError("manifest configuration differs from bound profile")
    for expected, case in zip(expected_cases, report["cases"], strict=True):
        if case["key"] != expected:
            raise ValueError("case coverage mismatch")
        if base_case(expected) not in input_lock["schedule"]:
            raise ValueError("case is outside locked schedule")
        record = next((r for r in input_lock["records"] if r["file"] == expected["record"]), None)
        if record is None or case.get("record_sha256") != record["sha256"]:
            raise ValueError("case record digest differs from locked record")
        if case.get("instruction") != record["instruction"]:
            raise ValueError("case instruction differs from locked record")
        if "evidence" in case:
            saved = read_json(verify_file(workspace, case["evidence"]))
            for key, value in case.items():
                if key != "evidence" and saved.get(key) != value:
                    raise ValueError("durable case evidence differs from manifest")
            for key, value in (("session", expected_session), ("stage", purpose),
                               ("input_fingerprint", input_lock["fingerprint"]),
                               ("status", "complete"), ("evidence_kind", "real_model")):
                if saved.get(key) != value:
                    raise ValueError(f"durable case {key} differs from launcher")
            validate_profile(saved["profile"])
            if fingerprint_configuration(saved["profile"]) != case["profile_fingerprint"]:
                raise ValueError("durable profile digest mismatch")
            if fingerprint_configuration(profile_configuration(saved["profile"])) != report["configuration_fingerprint"]:
                raise ValueError("mixed effective configuration in manifest")
        elif len(expected_cases) > 1 or report.get("configuration_fingerprint"):
            raise ValueError("missing durable per-case evidence")
        arrays = read_tensors(workspace, case["tensors"])
        for key, shape in (("raw", RAW_SHAPE), ("noise", RAW_SHAPE), ("decoded", DECODED_SHAPE)):
            if key not in arrays or arrays[key].shape != shape:
                raise ValueError(f"missing/wrong full {key} shape")
            if arrays[key].dtype != np.float32:
                raise ValueError(f"unexpected serialized {key} dtype")
        if not case.get("observer_inert") or not case.get("record_sha256"):
            raise ValueError("missing input/observer evidence")


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


def load_input_lock(corpus: Path | str, checkpoint: Path | str) -> dict:
    corpus, checkpoint = Path(corpus), Path(checkpoint)
    manifest = read_json(corpus / "manifest.json")
    entries = manifest["records"]
    if manifest.get("record_count") != 120 or len(entries) != 120:
        raise ValueError("this session requires exactly 120 frozen records")
    if manifest.get("seed_verdict") != "not-honored" or manifest.get("seed_options_sent") is not True:
        raise ValueError("frozen seed provenance changed")
    if manifest.get("frame_shape") != [480, 640, 3]:
        raise ValueError("frozen camera shape changed")
    if manifest.get("action_modality_layout") != {"single_arm": [0, 5], "gripper": [5, 6]}:
        raise ValueError("frozen joint layout changed")
    records, schedule, seen = [], [], set()
    for i, entry in enumerate(entries):
        name = entry["file"]
        if name != f"record_{i:04d}.npz" or name in seen:
            raise ValueError("frozen record order/membership changed")
        seen.add(name)
        if len(entry["seeds"]) != 5 or len(set(entry["seeds"])) != 5:
            raise ValueError("five distinct stored seeds required")
        reference = {"path": name, "sha256": sha256_file(contained(corpus, name)), **entry}
        records.append(reference)
        schedule.extend({"record": name, "seed": seed} for seed in entry["seeds"])
    refs, fp32_bytes = checkpoint_inventory(checkpoint)
    result = {
        "schema_version": 1,
        "corpus_manifest": {"path": "manifest.json", "sha256": sha256_file(corpus / "manifest.json")},
        "records": records, "schedule": schedule, "checkpoint_files": refs,
        "checkpoint_fp32_tensor_bytes": fp32_bytes,
        "checkpoint_fingerprint": fingerprint_configuration(refs),
        "joint_order": list(JOINT_ORDER), "camera_order": list(CAMERA_ORDER),
        "historical_seed_verdict": manifest["seed_verdict"],
        "historical_seed_options_sent": manifest["seed_options_sent"],
    }
    result["fingerprint"] = fingerprint_configuration(result)
    validate_schedule(result, schedule, "replay")
    return result


def load_case(corpus: Path, lock: dict, key: dict) -> tuple[dict, dict]:
    key = base_case(key)
    if key not in lock["schedule"]:
        raise ValueError("case not in locked schedule")
    entry = next(item for item in lock["records"] if item["file"] == key["record"])
    arrays = load_numeric(verify_file(corpus, entry), frozen_instruction=True)
    if str(arrays.pop("instruction", None)) != entry["instruction"]:
        raise ValueError("stored instruction disagrees with locked manifest")
    expected = {
        "video_front": ((480, 640, 3), "uint8"), "video_wrist": ((480, 640, 3), "uint8"),
        "state": ((6,), "float32"), "ground_truth_action": ((6,), "float32"),
        "action_samples": ((5, 16, 6), "float32"), "seeds": ((5,), "int64"),
        "episode_index": ((), "int64"), "frame_index": ((), "int64"),
        "task_index": ((), "int64"),
    }
    if set(arrays) != set(expected):
        raise ValueError(f"unexpected frozen arrays: {set(arrays) ^ set(expected)}")
    for name, (shape, dtype) in expected.items():
        if arrays[name].shape != shape or str(arrays[name].dtype) != dtype:
            raise ValueError(f"frozen shape/dtype mismatch: {name}")
    for name in ("episode_index", "frame_index", "task_index"):
        if int(arrays[name]) != entry[name]:
            raise ValueError(f"record metadata changed: {name}")
    if arrays["seeds"].tolist() != entry["seeds"] or not entry["instruction"]:
        raise ValueError("record seed/instruction mismatch")
    return arrays, entry


def package_digest(package: Path) -> str:
    rows = [
        f"{sha256_file(path)}  {path.relative_to(package.parent).as_posix()}"
        for path in package.rglob("*.py") if path.is_file()
    ]
    return hashlib.sha256(("\n".join(sorted(rows)) + "\n").encode()).hexdigest()


def floating_dtypes(value) -> set[str]:
    import torch

    if isinstance(value, torch.Tensor):
        return {str(value.dtype)} if value.is_floating_point() else set()
    if hasattr(value, "items"):
        return set().union(*(floating_dtypes(v) for v in value.values()))
    if isinstance(value, (tuple, list)):
        return set().union(*(floating_dtypes(v) for v in value))
    return set()


def sampling_observer(seed: int, *, capture: bool = True):
    """Observe actual randn + SDPA through PyTorch's documented mode API.

    Seeding belongs immediately before the sampler's real draw. Inertness is
    measured against the same seed-only mode, which does no capture or cloning.
    No draw is substituted, predicted, or discarded.
    """
    import torch
    import torch.nn.functional as functional
    from torch.overrides import TorchFunctionMode
    from torch.utils._python_dispatch import TorchDispatchMode

    class ComputeObserver(TorchDispatchMode):
        def __init__(self, owner):
            self.owner = owner

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            self.owner.observe_context()
            dtypes = floating_dtypes(args) | floating_dtypes(kwargs)
            result = func(*args, **kwargs)
            dtypes.update(floating_dtypes(result))
            if dtypes:
                self.owner.floating_operation_count += 1
                self.owner.compute_dtypes.update(dtypes)
                if dtypes != {"torch.float32"} and len(self.owner.non_fp32_operations) < 64:
                    self.owner.non_fp32_operations.append({"operation": str(func), "dtypes": sorted(dtypes)})
            return result

    class Observer(TorchFunctionMode):
        def __init__(self):
            super().__init__()
            self.noise = None
            self.noise_draws = 0
            self.sdpa_calls = 0
            self.compute_dtypes = set()
            self.autocast = False
            self.tf32 = False
            self.noise_device = None
            self.rng_before = None
            self.rng_after = None
            self.floating_operation_count = 0
            self.non_fp32_operations = []
            self.dispatch = ComputeObserver(self) if capture else None

        def __enter__(self):
            super().__enter__()
            if self.dispatch is not None:
                self.dispatch.__enter__()
            return self

        def __exit__(self, *exc):
            try:
                if self.dispatch is not None:
                    self.dispatch.__exit__(*exc)
            finally:
                super().__exit__(*exc)

        def observe_context(self):
            self.autocast |= torch.is_autocast_enabled("cuda") or torch.is_autocast_enabled("cpu")
            self.tf32 |= torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32

        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if capture:
                # Function mode sees autocast before dispatcher lowering; the
                # dispatch mode observes actual operations, including direct ATen.
                self.observe_context()
            if func is torch.randn:
                # This mode surrounds inference only, never model initialization.
                shape = kwargs.get("size", args[0] if args else ())
                if isinstance(shape, int):
                    shape = args
                if tuple(shape) != RAW_SHAPE:
                    raise PrerequisiteError(f"unexpected actual sampler shape: {shape}")
                self.noise_draws += 1
                if self.noise_draws != 1:
                    raise PrerequisiteError("multiple sampler noise draws observed")
                device = torch.device(kwargs.get("device", "cpu"))
                if device.type == "cuda":
                    with torch.cuda.device(device):
                        torch.cuda.manual_seed(seed)
                    rng = lambda: torch.cuda.get_rng_state(device)
                else:
                    torch.random.default_generator.manual_seed(seed)
                    rng = torch.get_rng_state
                self.noise_device = str(device)
                if capture:
                    self.rng_before = rng().clone()
                output = func(*args, **kwargs)
                if capture:
                    self.noise = output.detach().clone()
                    self.rng_after = rng().clone()
                return output
            if capture and func is functional.scaled_dot_product_attention:
                self.sdpa_calls += 1
                self.compute_dtypes.update(floating_dtypes(args))
                output = func(*args, **kwargs)
                self.compute_dtypes.update(floating_dtypes(output))
                return output
            return func(*args, **kwargs)

    return Observer()


def observed_model(model):
    """Compose a model collaborator, forwarding get_action and its result intact."""
    import torch

    class ObservedModel(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.last_raw = None
            self.last_inputs = None

        @property
        def config(self):
            return self.inner.config

        def get_action(self, *args, **kwargs):
            result = self.inner.get_action(*args, **kwargs)
            self.last_raw = result["action_pred"].detach().clone()
            self.last_inputs = args[0] if args else kwargs.get("inputs")
            return result

    return ObservedModel(model)


def to_numpy(tensor):
    """Lossless numeric serialization of bf16 values; original dtype is separate."""
    import torch

    tensor = tensor.detach().cpu()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    return tensor.numpy().copy()


def trace_prediction(adapter, arrays, entry, seed: int, identity: dict) -> tuple[dict, dict]:
    """Observe one real complete path, then check observer inertness within backend."""
    import torch

    model = adapter.model
    raw_model = adapter.raw_model
    steps, backbone_dtypes, input_dtypes = [], set(), set()

    def before_backbone(module, args):
        input_dtypes.update(floating_dtypes(args))

    def after_backbone(module, args, result):
        backbone_dtypes.update(floating_dtypes(result))

    def before_action(module, args):
        steps.append(int(args[1].flatten()[0]))

    hooks = [
        raw_model.backbone.register_forward_pre_hook(before_backbone),
        raw_model.backbone.register_forward_hook(after_backbone),
        raw_model.action_head.action_encoder.register_forward_pre_hook(before_action),
    ]
    try:
        with torch.inference_mode(), sampling_observer(seed) as observer:
            decoded = adapter.predict(arrays, entry)
        raw = adapter.observed.last_raw
    finally:
        for hook in hooks:
            hook.remove()
    if observer.noise is None or raw is None:
        raise PrerequisiteError("actual raw output/noise observation unavailable")
    # Control uses the identical seed boundary, but has no tensor/module observer.
    with torch.inference_mode(), sampling_observer(seed, capture=False):
        control = adapter.predict(arrays, entry)
    inert = np.array_equal(decoded, control) and torch.equal(raw, adapter.observed.last_raw)
    buffers = [v for v in raw_model.buffers() if v.is_floating_point()]
    observations = {
        **identity,
        "parameter_dtypes": sorted({str(p.dtype) for p in raw_model.parameters() if p.is_floating_point()}),
        "buffer_dtypes": sorted({str(p.dtype) for p in buffers}),
        "input_dtypes": sorted(input_dtypes), "backbone_dtypes": sorted(backbone_dtypes),
        "compute_dtypes": sorted(observer.compute_dtypes),
        "floating_operation_count": observer.floating_operation_count,
        "non_fp32_operations": observer.non_fp32_operations,
        "noise_dtype": str(observer.noise.dtype), "raw_dtype": str(raw.dtype),
        "attention": sorted(adapter.attention_implementations()),
        "sdpa_calls": observer.sdpa_calls, "eval": all(not m.training for m in model.modules()),
        "autocast": observer.autocast, "tf32": observer.tf32,
        "flow_steps": len(steps), "step_buckets": steps,
        "raw_shape": list(raw.shape), "noise_shape": list(observer.noise.shape),
        "decoded_shape": list(decoded.shape), "noise_draws": observer.noise_draws,
        "seed_at_sampling_boundary": observer.noise_draws == 1,
        "observer_inert": inert, "device": observer.noise_device,
        "rng_algorithm": f"torch.default_generator.{torch.device(observer.noise_device).type}",
        "joint_order": list(JOINT_ORDER), "camera_order": list(CAMERA_ORDER),
        "path": adapter.path,
    }
    profile = ReplayProfile(adapter.backend, adapter.purpose, observations).to_dict()
    tensors = {
        "raw": to_numpy(raw), "noise": to_numpy(observer.noise), "decoded": decoded,
        "rng_before": to_numpy(observer.rng_before), "rng_after": to_numpy(observer.rng_after),
    }
    # Persist actual observed controls even when a required profile is unsupported.
    return profile, tensors


def runtime_identity(backend: str, image_digest: str, lock: dict) -> dict:
    import importlib.metadata
    import importlib.util
    import platform
    from huggingface_hub.constants import HF_HUB_CACHE

    package = "gr00t" if backend == "native" else "lerobot"
    spec = importlib.util.find_spec(package)
    if spec is None or spec.origin is None:
        raise PrerequisiteError(f"missing installed {package}")
    source = package_digest(Path(spec.origin).parent)
    if backend == "native":
        if platform.python_version_tuple()[:2] != ("3", "10"):
            raise PrerequisiteError("native producer must use Python 3.10")
        if source != NATIVE_SOURCE_DIGEST:
            raise PrerequisiteError("native package differs from pinned source content")
    elif importlib.metadata.version("lerobot") != "0.6.1":
        raise PrerequisiteError("LeRobot version must be 0.6.1")
    hub = Path(HF_HUB_CACHE).resolve()
    model_dir = hub / "models--nvidia--Cosmos-Reason2-2B"
    snapshot = model_dir / "snapshots" / BACKBONE_REVISION
    if not snapshot.is_dir() or (model_dir / "refs/main").read_text().strip() != BACKBONE_REVISION:
        raise PrerequisiteError("default offline backbone differs from pinned revision")
    backbone_files = []
    for path in sorted(snapshot.rglob("*")):
        if path.is_file():
            if not path.resolve().is_relative_to(model_dir.resolve()):
                raise ValueError("backbone symlink escapes model cache")
            backbone_files.append({"path": str(path.relative_to(snapshot)), "sha256": sha256_file(path)})
    if not backbone_files:
        raise PrerequisiteError("empty backbone cache")
    packages = dict(sorted(
        (dist.metadata["Name"], dist.version)
        for dist in importlib.metadata.distributions() if dist.metadata["Name"]
    ))
    owned_root = Path(__file__).resolve().parents[1]
    owned_sources = {
        name: sha256_file(owned_root / name)
        for name in (
            "policy_guard/replay_contract.py", "policy_guard/groot_guard.py",
            "scripts/replay_groot_native.py", "scripts/replay_checkpoint_parity.py",
            "docker/lerobot-policy/replay_checkpoint.py", "docker/lerobot-policy/server.py",
            "policy/lerobot/features.py",
        )
    }
    return {
        "source": {"package": package, "sha256": source, "native_pin": NATIVE_PIN if backend == "native" else None},
        "packages": packages, "python": platform.python_version(),
        "image_digest": image_digest,
        "checkpoint_fingerprint": lock["checkpoint_fingerprint"],
        "backbone_fingerprint": fingerprint_configuration(backbone_files),
        "backbone_revision": BACKBONE_REVISION, "backbone_files": backbone_files,
        "instrumentation_fingerprint": sha256_file(Path(__file__)),
        "owned_source_files": owned_sources,
        "owned_source_fingerprint": fingerprint_configuration(owned_sources),
    }


def prerequisite_exception(exc: Exception) -> bool:
    return isinstance(exc, (PrerequisiteError, FileNotFoundError, ImportError, MemoryError)) or type(exc).__name__ in (
        "OutOfMemoryError", "LocalEntryNotFoundError", "OfflineModeIsEnabled",
    )


def worker_main(backend: str, adapter_factory, stock_loader=None) -> int:
    """Immutable worker lifecycle; failed profiles retain measured evidence."""
    import argparse
    import resource
    import time
    import traceback

    parser = argparse.ArgumentParser(description="Isolated arm-free checkpoint replay worker")
    for name in ("input-lock", "schedule", "workspace", "corpus", "checkpoint", "output-manifest", "image-digest"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--profile", choices=("diagnostic", "operational", "stock-capacity"), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    workspace = Path(args.workspace)
    destination = contained(workspace, args.output_manifest)
    if destination.exists():
        print(json.dumps({"status": "failed", "error": "immutable worker destination exists"}))
        return 1
    start = time.perf_counter()
    report = None
    try:
        session = read_json(workspace / "session.json")
        lock = read_json(args.input_lock)
        schedule_record = read_json(args.schedule)
        schedule = schedule_record["cases"]
        report = ReplayManifest(
            session=session["session_id"], stage=args.profile,
            input_fingerprint=lock["fingerprint"], expected_cases=schedule,
        )
        current = load_input_lock(args.corpus, args.checkpoint)
        if current != lock:
            raise ValueError("input lock changed before worker inference")
        validate_schedule(lock, schedule, schedule_record.get("kind", "tracer"))
        if schedule_record.get("session") != session["session_id"]:
            raise ValueError("schedule belongs to another session")
        if schedule_record.get("input_fingerprint", lock["fingerprint"]) != lock["fingerprint"]:
            raise ValueError("schedule belongs to another input lock")
        import torch

        identity = runtime_identity(backend, args.image_digest, lock)
        report.resources["runtime_identity"] = identity
        report.resources["checkpoint_fp32_tensor_bytes"] = lock["checkpoint_fp32_tensor_bytes"]
        report.resources["requested_device"] = args.device
        report.resources["host_meminfo_before"] = Path("/proc/meminfo").read_text()
        report.resources["cpu_model"] = next((line for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), "unavailable")
        for name in ("memory.max", "memory.swap.max"):
            path = Path("/sys/fs/cgroup") / name
            if path.exists():
                report.resources[name] = path.read_text().strip()
        report.resources["stock_two_model_tensor_lower_bound_bytes"] = lock["checkpoint_fp32_tensor_bytes"] * 3 // 2
        if args.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise PrerequisiteError("CUDA unavailable in pinned worker")
            free, total = torch.cuda.mem_get_info()
            report.resources.update(device_name=torch.cuda.get_device_name(), free_before_bytes=free, total_bytes=total)
            torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_num_threads(4)
        if args.profile == "stock-capacity":
            if stock_loader is None:
                raise ValueError("stock capacity belongs to native environment")
            models = stock_loader(Path(args.checkpoint), args.device)
            report.resources["resident_models"] = len(models)
            report.resources["resident_model_storage"] = []
            for loaded in models:
                model = loaded if isinstance(loaded, torch.nn.Module) else loaded.model
                parameters = list(model.parameters())
                report.resources["resident_model_storage"].append({
                    "devices": sorted({str(p.device) for p in parameters}),
                    "dtypes": sorted({str(p.dtype) for p in parameters if p.is_floating_point()}),
                    "parameter_bytes": sum(p.numel() * p.element_size() for p in parameters),
                })
            report.status = "complete"
        else:
            adapter = adapter_factory(Path(args.checkpoint), args.profile, args.device)
            report.resources["load_seconds"] = time.perf_counter() - start
            effective = configuration_value(adapter.effective_configuration())
            identity["effective_configuration"] = effective
            identity["effective_configuration_fingerprint"] = fingerprint_configuration(effective)
            def trace_case(key):
                arrays, entry = load_case(Path(args.corpus), lock, key)
                profile, tensors = trace_prediction(adapter, arrays, entry, key["seed"], identity)
                return profile, tensors, entry
            execute_cases(report, workspace, lock, schedule, trace_case, Path(args.output_manifest).stem)
            report.status = "complete"
    except Exception as exc:
        if report is None:
            print(json.dumps({"status": "not_run", "message": f"not run: {type(exc).__name__}: {exc}"}))
            return 2
        unavailable = prerequisite_exception(exc)
        report.status = "not_run" if unavailable else "failed"
        report.prerequisite_errors.append({
            "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(),
        })
    finally:
        if report is not None:
            report.ended_at = now()
            report.resources["elapsed_seconds"] = time.perf_counter() - start
            report.resources["max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            try:
                import torch
                if torch.cuda.is_initialized():
                    report.resources["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                    report.resources["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            except ImportError:
                pass
            write_evidence(workspace, args.output_manifest, asdict(report))
    print(json.dumps({"status": report.status, "message": "not run" if report.status == "not_run" else report.status, "manifest": args.output_manifest}))
    return {"complete": 0, "failed": 1, "not_run": 2}[report.status]
