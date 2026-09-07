#!/usr/bin/env python3
"""Capture the frozen v1.0 observation-to-action corpus (LR-05 / D-06 / D-07 / D-09).

Reads a stratified subset of the ``aaronsu11/so101_fruit`` LeRobot dataset
DIRECTLY from raw parquet + AV1 video, pushes each frame through the running
v1.0 ``groot-native`` policy server (Isaac-GR00T n1.7 ZMQ ``:5555``), and writes
one ``.npz`` per frame holding the observation, the action chunk(s) that server
produced, and the dataset's ground-truth action.

WHY RAW PARQUET + AV1 (do not "simplify" this to ``LeRobotDataset``):
  1. ``lerobot`` 0.6.x's ``LeRobotDataset._load_metadata`` calls
     ``check_version_compatibility`` against ``CODEBASE_VERSION = "v3.0"`` and
     hard-raises ``BackwardCompatibilityError`` on this dataset's
     ``codebase_version: v2.1``.
  2. The 0.6.1 bump REMOVES ``torchcodec``/``av``/``pandas``/``pyarrow``/
     ``datasets``/``jsonlines`` from the client venv (0.6.0 moved dataset deps
     behind the ``dataset`` extra). This script therefore imports NO ``lerobot``
     module at all, and must run BEFORE that bump.

The v2.1 dataset and the local checkpoint are the ground-truth reference for the
whole milestone: this script is STRICTLY READ-ONLY against them. It never writes,
converts, re-pushes, or runs ``convert_dataset_v21_to_v30.py``.

Usage:
    # 0. start the v1.0 policy server first (see README "Start the policy server")
    #    docker start gr00t-server        # or the documented `docker run` command

    # 1. one-frame smoke (falsifies the reader path + server liveness cheaply)
    uv run python scripts/capture_frozen_corpus.py --records 1 --samples-per-obs 1 \
        --out corpus/smoke_one

    # 2. D-07 seed-reproducibility verdict (double run + different-seed control)
    uv run python scripts/capture_frozen_corpus.py --seed-check-only --out corpus/smoke_seed

    # 3. bulk stratified capture (the deliverable)
    uv run python scripts/capture_frozen_corpus.py --records 120 --samples-per-obs 5 \
        --out corpus/frozen_v1_0

    # 4. prove it is replayable
    uv run python scripts/verify_frozen_corpus.py --corpus corpus/frozen_v1_0

On-disk layout written under ``--out``:
    manifest.json      corpus_schema_version, dataset provenance (repo id +
                       resolved revision SHA + codebase version), instruction,
                       samples_per_observation, record_count, seed_verdict +
                       evidence, stratification recipe, per-record provenance
    record_NNNN.npz    video_wrist/video_front (480,640,3) uint8, state (6,) f32,
                       ground_truth_action (6,) f32, action_samples (S,T,6) f32,
                       seeds (S,) i64, episode_index/frame_index/task_index i64,
                       instruction (0-d unicode)

``action_samples`` keeps the RAW float32 chunk the server returned, laid out per
``meta/modality.json``: columns [0:5] are ``single_arm``, column [5:6] is
``gripper`` (also recorded as ``action_modality_layout`` in the manifest). This is
why the capture calls ``ExternalRobotInferenceClient.get_action`` directly rather
than ``Gr00tRobotInferenceClient.get_action`` — the wrapper passes no ``options``
(the only channel a seed can ride) and flattens the chunk into per-step python
floats, discarding the arrays Phase 7 needs to diff.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

# Ensure the repo root is importable regardless of CWD (the scripts/ dir would
# otherwise shadow the repo root on sys.path[0]).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mirror the wire contract from Dum-E's client-side serializer — NOT the
# server-only upstream ``gr00t`` package (not installed in the client uv env).
from policy.gr00t.service import ExternalRobotInferenceClient  # noqa: E402

# --- Pinned contract constants ----------------------------------------------

#: Corpus schema version written into (and asserted by) the manifest.
CORPUS_SCHEMA_VERSION = 1

#: The corpus source and ground-truth reference (CONTEXT.md D-09).
DEFAULT_REPO_ID = "aaronsu11/so101_fruit"

#: The dataset's codebase version. The AWS recipe that produced the checkpoint
#: REQUIRES v2.1; never convert the source to v3.0.
EXPECTED_CODEBASE_VERSION = "v2.1"

#: Flat observation keys, verbatim from scripts/test_live_policy_server.py:41-49.
ROBOT_STATE_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]
CAMERA_KEYS = ["wrist", "front"]

#: video key -> dataset feature name, verbatim from the dataset's meta/modality.json.
VIDEO_KEY_TO_FEATURE = {
    "wrist": "observation.images.wrist",
    "front": "observation.images.front",
}

#: PINNED N1.7 INFERENCE KEY — identical to embodiment/so_arm10x/controller.py:123.
#: NO ``.action.`` segment. tests/test_frozen_corpus.py proves the two assemblies
#: agree key-for-key so they cannot drift apart silently.
LANGUAGE_KEY = "annotation.human.task_description"

#: Modality layout of ``action_samples`` columns (meta/modality.json).
ACTION_MODALITY_LAYOUT = {"single_arm": [0, 5], "gripper": [5, 6]}

#: Modality keys the server returns in the action chunk.
MODALITY_KEYS = ("single_arm", "gripper")

#: Plausible spellings for a seed inside ``get_action``'s ``options`` dict. Whether
#: the upstream N1.7 server honours ANY of them is unverified (RESEARCH A4), so the
#: check tries each independently — a negative under one spelling is not evidence
#: the channel does not exist.
SEED_OPTION_KEYS = ("seed", "random_seed", "rng_seed")

#: Offset used for the different-seed control run in the seed check.
CONTROL_SEED_OFFSET = 987_654


# --- Verdict harness (shape copied from scripts/test_live_policy_server.py) ---


def _green(s: str) -> str:
    return f"\033[92m{s}\033[0m"


def _red(s: str) -> str:
    return f"\033[91m{s}\033[0m"


class Checks:
    """Numbered-check PASS/FAIL harness with a ``passed == total`` exit contract."""

    def __init__(self, total: int) -> None:
        self.total = total
        self.index = 0
        self.results: dict[str, bool] = {}

    def start(self, description: str) -> None:
        self.index += 1
        print(f"\n[{self.index}/{self.total}] {description} ...")

    def ok(self, name: str, message: str) -> bool:
        print(_green(f"  PASS: {message}"))
        self.results[name] = True
        return True

    def fail(self, name: str, message: str) -> bool:
        print(_red(f"  FAIL: {message}"))
        self.results[name] = False
        return False

    def report(self) -> int:
        print("\n" + "=" * 72)
        passed = sum(1 for ok in self.results.values() if ok)
        for name, ok in self.results.items():
            print(f"  {_green('PASS') if ok else _red('FAIL')}  {name}")
        print(f" {passed}/{len(self.results)} checks passed")
        print("=" * 72)
        return 0 if passed == len(self.results) else 1


# --- Dataset reading (raw parquet + AV1, no lerobot import) -------------------


@dataclass
class DatasetHandle:
    """Resolved, pinned view of the corpus source dataset."""

    repo_id: str
    revision: str
    codebase_version: str
    chunks_size: int
    fps: int
    total_episodes: int
    total_frames: int
    data_path: str
    video_path: str
    frame_shape: tuple[int, int, int]
    episode_lengths: dict[int, int] = field(default_factory=dict)
    episode_tasks: dict[int, str] = field(default_factory=dict)
    episodes_with_stats: set[int] = field(default_factory=set)
    tasks: dict[int, str] = field(default_factory=dict)


def _hf_download(repo_id: str, filename: str, revision: str | None = None) -> Path:
    """Download one dataset file from the Hub (read-only) and return its local path."""
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(repo_id, filename, repo_type="dataset", revision=revision)
    )


def _revision_from_snapshot_path(path: Path) -> str:
    """Extract the resolved commit SHA from a hf_hub cache snapshot path."""
    parts = path.parts
    if "snapshots" in parts:
        return parts[parts.index("snapshots") + 1]
    return ""


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def resolve_dataset(repo_id: str, revision: str | None = None) -> DatasetHandle:
    """Resolve the dataset's metadata and pin the revision SHA.

    Reads ``meta/info.json`` for the path templates + feature geometry,
    ``meta/episodes.jsonl`` for per-episode length/task, ``meta/tasks.jsonl`` for
    the instruction strings, and ``meta/episodes_stats.jsonl`` (this dataset has
    NO ``meta/stats.json``) so a sampled episode without stats is caught.
    """
    info_path = _hf_download(repo_id, "meta/info.json", revision)
    resolved_revision = _revision_from_snapshot_path(info_path)
    info = json.loads(info_path.read_text(encoding="utf-8"))

    codebase_version = info.get("codebase_version", "")
    if codebase_version != EXPECTED_CODEBASE_VERSION:
        raise RuntimeError(
            f"{repo_id} reports codebase_version {codebase_version!r}, expected "
            f"{EXPECTED_CODEBASE_VERSION!r}. The ground-truth reference must NOT be "
            f"converted — refusing to capture against an unexpected version."
        )

    wrist_feature = info["features"][VIDEO_KEY_TO_FEATURE["wrist"]]
    frame_shape = tuple(int(dim) for dim in wrist_feature["shape"])

    episode_lengths: dict[int, int] = {}
    episode_tasks: dict[int, str] = {}
    for row in _read_jsonl(_hf_download(repo_id, "meta/episodes.jsonl", revision)):
        episode_lengths[int(row["episode_index"])] = int(row["length"])
        tasks = row.get("tasks") or []
        if tasks:
            episode_tasks[int(row["episode_index"])] = str(tasks[0])

    tasks: dict[int, str] = {}
    for row in _read_jsonl(_hf_download(repo_id, "meta/tasks.jsonl", revision)):
        tasks[int(row["task_index"])] = str(row["task"])

    episodes_with_stats = {
        int(row["episode_index"])
        for row in _read_jsonl(
            _hf_download(repo_id, "meta/episodes_stats.jsonl", revision)
        )
    }

    return DatasetHandle(
        repo_id=repo_id,
        revision=resolved_revision,
        codebase_version=codebase_version,
        chunks_size=int(info["chunks_size"]),
        fps=int(info["fps"]),
        total_episodes=int(info["total_episodes"]),
        total_frames=int(info["total_frames"]),
        data_path=info["data_path"],
        video_path=info["video_path"],
        frame_shape=frame_shape,  # type: ignore[arg-type]
        episode_lengths=episode_lengths,
        episode_tasks=episode_tasks,
        episodes_with_stats=episodes_with_stats,
        tasks=tasks,
    )


@dataclass
class FrameRecord:
    """One dataset frame: the observation plus its ground-truth action."""

    episode_index: int
    frame_index: int
    task_index: int
    instruction: str
    state: np.ndarray            # (6,) float32
    ground_truth_action: np.ndarray  # (6,) float32
    videos: dict[str, np.ndarray]    # camera key -> (H, W, 3) uint8


def _decode_frames(path: Path, frame_indices: Sequence[int]) -> np.ndarray:
    """Decode the requested frame indices from an AV1 mp4 as NHWC uint8 RGB.

    Uses ``torchcodec`` (present today via the incumbent lerobot 0.3.3 pin) and
    falls back to ``av`` if torchcodec is unavailable.
    """
    try:
        from torchcodec.decoders import VideoDecoder

        decoder = VideoDecoder(str(path), dimension_order="NHWC")
        batch = decoder.get_frames_at(list(frame_indices))
        return np.ascontiguousarray(batch.data.numpy())
    except ImportError:  # pragma: no cover - fallback path
        import av

        wanted = sorted(set(int(i) for i in frame_indices))
        decoded: dict[int, np.ndarray] = {}
        with av.open(str(path)) as container:
            for position, frame in enumerate(container.decode(video=0)):
                if position in wanted:
                    decoded[position] = frame.to_ndarray(format="rgb24")
                if len(decoded) == len(wanted):
                    break
        return np.stack([decoded[int(i)] for i in frame_indices])


def read_episode_frames(
    handle: DatasetHandle,
    episode_index: int,
    frame_indices: Sequence[int],
    camera_keys: Sequence[str] = tuple(CAMERA_KEYS),
) -> list[FrameRecord]:
    """Read the given frames of one episode from raw parquet + AV1 video.

    NO ``lerobot`` import, so this survives the 0.6.1 bump and is immune to
    ``LeRobotDataset``'s v2.1 ``BackwardCompatibilityError``.
    """
    import pyarrow.parquet as pq

    if episode_index not in handle.episode_lengths:
        raise RuntimeError(
            f"episode {episode_index} is not listed in meta/episodes.jsonl "
            f"({handle.total_episodes} episodes available)"
        )
    if episode_index not in handle.episodes_with_stats:
        raise RuntimeError(
            f"episode {episode_index} has no meta/episodes_stats.jsonl entry — "
            f"refusing to capture a frame the dataset does not describe"
        )

    chunk = episode_index // handle.chunks_size
    parquet_path = _hf_download(
        handle.repo_id,
        handle.data_path.format(episode_chunk=chunk, episode_index=episode_index),
        handle.revision or None,
    )
    table = pq.read_table(
        parquet_path,
        columns=["observation.state", "action", "frame_index", "task_index"],
    )
    n_rows = table.num_rows
    for index in frame_indices:
        if not 0 <= index < n_rows:
            raise RuntimeError(
                f"frame_index {index} out of range for episode {episode_index} "
                f"({n_rows} frames)"
            )

    states = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)
    actions = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
    parquet_frame_index = np.asarray(
        table.column("frame_index").to_pylist(), dtype=np.int64
    )
    task_indices = np.asarray(table.column("task_index").to_pylist(), dtype=np.int64)

    videos: dict[str, np.ndarray] = {}
    for camera_key in camera_keys:
        video_path = _hf_download(
            handle.repo_id,
            handle.video_path.format(
                episode_chunk=chunk,
                video_key=VIDEO_KEY_TO_FEATURE[camera_key],
                episode_index=episode_index,
            ),
            handle.revision or None,
        )
        frames = _decode_frames(video_path, frame_indices)
        if frames.shape[1:] != handle.frame_shape or frames.dtype != np.uint8:
            raise RuntimeError(
                f"{camera_key} frames decoded as {frames.shape[1:]}/{frames.dtype}, "
                f"expected {handle.frame_shape}/uint8"
            )
        videos[camera_key] = frames

    records: list[FrameRecord] = []
    for position, frame_index in enumerate(frame_indices):
        # The parquet's own frame_index column must agree with the row offset;
        # if it does not, the row ordering assumption behind the video alignment
        # is wrong and the pairing would be silently bogus.
        if int(parquet_frame_index[frame_index]) != int(frame_index):
            raise RuntimeError(
                f"episode {episode_index} row {frame_index} carries "
                f"frame_index={int(parquet_frame_index[frame_index])} — parquet row "
                f"order does not match frame order; video/state pairing unsafe"
            )
        task_index = int(task_indices[frame_index])
        instruction = handle.tasks.get(
            task_index, handle.episode_tasks.get(episode_index, "")
        )
        records.append(
            FrameRecord(
                episode_index=episode_index,
                frame_index=int(frame_index),
                task_index=task_index,
                instruction=instruction,
                state=states[frame_index].astype(np.float32),
                ground_truth_action=actions[frame_index].astype(np.float32),
                videos={key: videos[key][position] for key in camera_keys},
            )
        )
    return records


# --- Observation assembly (mirrors controller.py:105-132 exactly) -------------


def _recursive_add_extra_dim(obs: dict) -> dict:
    """Byte-for-byte mirror of controller.py's ``_recursive_add_extra_dim``."""
    for key, val in obs.items():
        if isinstance(val, np.ndarray):
            obs[key] = val[np.newaxis, ...]
        elif isinstance(val, dict):
            obs[key] = _recursive_add_extra_dim(val)
        else:
            obs[key] = [val]
    return obs


def assemble_observation(
    observation_dict: dict[str, Any],
    instruction: str,
    camera_keys: Sequence[str] = tuple(CAMERA_KEYS),
    robot_state_keys: Sequence[str] = tuple(ROBOT_STATE_KEYS),
) -> dict[str, Any]:
    """Build the nested server observation from a flat observation dict.

    Reproduces ``Gr00tRobotInferenceClient.get_action``'s assembly
    (``controller.py:105-132``) exactly: ``video`` keyed wrist/front, ``state``
    split single_arm=state[:5] / gripper=state[5:6] as float32, the language dict
    under the PINNED ``annotation.human.task_description`` key (no ``.action.``),
    then the T=1 and B=1 dimension additions.

    A missing instruction RAISES rather than sending a null: an unconditioned
    capture would silently record garbage, and the live client raises here too.
    """
    if not instruction:
        raise ValueError(
            "instruction is required — the policy is language-conditioned and a "
            "null instruction would record an unconditioned action chunk"
        )
    state = np.array([observation_dict[k] for k in robot_state_keys])
    obs_dict: dict[str, Any] = {
        "video": {k: observation_dict[k] for k in camera_keys},
        "state": {
            "single_arm": state[:5].astype(np.float32),
            "gripper": state[5:6].astype(np.float32),
        },
        "language": {LANGUAGE_KEY: instruction},
    }
    # Add T=1 dim then B=1 dim (two successive calls, as the live path does).
    obs_dict = _recursive_add_extra_dim(obs_dict)
    obs_dict = _recursive_add_extra_dim(obs_dict)
    return obs_dict


def flat_observation(record: FrameRecord) -> dict[str, Any]:
    """Flat ``{cam_key: HxWx3 uint8, "<joint>.pos": float}`` dict for a frame.

    This is the same shape ``Gr00tRobotInferenceClient.get_action`` consumes, which
    is what lets tests/test_frozen_corpus.py drive both assemblies with one input.
    """
    obs: dict[str, Any] = {key: record.videos[key] for key in record.videos}
    for position, key in enumerate(ROBOT_STATE_KEYS):
        obs[key] = float(record.state[position])
    return obs


# --- Server interaction ------------------------------------------------------


def _chunk_to_array(action_chunk: dict[str, Any]) -> np.ndarray:
    """Concatenate the (B=1, T, D) modality arrays into a (T, 6) float32 array."""
    columns = [np.asarray(action_chunk[key], dtype=np.float32)[0] for key in MODALITY_KEYS]
    return np.concatenate(columns, axis=-1)


def capture_record(
    client: ExternalRobotInferenceClient,
    observation: dict[str, Any],
    options: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """One server round trip. Returns ``((T, 6) float32 chunk, info)``.

    Calls ``ExternalRobotInferenceClient.get_action`` directly so the RAW
    ``(action_chunk, info)`` reply and the ``options`` seed channel both survive.
    """
    action_chunk, info = client.get_action(observation, options=options)
    return _chunk_to_array(action_chunk), (info if isinstance(info, dict) else {})


def _max_abs_diff(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    return float(np.max(np.abs(left - right)))


def validate_seed_reproducibility(
    client: ExternalRobotInferenceClient,
    observation: dict[str, Any],
    seed: int,
    option_keys: Sequence[str] = SEED_OPTION_KEYS,
) -> tuple[str, dict[str, Any]]:
    """Settle D-07 empirically: does the server reproduce a chunk under one seed?

    For each candidate ``options`` key: two calls with the SAME seed, then a
    control call with a DIFFERENT seed. Same-seed equality alone is NOT enough —
    a server that ignores ``options`` but is deterministic would read as
    "honored", which is the false positive that would send Phase 7 down the wrong
    comparison path. ``honored`` therefore requires same-seed diff == 0 AND
    different-seed diff > 0.

    Returns ``(verdict, evidence)`` where verdict is exactly one of
    ``honored`` / ``not-honored`` / ``undetermined``.
    """
    per_key: dict[str, dict[str, Any]] = {}
    honored_key: str | None = None
    any_success = False
    any_reproduction_failure = False

    for option_key in option_keys:
        entry: dict[str, Any] = {
            "same_seed_max_abs_diff": None,
            "different_seed_max_abs_diff": None,
            "error": None,
        }
        try:
            first, _ = capture_record(client, observation, {option_key: int(seed)})
            second, _ = capture_record(client, observation, {option_key: int(seed)})
            control, _ = capture_record(
                client, observation, {option_key: int(seed) + CONTROL_SEED_OFFSET}
            )
        except Exception as exc:  # noqa: BLE001 - recorded as evidence, not raised
            entry["error"] = f"{type(exc).__name__}: {exc}"
            per_key[option_key] = entry
            continue

        same = _max_abs_diff(first, second)
        different = _max_abs_diff(first, control)
        entry["same_seed_max_abs_diff"] = same
        entry["different_seed_max_abs_diff"] = different
        per_key[option_key] = entry
        any_success = True

        if same == 0.0 and different > 0.0:
            honored_key = option_key
            break
        if same > 0.0:
            any_reproduction_failure = True

    if honored_key is not None:
        verdict = "honored"
        chosen = honored_key
    elif any_reproduction_failure:
        # The server replied but did not reproduce under any spelling.
        verdict = "not-honored"
        chosen = next(
            key
            for key, entry in per_key.items()
            if (entry["same_seed_max_abs_diff"] or 0.0) > 0.0
        )
    else:
        # Either every attempt errored, or the server is deterministic regardless
        # of the seed (same == 0 AND control == 0) — reproduction works but the
        # seed channel is unproven, so `honored` must not be claimed.
        verdict = "undetermined"
        chosen = next(iter(per_key), "")

    chosen_entry = per_key.get(chosen, {})
    evidence: dict[str, Any] = {
        "option_key_tried": list(per_key.keys()),
        "option_key_selected": chosen,
        "same_seed_max_abs_diff": chosen_entry.get("same_seed_max_abs_diff"),
        "different_seed_max_abs_diff": chosen_entry.get("different_seed_max_abs_diff"),
        "control_seed_offset": CONTROL_SEED_OFFSET,
        "seed": int(seed),
        "per_key": per_key,
        "any_server_reply": any_success,
        "method": (
            "two same-seed calls compared element-wise, plus a different-seed "
            "control; 'honored' requires same-seed diff == 0 AND control diff > 0"
        ),
    }
    return verdict, evidence


# --- Stratification ----------------------------------------------------------


def plan_stratified_frames(
    handle: DatasetHandle,
    records: int,
    episodes: Sequence[int] | None = None,
    episode_count: int | None = None,
    base_seed: int = 0,
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    """Choose (episode_index, frame_index) pairs along two declared axes.

    Axis 1 — ACROSS EPISODES: evenly spaced episode indices, never a single
    episode (only episode 0 was inspected during research; single-episode
    sampling would inherit its particular wrist pose).
    Axis 2 — ACROSS PHASES OF MOTION: frame indices at evenly spaced fractions of
    each episode's own length, so approach/grasp/lift are all represented rather
    than clustering at the start.

    Returns ``(pairs, recipe)``; the recipe goes into the manifest so the exact
    subset is reproducible.
    """
    if records < 1:
        raise ValueError(f"--records must be >= 1, got {records}")

    available = sorted(handle.episode_lengths)
    if not available:
        raise RuntimeError("dataset metadata lists no episodes")

    if episodes:
        chosen_episodes = [int(e) for e in episodes]
    else:
        if episode_count is None:
            # One episode per ~10 records, at least 1 and at most 20 — for the
            # 120-record default this yields 12 episodes (>= the 5-episode floor).
            episode_count = max(1, min(20, (records + 9) // 10))
        episode_count = min(episode_count, len(available), records)
        step = len(available) / episode_count
        chosen_episodes = [
            available[min(len(available) - 1, int(i * step))]
            for i in range(episode_count)
        ]

    per_episode = [records // len(chosen_episodes)] * len(chosen_episodes)
    for i in range(records % len(chosen_episodes)):
        per_episode[i] += 1

    pairs: list[tuple[int, int]] = []
    fractions_by_episode: dict[str, list[float]] = {}
    for episode_index, count in zip(chosen_episodes, per_episode):
        if count == 0:
            continue
        length = handle.episode_lengths[episode_index]
        if length < count:
            raise RuntimeError(
                f"episode {episode_index} has {length} frames, cannot draw {count} "
                f"distinct frames from it"
            )
        fractions = [(i + 0.5) / count for i in range(count)]
        fractions_by_episode[str(episode_index)] = fractions
        seen: set[int] = set()
        for fraction in fractions:
            frame_index = min(length - 1, int(fraction * length))
            while frame_index in seen:  # keep pairs unique
                frame_index = (frame_index + 1) % length
            seen.add(frame_index)
            pairs.append((episode_index, frame_index))

    recipe = {
        "axes": ["episode_index (evenly spaced)", "frame fraction of episode length"],
        "episodes": chosen_episodes,
        "records_per_episode": per_episode,
        "frame_fractions": fractions_by_episode,
        "base_seed": int(base_seed),
        "requested_records": int(records),
    }
    return pairs, recipe


def seeds_for(base_seed: int, ordinal: int, samples: int) -> list[int]:
    """Deterministic per-sample seeds so a re-run reproduces the same requests."""
    return [int(base_seed) + ordinal * 1000 + sample for sample in range(samples)]


# --- Corpus writing ----------------------------------------------------------


def record_filename(ordinal: int) -> str:
    return f"record_{ordinal:04d}.npz"


def write_record(
    out_dir: Path,
    ordinal: int,
    record: FrameRecord,
    action_samples: np.ndarray,
    seeds: Sequence[int],
) -> Path:
    """Write one ``record_NNNN.npz``. ``allow_pickle`` is left at its safe default."""
    path = out_dir / record_filename(ordinal)
    np.savez_compressed(
        path,
        video_wrist=record.videos["wrist"],
        video_front=record.videos["front"],
        state=record.state.astype(np.float32),
        ground_truth_action=record.ground_truth_action.astype(np.float32),
        action_samples=np.asarray(action_samples, dtype=np.float32),
        seeds=np.asarray(seeds, dtype=np.int64),
        episode_index=np.int64(record.episode_index),
        frame_index=np.int64(record.frame_index),
        task_index=np.int64(record.task_index),
        instruction=np.array(record.instruction),
    )
    return path


def manifest_entry(
    ordinal: int,
    record: FrameRecord,
    seeds: Sequence[int],
    server_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "file": record_filename(ordinal),
        "episode_index": record.episode_index,
        "frame_index": record.frame_index,
        "task_index": record.task_index,
        "instruction": record.instruction,
        "seeds": [int(s) for s in seeds],
    }
    if server_info:
        entry["server_info"] = server_info
    return entry


def write_manifest(out_dir: Path, manifest: dict[str, Any]) -> Path:
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _load_existing_record(path: Path) -> tuple[FrameRecord, np.ndarray] | None:
    """Reload a previously written record so an interrupted run can resume."""
    try:
        with np.load(path, allow_pickle=False) as data:
            record = FrameRecord(
                episode_index=int(data["episode_index"]),
                frame_index=int(data["frame_index"]),
                task_index=int(data["task_index"]),
                instruction=str(data["instruction"]),
                state=data["state"],
                ground_truth_action=data["ground_truth_action"],
                videos={
                    "wrist": data["video_wrist"],
                    "front": data["video_front"],
                },
            )
            return record, data["seeds"]
    except Exception:  # noqa: BLE001 - a corrupt/partial file is simply recaptured
        return None


# --- main --------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=None,
                        help="Pin an explicit dataset revision (default: main, SHA recorded).")
    parser.add_argument("--out", default="corpus/frozen_v1_0")
    parser.add_argument("--records", type=int, default=120)
    parser.add_argument("--samples-per-obs", type=int, default=5)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--timeout-ms", type=int, default=60000,
                        help="Per-request ZMQ timeout (real N1.7 inference is slow).")
    parser.add_argument("--instruction", default=None,
                        help="Override the language instruction (default: the "
                             "dataset's own task string for each frame).")
    parser.add_argument("--seed-base", type=int, default=20260907)
    parser.add_argument("--episodes", default=None,
                        help="Comma-separated explicit episode indices to sample.")
    parser.add_argument("--episode-count", type=int, default=None,
                        help="Number of episodes to spread records over.")
    parser.add_argument("--seed-check-only", action="store_true",
                        help="Run the D-07 seed check, write its verdict, capture nothing.")
    parser.add_argument("--server-label", default="gr00t:latest",
                        help="Provenance label for the producing server image.")
    parser.add_argument("--force", action="store_true",
                        help="Recapture records whose .npz already exists.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    out_dir = Path(args.out)
    episodes = (
        [int(e) for e in args.episodes.split(",") if e.strip()] if args.episodes else None
    )
    seed_check_only = bool(args.seed_check_only)
    total_checks = 4 if seed_check_only else 6
    checks = Checks(total_checks)

    print("=" * 72)
    print(f" Frozen v1.0 corpus capture -> {out_dir}")
    print(f" dataset {args.repo_id}   server {args.host}:{args.port} ({args.server_label})")
    print("=" * 72)

    # [1] Resolve the dataset (read-only) and pin the revision.
    checks.start("resolve dataset metadata (raw parquet + AV1, no lerobot import)")
    try:
        handle = resolve_dataset(args.repo_id, args.revision)
    except Exception as exc:  # noqa: BLE001
        checks.fail("dataset", f"{type(exc).__name__}: {exc}")
        return checks.report()
    checks.ok(
        "dataset",
        f"{handle.repo_id}@{handle.revision[:12]} codebase={handle.codebase_version} "
        f"episodes={handle.total_episodes} frames={handle.total_frames} "
        f"shape={handle.frame_shape}",
    )

    # [2] Plan the stratified subset and read the first frame (reader smoke).
    checks.start("stratify + read the first frame from parquet + AV1")
    try:
        pairs, recipe = plan_stratified_frames(
            handle,
            records=1 if seed_check_only else args.records,
            episodes=episodes,
            episode_count=args.episode_count,
            base_seed=args.seed_base,
        )
        first_pair = pairs[0]
        first_frames = read_episode_frames(handle, first_pair[0], [first_pair[1]])
    except Exception as exc:  # noqa: BLE001
        checks.fail("reader", f"{type(exc).__name__}: {exc}")
        return checks.report()
    first_record = first_frames[0]
    if args.instruction:
        first_record.instruction = args.instruction
    checks.ok(
        "reader",
        f"episode {first_record.episode_index} frame {first_record.frame_index}: "
        f"wrist {first_record.videos['wrist'].shape}/{first_record.videos['wrist'].dtype}, "
        f"state {first_record.state.shape}, "
        f"gt_action {first_record.ground_truth_action.shape}, "
        f"instruction={first_record.instruction!r}",
    )

    # [3] Server reachability BEFORE any capture attempt.
    checks.start(f"policy server reachability (ping {args.host}:{args.port})")
    client = ExternalRobotInferenceClient(
        host=args.host, port=args.port, timeout_ms=args.timeout_ms
    )
    if not client.ping():
        checks.fail(
            "server",
            f"ping returned False — the v1.0 policy server is not serving "
            f"{args.host}:{args.port}. Start it (see README) before capturing; "
            f"capturing against a broken producer is refused.",
        )
        return checks.report()
    checks.ok("server", f"{args.host}:{args.port} reachable ({args.server_label})")

    out_dir.mkdir(parents=True, exist_ok=True)

    # [4] D-07 seed verdict — reuse a recorded verdict, never recompute silently.
    checks.start("D-07 seed reproducibility verdict (double run + control)")
    manifest_path = out_dir / "manifest.json"
    existing: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    reused = (
        not seed_check_only
        and existing.get("seed_verdict")
        in ("honored", "not-honored", "undetermined")
    )
    if reused:
        seed_verdict = existing["seed_verdict"]
        seed_evidence = existing.get("seed_verdict_evidence", {})
        checks.ok("seed_verdict", f"reused recorded verdict {seed_verdict!r} from manifest")
    else:
        try:
            seed_observation = assemble_observation(
                flat_observation(first_record), first_record.instruction
            )
            seed_verdict, seed_evidence = validate_seed_reproducibility(
                client, seed_observation, args.seed_base
            )
        except Exception as exc:  # noqa: BLE001
            checks.fail("seed_verdict", f"{type(exc).__name__}: {exc}")
            return checks.report()
        seed_evidence["observation"] = {
            "episode_index": first_record.episode_index,
            "frame_index": first_record.frame_index,
            "instruction": first_record.instruction,
        }
        seed_evidence["server_label"] = args.server_label
        # A not-honored/undetermined verdict is a legitimate recorded outcome; it
        # decides which comparison Phase 7 USES, and never blocks capture.
        checks.ok(
            "seed_verdict",
            f"VERDICT {seed_verdict} "
            f"(same-seed max|diff|={seed_evidence.get('same_seed_max_abs_diff')}, "
            f"different-seed max|diff|={seed_evidence.get('different_seed_max_abs_diff')}, "
            f"keys tried={seed_evidence.get('option_key_tried')})",
        )

    # If NO candidate key ever produced a successful reply, the server rejects an
    # options payload outright (e.g. it splats options as kwargs). Sending one
    # during capture would then fail every request, so degrade to options=None and
    # record that the seeds are nominal only — a corpus must not claim a seed the
    # server never saw (T-05-09).
    seed_options_sent = bool((seed_evidence or {}).get("any_server_reply", True))
    seed_option_key = (
        ((seed_evidence or {}).get("option_key_selected") or SEED_OPTION_KEYS[0])
        if seed_options_sent
        else None
    )
    if not seed_options_sent:
        print(
            _red(
                "  NOTE: the server rejected every options payload — capturing with "
                "options=None; recorded seeds are nominal sample ordinals, not a "
                "channel the server observed."
            )
        )

    def seed_options(seed: int) -> dict[str, Any] | None:
        return {seed_option_key: int(seed)} if seed_option_key else None

    manifest: dict[str, Any] = {
        "corpus_schema_version": CORPUS_SCHEMA_VERSION,
        "dataset_repo_id": handle.repo_id,
        "dataset_revision": handle.revision,
        "dataset_codebase_version": handle.codebase_version,
        "dataset_fps": handle.fps,
        "instruction": args.instruction or first_record.instruction,
        "instruction_source": "cli" if args.instruction else "dataset-task",
        "samples_per_observation": int(args.samples_per_obs),
        "seed_verdict": seed_verdict,
        "seed_verdict_evidence": seed_evidence,
        "seed_option_key": seed_option_key,
        "seed_options_sent": seed_options_sent,
        "action_modality_layout": ACTION_MODALITY_LAYOUT,
        "stratification": recipe,
        "server_image": args.server_label,
        "server_endpoint": f"{args.host}:{args.port}",
        "captured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "capture_script": "scripts/capture_frozen_corpus.py",
    }

    if seed_check_only:
        manifest["record_count"] = 0
        manifest["records"] = []
        manifest["seed_check_only"] = True
        write_manifest(out_dir, manifest)
        print(f"\nVERDICT {seed_verdict}  ->  {out_dir / 'manifest.json'}")
        return checks.report()

    # [5] Bulk capture. Restartable: an existing .npz is reused, not recaptured.
    checks.start(
        f"capture {len(pairs)} records x {args.samples_per_obs} samples "
        f"across {len(recipe['episodes'])} episodes"
    )
    entries: list[dict[str, Any]] = []
    by_episode: dict[int, list[tuple[int, int]]] = {}
    for ordinal, (episode_index, frame_index) in enumerate(pairs):
        by_episode.setdefault(episode_index, []).append((ordinal, frame_index))

    started = time.time()
    captured = 0
    resumed = 0
    try:
        for episode_index, items in by_episode.items():
            pending = [
                (ordinal, frame_index)
                for ordinal, frame_index in items
                if args.force or not (out_dir / record_filename(ordinal)).exists()
            ]
            for ordinal, frame_index in items:
                path = out_dir / record_filename(ordinal)
                if (ordinal, frame_index) not in pending:
                    reloaded = _load_existing_record(path)
                    if reloaded is None:
                        pending.append((ordinal, frame_index))
                        continue
                    record, seeds = reloaded
                    entries.append(manifest_entry(ordinal, record, seeds.tolist()))
                    resumed += 1

            if not pending:
                continue
            frames = read_episode_frames(
                handle, episode_index, [frame_index for _, frame_index in pending]
            )
            for (ordinal, _frame_index), record in zip(pending, frames):
                if args.instruction:
                    record.instruction = args.instruction
                observation = assemble_observation(
                    flat_observation(record), record.instruction
                )
                seeds = seeds_for(args.seed_base, ordinal, args.samples_per_obs)
                samples = []
                info_seen: dict[str, Any] = {}
                for seed in seeds:
                    chunk, info = capture_record(
                        client, observation, seed_options(seed)
                    )
                    samples.append(chunk)
                    if info:
                        info_seen = {k: str(v) for k, v in info.items()}
                action_samples = np.stack(samples).astype(np.float32)
                write_record(out_dir, ordinal, record, action_samples, seeds)
                entries.append(
                    manifest_entry(ordinal, record, seeds, info_seen or None)
                )
                captured += 1
                if captured % 10 == 0 or captured == 1:
                    rate = (time.time() - started) / captured
                    print(
                        f"       {captured} captured "
                        f"(ep {record.episode_index} frame {record.frame_index}, "
                        f"{rate:.1f}s/record)"
                    )
    except Exception as exc:  # noqa: BLE001
        # Persist what we have so an interrupted long run resumes instead of
        # restarting: the records already on disk stay valid.
        entries.sort(key=lambda e: e["file"])
        manifest["record_count"] = len(entries)
        manifest["records"] = entries
        write_manifest(out_dir, manifest)
        checks.fail(
            "capture",
            f"{type(exc).__name__}: {exc} (after {captured} new records; "
            f"{len(entries)} on disk, manifest written — re-run to resume)",
        )
        return checks.report()

    entries.sort(key=lambda e: e["file"])
    checks.ok(
        "capture",
        f"{captured} newly captured, {resumed} reused, {len(entries)} total in "
        f"{time.time() - started:.0f}s",
    )

    # [6] Write the manifest and self-check the count/uniqueness invariants.
    checks.start("write manifest and self-check record invariants")
    manifest["record_count"] = len(entries)
    manifest["records"] = entries
    write_manifest(out_dir, manifest)
    unique_pairs = {(e["episode_index"], e["frame_index"]) for e in entries}
    distinct_episodes = {e["episode_index"] for e in entries}
    on_disk = len(sorted(out_dir.glob("record_*.npz")))
    problems = []
    if len(entries) != len(pairs):
        problems.append(f"expected {len(pairs)} records, have {len(entries)}")
    if len(unique_pairs) != len(entries):
        problems.append(
            f"duplicate (episode, frame) pairs: {len(entries) - len(unique_pairs)}"
        )
    if on_disk != len(entries):
        problems.append(f"{on_disk} .npz on disk vs record_count {len(entries)}")
    if problems:
        checks.fail("manifest", "; ".join(problems))
        return checks.report()
    checks.ok(
        "manifest",
        f"{len(entries)} records across {len(distinct_episodes)} episodes, "
        f"seed_verdict={seed_verdict} -> {out_dir / 'manifest.json'}",
    )
    return checks.report()


if __name__ == "__main__":
    sys.exit(main())
