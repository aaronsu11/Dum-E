#!/usr/bin/env python3
"""Replay and validate the frozen v1.0 corpus — exits non-zero, never skips quietly.

The corpus under ``corpus/frozen_v1_0/`` is the evidence the checkpoint parity gate is
measured against, and Phases 6-8 read it repeatedly. This harness is the
instrument that proves it is actually replayable: it walks every record on disk,
re-loads it, and asserts the schema, geometry and provenance the capture script
promised.

It is deliberately LOUD. A verifier that skips on a missing or short corpus is a
silent pass, which is the exact failure this instrument exists to prevent — so an
absent directory, a record count below the floor, a disk/manifest disagreement, a
missing array, a wrong dtype/shape, or a missing seed verdict all FAIL.

Every ``.npz`` is loaded with ``allow_pickle=False`` passed EXPLICITLY: a pickled
object array smuggled into a corpus file would otherwise execute code in every
later phase that reads it (the same choice ``policy/gr00t/service.py`` makes on
the wire).

Needs no policy server, no GPU, no hardware, and no ``lerobot`` import.

Usage:
    uv run python scripts/verify_frozen_corpus.py
    uv run python scripts/verify_frozen_corpus.py --corpus corpus/frozen_v1_0 --min-records 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Ensure the repo root is importable regardless of CWD (the scripts/ dir would
# otherwise shadow the repo root on sys.path[0]).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: The only corpus schema version this verifier understands.
EXPECTED_SCHEMA_VERSION = 1

#: The permitted seed-reproducibility verdict literals.
SEED_VERDICTS = ("honored", "not-honored", "undetermined")

#: Arrays every record must carry, with their expected dtype.
REQUIRED_ARRAYS: dict[str, Any] = {
    "video_wrist": np.uint8,
    "video_front": np.uint8,
    "state": np.float32,
    "ground_truth_action": np.float32,
    "action_samples": np.float32,
    "seeds": np.int64,
}

#: Substrings in ``server_image`` that mean the producer was NOT the v1.0 server.
#: A corpus that silently is not what it claims is not evidence (threat T-05-09).
MOCK_PRODUCER_MARKERS = ("mock", "smoke", "fake", "stub", "synthetic")

DEFAULT_FRAME_SHAPE = (480, 640, 3)


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


def verify_corpus(
    corpus_dir: Path,
    min_records: int = 50,
    require_real_producer: bool = True,
) -> int:
    """Verify the corpus at ``corpus_dir``. Returns 0 on success, non-zero on any failure."""
    checks = Checks(8)

    print("=" * 72)
    print(f" Frozen corpus verification -> {corpus_dir}")
    print("=" * 72)

    # [1] Presence. An absent corpus is a FAILURE, never a skip.
    checks.start("corpus directory and manifest.json exist")
    manifest_path = corpus_dir / "manifest.json"
    if not corpus_dir.is_dir():
        checks.fail("presence", f"corpus directory {corpus_dir} does not exist")
        return checks.report()
    if not manifest_path.is_file():
        checks.fail("presence", f"{manifest_path} does not exist")
        return checks.report()
    try:
        manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        checks.fail("presence", f"manifest.json is not valid JSON: {exc}")
        return checks.report()
    if not isinstance(manifest, dict):
        checks.fail("presence", f"manifest.json is a {type(manifest).__name__}, expected an object")
        return checks.report()
    checks.ok("presence", f"{manifest_path} present and parseable")

    # [2] Schema version.
    checks.start(f"corpus_schema_version == {EXPECTED_SCHEMA_VERSION}")
    version = manifest.get("corpus_schema_version")
    if version != EXPECTED_SCHEMA_VERSION:
        checks.fail(
            "schema_version",
            f"corpus_schema_version is {version!r}, expected {EXPECTED_SCHEMA_VERSION}",
        )
    else:
        checks.ok("schema_version", f"corpus_schema_version {version}")

    # [3] Record count floor.
    checks.start(f"record_count >= {min_records}")
    record_count = manifest.get("record_count")
    records = manifest.get("records")
    if not isinstance(record_count, int):
        checks.fail("record_count", f"record_count is {record_count!r}, expected an int")
    elif record_count < min_records:
        checks.fail(
            "record_count",
            f"record_count {record_count} is below the floor of {min_records}",
        )
    elif not isinstance(records, list) or len(records) != record_count:
        checks.fail(
            "record_count",
            f"records list has {len(records) if isinstance(records, list) else 'n/a'} "
            f"entries but record_count is {record_count}",
        )
    else:
        checks.ok("record_count", f"{record_count} records listed (floor {min_records})")

    entries = records if isinstance(records, list) else []

    # [4] Disk agreement.
    checks.start("the .npz files on disk agree with record_count")
    on_disk = sorted(p.name for p in corpus_dir.glob("record_*.npz"))
    listed = [e.get("file") for e in entries if isinstance(e, dict)]
    missing = [name for name in listed if name not in on_disk]
    orphans = [name for name in on_disk if name not in listed]
    if not isinstance(record_count, int) or len(on_disk) != record_count:
        checks.fail(
            "disk_agreement",
            f"{len(on_disk)} .npz on disk vs record_count {record_count!r}",
        )
    elif missing or orphans:
        checks.fail(
            "disk_agreement",
            f"manifest entries with no file: {missing[:5]}; files not in manifest: {orphans[:5]}",
        )
    else:
        checks.ok("disk_agreement", f"{len(on_disk)} .npz files match the manifest exactly")

    # [5] seed-reproducibility verdict.
    checks.start("seed_verdict is recorded with cited numbers")
    verdict = manifest.get("seed_verdict")
    evidence = manifest.get("seed_verdict_evidence")
    if verdict not in SEED_VERDICTS:
        checks.fail(
            "seed_verdict",
            f"seed_verdict is {verdict!r}, expected one of {SEED_VERDICTS}",
        )
    elif not isinstance(evidence, dict) or not evidence.get("option_key_tried"):
        checks.fail("seed_verdict", "seed_verdict_evidence is missing option_key_tried")
    elif (
        "same_seed_max_abs_diff" not in evidence
        or "different_seed_max_abs_diff" not in evidence
    ):
        checks.fail(
            "seed_verdict",
            "seed_verdict_evidence lacks same_seed_max_abs_diff or "
            "different_seed_max_abs_diff — a verdict with no cited number is not evidence",
        )
    else:
        checks.ok(
            "seed_verdict",
            f"{verdict} (same-seed {evidence.get('same_seed_max_abs_diff')}, "
            f"different-seed {evidence.get('different_seed_max_abs_diff')})",
        )

    # [6] Dataset + stratification provenance.
    checks.start("dataset provenance and stratification recipe are recorded")
    problems: list[str] = []
    if not manifest.get("dataset_repo_id"):
        problems.append("dataset_repo_id missing")
    if not manifest.get("dataset_revision"):
        problems.append("dataset_revision missing (a Hub change would be undetectable)")
    if manifest.get("dataset_codebase_version") != "v2.1":
        problems.append(
            f"dataset_codebase_version is {manifest.get('dataset_codebase_version')!r}, expected 'v2.1'"
        )
    if not manifest.get("instruction"):
        problems.append("instruction missing")
    samples_per_observation = manifest.get("samples_per_observation")
    if not isinstance(samples_per_observation, int) or samples_per_observation < 1:
        problems.append(f"samples_per_observation is {samples_per_observation!r}")
    recipe = manifest.get("stratification")
    if not isinstance(recipe, dict) or not recipe.get("episodes"):
        problems.append("stratification recipe missing its episode set")
    elif "base_seed" not in recipe or not recipe.get("frame_fractions"):
        problems.append("stratification recipe lacks base_seed or frame_fractions")
    if problems:
        checks.fail("provenance", "; ".join(problems))
    else:
        checks.ok(
            "provenance",
            f"{manifest['dataset_repo_id']}@{str(manifest['dataset_revision'])[:12]} "
            f"codebase={manifest['dataset_codebase_version']}, "
            f"{samples_per_observation} samples/observation, "
            f"{len(recipe['episodes'])} episodes in the recipe",
        )

    # [7] Per-record replay: every array, dtype, shape and provenance field.
    checks.start("replay every record (allow_pickle=False) and validate its arrays")
    frame_shape = tuple(manifest.get("frame_shape") or DEFAULT_FRAME_SHAPE)
    failures: list[str] = []
    seen_pairs: set[tuple[int, int]] = set()
    horizons: set[int] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            failures.append(f"records entry is a {type(entry).__name__}, expected an object")
            continue
        name = entry.get("file")
        if not name:
            failures.append("a records entry has no 'file'")
            continue
        path = corpus_dir / name
        if not path.is_file():
            failures.append(f"{name}: file missing")
            continue
        try:
            with np.load(path, allow_pickle=False) as data:
                present = set(data.files)
                for array_name, dtype in REQUIRED_ARRAYS.items():
                    if array_name not in present:
                        failures.append(f"{name}: missing array {array_name!r}")
                        continue
                    array = data[array_name]
                    if array.dtype != dtype:
                        failures.append(
                            f"{name}: {array_name} dtype {array.dtype}, expected {np.dtype(dtype)}"
                        )
                if "video_wrist" in present and data["video_wrist"].shape != frame_shape:
                    failures.append(
                        f"{name}: video_wrist shape {data['video_wrist'].shape}, expected {frame_shape}"
                    )
                if "video_front" in present and data["video_front"].shape != frame_shape:
                    failures.append(
                        f"{name}: video_front shape {data['video_front'].shape}, expected {frame_shape}"
                    )
                if "state" in present and data["state"].shape != (6,):
                    failures.append(f"{name}: state shape {data['state'].shape}, expected (6,)")
                if (
                    "ground_truth_action" in present
                    and data["ground_truth_action"].shape != (6,)
                ):
                    failures.append(
                        f"{name}: ground_truth_action shape "
                        f"{data['ground_truth_action'].shape}, expected (6,)"
                    )
                if "action_samples" in present:
                    samples = data["action_samples"]
                    if samples.ndim != 3 or samples.shape[-1] != 6:
                        failures.append(
                            f"{name}: action_samples shape {samples.shape}, expected (S, T, 6)"
                        )
                    else:
                        horizons.add(int(samples.shape[1]))
                        if (
                            isinstance(samples_per_observation, int)
                            and samples.shape[0] != samples_per_observation
                        ):
                            failures.append(
                                f"{name}: action_samples has {samples.shape[0]} samples, "
                                f"manifest says {samples_per_observation}"
                            )
                if "seeds" in present and isinstance(samples_per_observation, int):
                    if data["seeds"].shape != (samples_per_observation,):
                        failures.append(
                            f"{name}: seeds shape {data['seeds'].shape}, expected "
                            f"({samples_per_observation},)"
                        )
                    if [int(s) for s in data["seeds"]] != [
                        int(s) for s in entry.get("seeds", [])
                    ]:
                        failures.append(f"{name}: seeds disagree with the manifest entry")
                # Provenance: the record must be attributable to a specific frame.
                for field_name in ("episode_index", "frame_index"):
                    if field_name not in present:
                        failures.append(f"{name}: missing array {field_name!r}")
                    elif int(data[field_name]) != int(entry.get(field_name, -1)):
                        failures.append(
                            f"{name}: {field_name} {int(data[field_name])} disagrees with "
                            f"manifest {entry.get(field_name)!r}"
                        )
                if "episode_index" in present and "frame_index" in present:
                    pair = (int(data["episode_index"]), int(data["frame_index"]))
                    if pair in seen_pairs:
                        failures.append(
                            f"{name}: duplicate (episode, frame) pair {pair} — the same "
                            f"frame was captured more than once"
                        )
                    seen_pairs.add(pair)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {type(exc).__name__}: {exc}")

    if not entries:
        checks.fail("records", "no records to replay")
    elif failures:
        for failure in failures[:15]:
            print(_red(f"       {failure}"))
        checks.fail(
            "records",
            f"{len(failures)} record problem(s) across {len(entries)} records",
        )
    else:
        checks.ok(
            "records",
            f"{len(entries)} records replayed, {len(seen_pairs)} distinct "
            f"(episode, frame) pairs, action horizon(s) {sorted(horizons)}",
        )

    # [8] Producer provenance — a mock-produced corpus is not the v1.0 evidence.
    checks.start("server_image identifies the real v1.0 producer")
    server_image = str(manifest.get("server_image") or "")
    marker = next(
        (m for m in MOCK_PRODUCER_MARKERS if m in server_image.lower()), None
    )
    if not server_image:
        checks.fail("producer", "server_image missing — the corpus producer is unattributable")
    elif marker and require_real_producer:
        checks.fail(
            "producer",
            f"server_image {server_image!r} names a {marker!r} producer — this corpus "
            f"was NOT produced by the v1.0 groot-native server and must not be cited "
            f"as parity evidence (pass --allow-mock-producer to inspect it anyway)",
        )
    else:
        checks.ok("producer", f"server_image {server_image!r}")

    return checks.report()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--corpus", default="corpus/frozen_v1_0")
    parser.add_argument("--min-records", type=int, default=50)
    parser.add_argument(
        "--allow-mock-producer",
        action="store_true",
        help="Permit a corpus whose server_image names a mock/smoke producer "
             "(for inspecting a smoke run; never for parity evidence).",
    )
    args = parser.parse_args()
    return verify_corpus(
        Path(args.corpus),
        min_records=args.min_records,
        require_real_producer=not args.allow_mock_producer,
    )


if __name__ == "__main__":
    sys.exit(main())
