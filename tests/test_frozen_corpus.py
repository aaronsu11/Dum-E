"""Keyless tests for the frozen v1.0 corpus capture + verification harness.

Every test here is CI-runnable with NO policy server, NO GPU, NO network and NO
SO101 hardware. The corpus fixtures are SYNTHETIC — correct shapes, fabricated
values — written through the capture script's own writer so the writer/verifier
loop is closed: whatever ``capture_frozen_corpus.write_record`` produces must be
what ``verify_frozen_corpus.verify_corpus`` accepts.

Surfaces covered:
- The verifier FAILS loudly (non-zero) on an absent corpus, a short corpus, a
  wrong-shape record, a missing seed verdict, and a mock-produced corpus — the
  "silent pass" failure mode it exists to prevent.
- The verifier ACCEPTS a well-formed synthetic corpus.
- ``assemble_observation`` is key-for-key and shape-for-shape identical to
  ``Gr00tRobotInferenceClient``'s live assembly, including the pinned
  ``annotation.human.task_description`` language key. This is the guard against
  the capture path and the live path silently drifting apart.
- The seed-reproducibility verdict classification, including the different-seed control that
  stops a deterministic-but-seed-ignoring server reading as ``honored``.
- The capture script imports no ``lerobot`` module (it must survive the 0.6.1
  bump that removes its readers).
"""

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Make scripts/ importable so we can reuse the capture/verify harnesses.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

import capture_frozen_corpus as capture  # noqa: E402
import verify_frozen_corpus as verify  # noqa: E402

INSTRUCTION = "Grab a banana and put it on the plate"
FRAME_SHAPE = (480, 640, 3)
ACTION_HORIZON = 16
SAMPLES = 5
MIN_RECORDS = 50


# --- Synthetic corpus fixture -----------------------------------------------


def _synthetic_frame_record(ordinal: int, episode_index: int, frame_index: int):
    """A FrameRecord with correct shapes/dtypes and fabricated values."""
    return capture.FrameRecord(
        episode_index=episode_index,
        frame_index=frame_index,
        task_index=0,
        instruction=INSTRUCTION,
        state=np.full((6,), float(ordinal), dtype=np.float32),
        ground_truth_action=np.full((6,), float(ordinal) + 0.5, dtype=np.float32),
        videos={
            "wrist": np.zeros(FRAME_SHAPE, dtype=np.uint8),
            "front": np.zeros(FRAME_SHAPE, dtype=np.uint8),
        },
    )


def _write_synthetic_corpus(
    out_dir: Path,
    record_count: int = MIN_RECORDS,
    episodes: int = 5,
    server_image: str = "gr00t:latest",
) -> Path:
    """Build a valid synthetic corpus through the capture script's own writer."""
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    per_episode = max(1, record_count // episodes)
    for ordinal in range(record_count):
        episode_index = min(episodes - 1, ordinal // per_episode)
        frame_index = ordinal % per_episode
        record = _synthetic_frame_record(ordinal, episode_index, frame_index)
        seeds = capture.seeds_for(1234, ordinal, SAMPLES)
        action_samples = np.arange(
            SAMPLES * ACTION_HORIZON * 6, dtype=np.float32
        ).reshape(SAMPLES, ACTION_HORIZON, 6)
        capture.write_record(out_dir, ordinal, record, action_samples, seeds)
        entries.append(capture.manifest_entry(ordinal, record, seeds))

    manifest = {
        "corpus_schema_version": capture.CORPUS_SCHEMA_VERSION,
        "dataset_repo_id": capture.DEFAULT_REPO_ID,
        "dataset_revision": "0" * 40,
        "dataset_codebase_version": capture.EXPECTED_CODEBASE_VERSION,
        "frame_shape": list(FRAME_SHAPE),
        "instruction": INSTRUCTION,
        "instruction_source": "dataset-task",
        "samples_per_observation": SAMPLES,
        "seed_verdict": "undetermined",
        "seed_verdict_evidence": {
            "option_key_tried": list(capture.SEED_OPTION_KEYS),
            "option_key_selected": "seed",
            "same_seed_max_abs_diff": 0.0,
            "different_seed_max_abs_diff": 0.0,
        },
        "seed_option_key": "seed",
        "seed_options_sent": True,
        "action_modality_layout": capture.ACTION_MODALITY_LAYOUT,
        "stratification": {
            "episodes": list(range(episodes)),
            "frame_fractions": {"0": [0.5]},
            "base_seed": 1234,
        },
        "server_image": server_image,
        "captured_at": "2026-09-07T00:00:00Z",
        "record_count": len(entries),
        "records": entries,
    }
    capture.write_manifest(out_dir, manifest)
    return out_dir


def _load_manifest(corpus_dir: Path) -> dict:
    return json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))


def _save_manifest(corpus_dir: Path, manifest: dict) -> None:
    (corpus_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


@pytest.fixture
def synthetic_corpus(tmp_path: Path) -> Path:
    return _write_synthetic_corpus(tmp_path / "frozen_v1_0")


# --- Verifier: the happy path ------------------------------------------------


def test_verifier_accepts_synthetic_valid_corpus(synthetic_corpus: Path):
    """A well-formed corpus verifies clean (exit 0)."""
    assert verify.verify_corpus(synthetic_corpus, min_records=MIN_RECORDS) == 0


# --- Verifier: it must FAIL, never skip quietly -------------------------------


def test_verifier_rejects_missing_corpus_directory(tmp_path: Path):
    """An absent corpus is a FAILURE, not a skip — this is the silent-pass guard."""
    assert verify.verify_corpus(tmp_path / "does_not_exist", min_records=MIN_RECORDS) != 0


def test_verifier_rejects_corpus_below_min_records(tmp_path: Path):
    """A corpus below the 50-record floor fails (roadmap criterion 5's lower bound)."""
    corpus = _write_synthetic_corpus(tmp_path / "short", record_count=10, episodes=5)
    assert verify.verify_corpus(corpus, min_records=MIN_RECORDS) != 0
    # ...and the same corpus is fine against a floor it actually meets, proving the
    # failure above is the floor and not some unrelated defect in the fixture.
    assert verify.verify_corpus(corpus, min_records=10) == 0


def test_verifier_rejects_record_with_wrong_shapes(synthetic_corpus: Path):
    """One record's state reshaped wrong must fail the whole corpus."""
    manifest = _load_manifest(synthetic_corpus)
    target = synthetic_corpus / manifest["records"][3]["file"]
    with np.load(target, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    arrays["state"] = np.zeros((7,), dtype=np.float32)  # was (6,)
    np.savez_compressed(target, **arrays)

    assert verify.verify_corpus(synthetic_corpus, min_records=MIN_RECORDS) != 0


def test_verifier_rejects_missing_seed_verdict(synthetic_corpus: Path):
    """A corpus with no seed verdict (or a bogus literal) fails."""
    manifest = _load_manifest(synthetic_corpus)
    manifest["seed_verdict"] = "probably-fine"
    _save_manifest(synthetic_corpus, manifest)
    assert verify.verify_corpus(synthetic_corpus, min_records=MIN_RECORDS) != 0

    del manifest["seed_verdict"]
    _save_manifest(synthetic_corpus, manifest)
    assert verify.verify_corpus(synthetic_corpus, min_records=MIN_RECORDS) != 0


def test_verifier_rejects_corpus_whose_record_count_disagrees_with_disk(
    synthetic_corpus: Path,
):
    """Deleting a record file without touching the manifest must fail (T-05-09)."""
    manifest = _load_manifest(synthetic_corpus)
    (synthetic_corpus / manifest["records"][0]["file"]).unlink()
    assert verify.verify_corpus(synthetic_corpus, min_records=MIN_RECORDS) != 0


def test_verifier_rejects_mock_produced_corpus(tmp_path: Path):
    """A corpus produced by the mock server is not v1.0 parity evidence (T-05-09)."""
    corpus = _write_synthetic_corpus(
        tmp_path / "mocked", server_image="mock_policy_server"
    )
    assert verify.verify_corpus(corpus, min_records=MIN_RECORDS) != 0
    # Explicitly opting in lets an operator inspect a smoke run.
    assert (
        verify.verify_corpus(corpus, min_records=MIN_RECORDS, require_real_producer=False)
        == 0
    )


# --- Assembly equivalence with the live path ---------------------------------


def _flat_observation() -> dict:
    """The flat ``{cam: HxWx3 uint8, "<joint>.pos": float}`` shape both paths consume."""
    rng = np.random.default_rng(7)
    obs: dict = {
        "wrist": rng.integers(0, 255, FRAME_SHAPE, dtype=np.uint8),
        "front": rng.integers(0, 255, FRAME_SHAPE, dtype=np.uint8),
    }
    for position, key in enumerate(capture.ROBOT_STATE_KEYS):
        obs[key] = float(position) - 2.5
    return obs


def _assert_same_structure(actual, expected, path: str = ""):
    """Recursively assert two observation structures match key/shape/dtype/value."""
    assert type(actual) is type(expected), f"{path}: {type(actual)} != {type(expected)}"
    if isinstance(expected, dict):
        assert set(actual) == set(expected), f"{path}: keys {set(actual)} != {set(expected)}"
        for key in expected:
            _assert_same_structure(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, np.ndarray):
        assert actual.shape == expected.shape, f"{path}: {actual.shape} != {expected.shape}"
        assert actual.dtype == expected.dtype, f"{path}: {actual.dtype} != {expected.dtype}"
        assert np.array_equal(actual, expected), f"{path}: values differ"
    elif isinstance(expected, list):
        assert len(actual) == len(expected), f"{path}: len {len(actual)} != {len(expected)}"
        for index, item in enumerate(expected):
            _assert_same_structure(actual[index], item, f"{path}[{index}]")
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def test_capture_observation_assembly_matches_controller(mocker):
    """The capture assembly and controller.py's live assembly must be identical.

    Builds a ``Gr00tRobotInferenceClient`` via ``__new__`` (no socket opened, per
    tests/test_gr00t_service.py's builder), captures the observation dict it hands
    to the policy, and compares it key-for-key / shape-for-shape against
    ``assemble_observation``'s output for the same flat input — including the
    pinned ``annotation.human.task_description`` key.
    """
    from embodiment.so_arm10x.controller import Gr00tRobotInferenceClient

    client = Gr00tRobotInferenceClient.__new__(Gr00tRobotInferenceClient)
    client.camera_keys = list(capture.CAMERA_KEYS)
    client.robot_state_keys = list(capture.ROBOT_STATE_KEYS)
    client.show_images = False
    # `language_instruction` is a read-only property; seed the backing field the
    # same way `__init__` does, since this client is built via `__new__`.
    client._language_instruction = None
    client.modality_keys = ["single_arm", "gripper"]

    seen: dict = {}

    def _fake_get_action(observation_dict):
        seen["observation"] = observation_dict
        return (
            {
                "single_arm": np.zeros((1, ACTION_HORIZON, 5), dtype=np.float32),
                "gripper": np.zeros((1, ACTION_HORIZON, 1), dtype=np.float32),
            },
            {},
        )

    client.policy = mocker.MagicMock()
    client.policy.get_action.side_effect = _fake_get_action

    client.get_action(_flat_observation(), lang=INSTRUCTION)
    mine = capture.assemble_observation(_flat_observation(), INSTRUCTION)

    _assert_same_structure(seen["observation"], mine)
    # Pin the language key explicitly: an equal-but-wrong key on both sides would
    # otherwise satisfy the structural comparison above.
    assert capture.LANGUAGE_KEY == "annotation.human.task_description"
    assert list(mine["language"].keys()) == [capture.LANGUAGE_KEY]
    assert mine["language"][capture.LANGUAGE_KEY] == [[INSTRUCTION]]


def test_assemble_observation_raises_without_instruction():
    """A null instruction must raise, not silently record an unconditioned chunk."""
    with pytest.raises(ValueError, match="instruction is required"):
        capture.assemble_observation(_flat_observation(), "")


# --- seed-reproducibility verdict classification (no server) -----------------


class _StubClient:
    """Minimal stand-in for ExternalRobotInferenceClient.get_action."""

    def __init__(self, mode: str):
        self.mode = mode
        self.calls = 0

    def get_action(self, observation, options=None):
        self.calls += 1
        if self.mode == "rejects-options":
            raise RuntimeError("Server error: unexpected keyword in options")
        if self.mode == "deterministic":
            value = 1.0
        elif self.mode == "honors-seed":
            value = float(next(iter(options.values()))) if options else 0.0
        elif self.mode == "nondeterministic":
            value = float(self.calls)
        else:  # pragma: no cover - guard
            raise AssertionError(self.mode)
        return (
            {
                "single_arm": np.full((1, ACTION_HORIZON, 5), value, dtype=np.float32),
                "gripper": np.full((1, ACTION_HORIZON, 1), value, dtype=np.float32),
            },
            {},
        )


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("honors-seed", "honored"),
        ("nondeterministic", "not-honored"),
        ("deterministic", "undetermined"),
        ("rejects-options", "undetermined"),
    ],
)
def test_seed_verdict_requires_different_seed_control(mode, expected):
    """A deterministic server that IGNORES the seed must not read as ``honored``.

    That false positive is what would send the parity gate down the wrong
    comparison path, so it is the single most important property of the check.
    """
    verdict, evidence = capture.validate_seed_reproducibility(
        _StubClient(mode), observation={}, seed=42
    )
    assert verdict == expected
    assert evidence["option_key_tried"]
    assert "same_seed_max_abs_diff" in evidence
    assert "different_seed_max_abs_diff" in evidence
    if mode == "rejects-options":
        assert evidence["any_server_reply"] is False
        assert all(entry["error"] for entry in evidence["per_key"].values())
    if mode == "deterministic":
        # The tell-tale pair: same-seed identical AND control identical.
        assert evidence["same_seed_max_abs_diff"] == 0.0
        assert evidence["different_seed_max_abs_diff"] == 0.0


# --- Version-agnosticism gate ------------------------------------------------


def test_capture_script_imports_no_lerobot():
    """The capture path must survive the 0.6.1 bump that removes its readers.

    ``lerobot`` 0.6.1's ``LeRobotDataset`` also hard-raises on this dataset's
    ``codebase_version: v2.1``, so the reader goes straight to parquet + AV1.
    """
    source = (REPO_ROOT / "scripts" / "capture_frozen_corpus.py").read_text()
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    offenders = [name for name in imported if name.split(".")[0] == "lerobot"]
    assert not offenders, f"capture script imports lerobot: {offenders}"
