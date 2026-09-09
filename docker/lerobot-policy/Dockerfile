# syntax=docker/dockerfile:1.7
#
# ==================== DELIBERATE DIVERGENCE: this repo vendors a Dockerfile ====================
#
# scripts/build_gr00t_image.sh's header states the opposite rule — "it never
# forks docker/build.sh, never vendors a Dockerfile" — and that rule was right
# THERE for a reason that does not hold HERE: Isaac-GR00T ships its own
# docker/build.sh, so there was an upstream build to delegate to and delegating
# was strictly better than reimplementing.
#
# LeRobot ships NO policy-server image build. There is nothing to delegate to.
# And the Cosmos backbone pre-cache needs a layer we own regardless, because
# _build_n1_7_processor accepts no `revision` argument
# (processor_groot.py:1369-1381), so a pinned revision can ONLY be enforced by
# what is in the image's HF cache plus HF_HUB_OFFLINE=1.
#
# The reproducibility anchor therefore shifts from a git SHA to A PIP PIN PLUS AN
# HF REVISION SHA, and both are fail-closed: the pip pin is exact, and the
# pre-cache layer asserts the snapshot directory it just downloaded actually
# exists. Do NOT "restore parity" by wrapping this in a fake delegator — there is
# no upstream script behind it, and a delegator that delegates to itself is a lie
# in the build graph.

ARG CUDA_BASE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04
FROM ${CUDA_BASE}

# lerobot 0.6.1 declares `Requires-Python: >=3.12`, and Ubuntu 24.04 ships
# Python 3.12 as its system python3 — that is why this base is 24.04 and not
# 22.04. (The Isaac-GR00T container is Py3.10; this one is Py3.12. Do not
# conflate them.) torch's own PyPI wheels bundle their CUDA libraries, so the
# base image only has to be driver-compatible, not a full CUDA devel image.
ARG LEROBOT_PIN
ARG BACKBONE_MODEL
ARG BACKBONE_REVISION

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONUNBUFFERED=1

# python3 + pip, git (some HF/torch paths still shell out to it), and the OpenCV
# runtime libraries the checkpoint's albumentations/cv2 image path needs.
# use_albumentations=True forces the cv2/CPU transform path
# (processor_groot.py:2076-2078), so these are load-bearing, not convenience.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        git \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# `groot` and `async` are DISJOINT extras: `groot` does NOT include grpcio
# (METADATA:182-189 lists only transformers-dep, peft-dep, diffusers-dep,
# dataset, dm-tree, timm-dep, decord), so `lerobot[groot]` alone would install
# cleanly and then fail at `import lerobot.async_inference`. BOTH are required.
RUN python3 -m pip install --no-cache-dir "lerobot[groot,async]==${LEROBOT_PIN}"

# Pre-cache the VLM backbone at the PINNED revision.
#
# HF_TOKEN arrives as a BUILD SECRET and is readable only inside this one RUN
# layer: nvidia/Cosmos-Reason2-2B reports `gated: "auto"`, so the download needs
# a token. NEVER `ARG HF_TOKEN` and NEVER `ENV HF_TOKEN` — either would persist
# the secret into an image layer / the image environment, which is exactly the
# leak `docker image inspect --format '{{json .Config.Env}}'` is asserted against.
#
# The assertion in the same RUN is what makes the layer fail-closed: a
# cache-empty layer and a correctly-cached layer are otherwise indistinguishable
# later, and under HF_HUB_OFFLINE=1 below a cache miss is FATAL at first
# inference — far from the build that caused it.
RUN --mount=type=secret,id=hf_token python3 <<'PY'
import os
import sys
from pathlib import Path

import huggingface_hub as h

# Read the secret in-process. It is never exported to a layer, an ARG or an ENV.
os.environ["HF_TOKEN"] = Path("/run/secrets/hf_token").read_text().strip()

model = os.environ["BACKBONE_MODEL"]
revision = os.environ["BACKBONE_REVISION"]

snapshot = Path(h.snapshot_download(model, revision=revision))
if not snapshot.is_dir() or revision not in str(snapshot):
    sys.exit(
        f"ERROR: backbone snapshot for {model} at revision {revision} was not cached "
        f"(resolved path: {snapshot}) - refusing to build."
    )
print(f"Cached {model} @ {revision} at {snapshot}")

# ==================== WHY refs/main IS WRITTEN BY HAND ====================
# snapshot_download(..., revision=<sha>) populates blobs/ and snapshots/<sha>/ but
# writes NO refs/ entry, because a SHA is not a moving reference. Every consumer
# downstream, however, asks for the repo by NAME with no revision -
# _build_n1_7_processor calls AutoTokenizer/Qwen2VLImageProcessor/
# Qwen3VLVideoProcessor .from_pretrained(model_name) (processor_groot.py:1371-1373)
# and cannot be given a revision (it takes no such argument). That resolves the
# revision "main", and resolving "main" under HF_HUB_OFFLINE=1 requires refs/main
# on disk. Without it the load fails with transformers' generic "couldn't connect
# ... and couldn't find them in the cached files" - while EVERY FILE IT NEEDS IS
# ALREADY THERE. The files were never the problem; the ref mapping was.
#
# So refs/main is written to point at the PINNED sha. That is deliberately
# stronger than pre-caching main would be: inside this image "main" now resolves
# to exactly BACKBONE_REVISION and can never drift to whatever upstream's main
# becomes later, which is the whole point of D-08's pin.
refs = snapshot.parent.parent / "refs"
refs.mkdir(parents=True, exist_ok=True)
(refs / "main").write_text(revision)
print(f"Pinned refs/main -> {revision} at {refs / 'main'}")
PY

# Fail the BUILD, not the first inference request, if the cache cannot actually
# satisfy an offline processor build.
#
# The previous assertion checked that the snapshot DIRECTORY existed, which it
# did - and the load still failed at first inference on the missing refs/main
# above. "The directory is there" and "the assets resolve offline" are different
# claims, and only the second one is what HF_HUB_OFFLINE=1 needs to be true. So
# this asserts the second, by calling the SAME private builder inference calls
# (processor_groot.py:2072) with offline forced on. It is CPU-only and loads no
# model weights: a tokenizer plus two image/video processors.
# tests/test_lerobot_upstream_surface.py pins the symbol.
RUN HF_HUB_OFFLINE=1 python3 <<'PY'
import sys

from lerobot.policies.groot.processor_groot import _build_n1_7_processor

try:
    processor = _build_n1_7_processor()
except Exception as exc:  # noqa: BLE001 - any failure here must fail the build
    sys.exit(
        "ERROR: the pinned backbone cache cannot satisfy an OFFLINE processor build "
        f"({type(exc).__name__}: {exc}) - refusing to build. Under HF_HUB_OFFLINE=1 this "
        "would otherwise surface as a swallowed 'Error in StreamActions' on the first "
        "inference request, hours after the build that caused it."
    )
print(f"Offline processor build OK: {type(processor).__name__}")
PY

# HF_HUB_OFFLINE=1: the pinned pre-cache above is the ONLY provenance mechanism
# available (no `revision` argument exists downstream), so the container must
# never silently reach the network and pull a different revision. A cache miss
# becomes a loud failure instead of a silent substitution.
ENV HF_HUB_OFFLINE=1
# Surfaced so the entrypoint can report (and, from plan 06-06, ASSERT) the pin at
# run time. A stale image is otherwise indistinguishable from a current one.
ENV DUME_BACKBONE_REVISION=${BACKBONE_REVISION}

COPY docker/lerobot-policy/*.py /app/docker/lerobot-policy/
WORKDIR /app

CMD ["python3", "/app/docker/lerobot-policy/entrypoint.py", "--host", "0.0.0.0", "--port", "8080"]
