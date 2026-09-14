# syntax=docker/dockerfile:1.7
# ==================== DELIBERATE DIVERGENCE: this repo vendors a Dockerfile ====================
# scripts/build_gr00t_image.sh's header states the opposite rule — "it never

ARG CUDA_BASE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04
FROM ${CUDA_BASE}

# lerobot 0.6.1 declares `Requires-Python: >=3.12`, and Ubuntu 24.04 ships
# Python 3.12 as its system python3 — that is why this base is 24.04 and not
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
# HF_TOKEN arrives as a BUILD SECRET and is readable only inside this one RUN
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
refs = snapshot.parent.parent / "refs"
refs.mkdir(parents=True, exist_ok=True)
(refs / "main").write_text(revision)
print(f"Pinned refs/main -> {revision} at {refs / 'main'}")
PY

# Fail the BUILD, not the first inference request, if the cache cannot actually
# satisfy an offline processor build.
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

COPY policy/ /app/policy/
COPY scripts/serve_observed_lerobot.py /app/scripts/serve_observed_lerobot.py
WORKDIR /app
ENV PYTHONPATH=/app
CMD ["python3", "/app/scripts/serve_observed_lerobot.py", "--host", "0.0.0.0", "--port", "8080"]
