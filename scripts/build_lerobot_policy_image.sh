#!/usr/bin/env bash
# Fail-closed build wrapper for the Dum-E lerobot-policy inference image.
# ==================== DELIBERATE DIVERGENCE from build_gr00t_image.sh ====================
set -euo pipefail

_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$_REPO_ROOT"

# == lerobot's exact PyPI pin. Changing this changes WHICH policy stack serves
# the checkpoint — it is half the reproducibility anchor and must not drift
# silently. It matches pyproject.toml's client-side pin on purpose: the wire
# payloads are pickled dataclasses, so client and server must be the same version.
LEROBOT_PIN="0.6.1"

# == the VLM backbone the N1.7 checkpoint's processor loads at first inference.
BACKBONE_MODEL="nvidia/Cosmos-Reason2-2B"

# == the OTHER half of the reproducibility anchor.
# RECORDED, and deliberately not overstated: this SHA is the repo's CURRENT
BACKBONE_REVISION="9ce19a195e423419c349abfc86fd07178b230561"

IMAGE_TAG="lerobot-policy"

DOCKERFILE="docker/lerobot-policy/Dockerfile"

# == FREEZE THE ANCHOR BEFORE ANYTHING ELSE CAN TOUCH IT.
# `readonly` here is load-bearing, not tidiness. The dotenv fallback below used to
readonly LEROBOT_PIN BACKBONE_MODEL BACKBONE_REVISION IMAGE_TAG DOCKERFILE

# ==================== DOTENV FALLBACK ====================
# This repo's convention is that credentials live in the repo-root .env, not in
if [ -z "${HF_TOKEN:-}" ] && [ "${DUME_NO_DOTENV:-0}" != "1" ] && [ -f "$_REPO_ROOT/.env" ]; then
  # Report — by NAME only, never by value — any pin the .env names. Without this the
  # correct new behaviour ("the .env entry is ignored") is as silent as the old wrong
  # behaviour ("the .env entry wins"), and an operator who put a revision in .env
  # expecting it to take effect would have no way to tell which happened.
  _shadowed="$(
    grep -oE '^[[:space:]]*(export[[:space:]]+)?(LEROBOT_PIN|BACKBONE_MODEL|BACKBONE_REVISION|IMAGE_TAG|DOCKERFILE)=' \
      "$_REPO_ROOT/.env" 2>/dev/null |
      grep -oE '(LEROBOT_PIN|BACKBONE_MODEL|BACKBONE_REVISION|IMAGE_TAG|DOCKERFILE)' |
      sort -u | tr '\n' ' '
  )" || true
  if [ -n "${_shadowed:-}" ]; then
    echo "WARNING: .env names build pin(s): ${_shadowed}" >&2
    echo "         They are IGNORED. These five values are this script's reproducibility" >&2
    echo "         anchor and are readonly; only HF_TOKEN is read from .env. Edit this" >&2
    echo "         script to change a pin." >&2
  fi
  # shellcheck disable=SC2016
  HF_TOKEN="$(
    bash -c 'set -a; . "$1" >/dev/null 2>&1; printf "%s" "${HF_TOKEN:-}"' _ "$_REPO_ROOT/.env" \
      2>/dev/null || true
  )"
  export HF_TOKEN
fi

# Fail closed on a missing token rather than building a cache-empty layer that
# only explodes at first inference, hours later and far from the cause.
if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN is not set — refusing to build." >&2
  echo "       ${BACKBONE_MODEL} is a GATED repo (gated: \"auto\"), so the backbone" >&2
  echo "       pre-cache cannot run without a token. Accept the terms once at" >&2
  echo "       https://huggingface.co/${BACKBONE_MODEL} then create a read-scope token at" >&2
  echo "       https://huggingface.co/settings/tokens and put HF_TOKEN in .env (or export it)." >&2
  exit 1
fi

# Fail closed on a missing Dockerfile rather than letting docker build report a
# less specific error about the build context.
if [ ! -f "$DOCKERFILE" ]; then
  echo "ERROR: $DOCKERFILE not found at repo root $_REPO_ROOT — refusing to build." >&2
  exit 1
fi

# == the tag the build writes FIRST, before the post-build assertion below has
# passed. The operator-facing $IMAGE_TAG is applied only afterwards, so "refusing
STAGING_TAG="${IMAGE_TAG}:unverified"

echo "Building $IMAGE_TAG"
echo "  lerobot pin        : $LEROBOT_PIN"
echo "  backbone model     : $BACKBONE_MODEL"
echo "  backbone revision  : $BACKBONE_REVISION"
echo "  dockerfile         : $DOCKERFILE"
echo "  staging tag        : $STAGING_TAG"

# --secret id=hf_token,env=HF_TOKEN, never --build-arg: a --build-arg lands in
# the image history and `docker image inspect` would show it.
DOCKER_BUILDKIT=1 docker build \
  --secret id=hf_token,env=HF_TOKEN \
  --build-arg "LEROBOT_PIN=$LEROBOT_PIN" \
  --build-arg "BACKBONE_MODEL=$BACKBONE_MODEL" \
  --build-arg "BACKBONE_REVISION=$BACKBONE_REVISION" \
  -f "$DOCKERFILE" \
  -t "$STAGING_TAG" \
  .

echo "Built staging image: $STAGING_TAG"

# ==================== POST-BUILD IN-IMAGE SNAPSHOT ASSERTION ====================
# This is the analogue of build_gr00t_image.sh:54-61's HEAD check, transposed from
echo "Asserting the pinned backbone snapshot is inside $STAGING_TAG ..."
if ! docker run --rm \
  -e "EXPECT_BACKBONE_REVISION=$BACKBONE_REVISION" \
  --entrypoint python3 "$STAGING_TAG" -c '
import os
import sys
from pathlib import Path

from huggingface_hub.constants import HF_HUB_CACHE

revision = os.environ["EXPECT_BACKBONE_REVISION"]
snapshots = Path(HF_HUB_CACHE) / "models--nvidia--Cosmos-Reason2-2B" / "snapshots"
found = sorted(p.name for p in snapshots.iterdir() if p.is_dir()) if snapshots.is_dir() else []
target = snapshots / revision
if not revision or not target.is_dir() or not any(target.iterdir()):
    sys.exit(
        f"asserted revision {revision!r} is not a non-empty snapshot directory under "
        f"{snapshots}; revisions found in the image: {found}"
    )
print(f"In-image backbone snapshot OK: {target}")
'; then
  echo "ERROR: backbone snapshot $BACKBONE_REVISION not found in image $IMAGE_TAG — refusing to tag." >&2
  echo "       The staging image is left as $STAGING_TAG for inspection, and $IMAGE_TAG still" >&2
  echo "       points at the last image that DID carry the pinned snapshot." >&2
  exit 1
fi

docker tag "$STAGING_TAG" "$IMAGE_TAG"
# Untag the staging name only; the image itself now lives under $IMAGE_TAG.
docker rmi "$STAGING_TAG" >/dev/null
echo "Built image: $IMAGE_TAG"
