#!/usr/bin/env bash
#
# Fail-closed build wrapper for the Dum-E lerobot-policy inference image.
#
# ==================== DELIBERATE DIVERGENCE from build_gr00t_image.sh ====================
# scripts/build_gr00t_image.sh is a thin DELEGATOR: it pins an upstream clone to
# a commit and hands off to the *unmodified* upstream docker/build.sh, and its
# header states it "never vendors a Dockerfile". This script says the OPPOSITE,
# and the reason is not a change of taste: LeRobot ships NO policy-server image
# build, so there is nothing to delegate to. On top of that the Cosmos backbone
# pre-cache needs a layer we own, because the downstream processor builder
# accepts no `revision` argument (processor_groot.py:1369-1381) — the image's HF
# cache contents plus HF_HUB_OFFLINE=1 are the only available pin enforcement.
#
# So the reproducibility anchor here is A PIP PIN PLUS AN HF REVISION SHA rather
# than a git SHA, and both are as fail-closed as the delegator's HEAD check. Do
# NOT "restore parity" by wrapping this in a fake delegator: there is no upstream
# script behind it.
#
# Usage:
#   bash scripts/build_lerobot_policy_image.sh          # -> image tag 'lerobot-policy'
#
# Env:
#   HF_TOKEN   Hugging Face token with read scope, required because the backbone
#              repo is gated. Passed to BuildKit as a --secret (never a
#              --build-arg, which would persist it into an image layer). If it is
#              not exported, this script loads it from the repo-root .env — see
#              DOTENV FALLBACK below. Set DUME_NO_DOTENV=1 to disable that.
#
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
#
# RECORDED, and deliberately not overstated: this SHA is the repo's CURRENT
# revision. It is NOT confirmed to be the revision the checkpoint was trained
# against, and NO local artifact records one — the checkpoint's config.json
# carries `model_name` only, never a revision. The pin's value is stopping silent
# upstream drift changing image preprocessing under a frozen checkpoint; it is a
# pin going forward, not a recovered historical pin. Do not write it up as
# though provenance were established.
BACKBONE_REVISION="9ce19a195e423419c349abfc86fd07178b230561"

IMAGE_TAG="lerobot-policy"

DOCKERFILE="docker/lerobot-policy/Dockerfile"

# == FREEZE THE ANCHOR BEFORE ANYTHING ELSE CAN TOUCH IT.
#
# `readonly` here is load-bearing, not tidiness. The dotenv fallback below used to
# run AFTER these five assignments and used `set -a` + `.`, which executes .env in
# THIS shell and exports everything it assigns — so a LEROBOT_PIN or
# BACKBONE_REVISION line in .env silently replaced the pin, unconditionally and
# with no message. The post-build assertion could not catch it either, because it
# compares the image against `$BACKBONE_REVISION`: the same overridden variable. A
# "reproducibility anchor" that reads its own overridden value is self-referential,
# and the image would have been tagged `lerobot-policy` while carrying a different
# backbone revision than this script declares.
#
# With `readonly`, an override attempt is a hard `set -e` failure naming the
# variable instead of a silent substitution. The extraction below no longer sources
# .env into this shell at all, so this is defence in depth rather than the only
# barrier — both are kept, because the failure being defended against is silent.
readonly LEROBOT_PIN BACKBONE_MODEL BACKBONE_REVISION IMAGE_TAG DOCKERFILE

# ==================== DOTENV FALLBACK ====================
# This repo's convention is that credentials live in the repo-root .env, not in
# the shell environment — tests/test_speech_live.py loads dotenv for exactly this
# reason, with the comment "the key normally lives in .env, not the shell
# environment". A build wrapper that refuses to build while the token sits in
# .env two lines away would be user-hostile, so load it here when it is absent.
#
# ONLY HF_TOKEN is extracted, and the sourcing happens in a SEPARATE `bash -c`
# PROCESS. That is the actual fix for the fail-open: nothing .env assigns can reach
# this shell at all, so the five pins above are untouchable through it, and the
# arbitrary shell .env executes cannot clobber any other exported variable of ours
# either. A `$( ... )` subshell would NOT do: it inherits the `readonly` attributes
# set above, so a `.env` naming a pin would make the assignment fail, take the
# subshell down under the inherited `set -e`, and abort this script with no message
# at all — fail-closed but undiagnosable, which is not the standard the rest of this
# file holds. `|| true` keeps a broken .env from aborting the build before the
# named, fail-closed HF_TOKEN check below can report it.
#
# The token is never echoed, and it still reaches Docker only as a BuildKit
# secret. DUME_NO_DOTENV=1 disables this so the fail-closed check below stays
# testable on a machine that does have a .env.
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
# to tag" is literally what happens on a mismatch rather than a message printed
# after the tag already moved — and the previously-good $IMAGE_TAG keeps pointing
# at the last image that DID carry the pinned snapshot.
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
# a git SHA to an HF revision SHA — the same fail-closed shape and the same
# `ERROR: ... — refusing to ...` + `exit 1` message shape, applied to the half of
# this script's reproducibility anchor that lives in an image layer rather than in
# a clone.
#
# Why it is worth a whole extra container start: without it, a STALE image is
# INDISTINGUISHABLE from a current one at inference time. The pre-cache layer's own
# assertions ran during the build that produced them, so a layer cached from an
# older BACKBONE_REVISION satisfies them and is then silently reused. And because
# _build_n1_7_processor accepts no `revision` argument
# (processor_groot.py:1369-1381), nothing downstream can notice: the image's HF
# cache contents plus HF_HUB_OFFLINE=1 are the ONLY available enforcement (D-08).
#
# It runs INSIDE the freshly built image rather than inspecting the host
# filesystem, because the host's HF cache is not what the container reads — the
# claim being checked is about this image.
#
# Scope, stated so it is not overread: this proves the image carries the revision
# THIS script pinned. It is DRIFT protection, not provenance — the SHA is the
# repo's current revision, not a recovered training-time revision (see the
# BACKBONE_REVISION comment above).
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
