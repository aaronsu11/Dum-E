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

# ==================== DOTENV FALLBACK ====================
# This repo's convention is that credentials live in the repo-root .env, not in
# the shell environment — tests/test_speech_live.py loads dotenv for exactly this
# reason, with the comment "the key normally lives in .env, not the shell
# environment". A build wrapper that refuses to build while the token sits in
# .env two lines away would be user-hostile, so load it here when it is absent.
#
# The token is never echoed, and it still reaches Docker only as a BuildKit
# secret. DUME_NO_DOTENV=1 disables this so the fail-closed check below stays
# testable on a machine that does have a .env.
if [ -z "${HF_TOKEN:-}" ] && [ "${DUME_NO_DOTENV:-0}" != "1" ] && [ -f "$_REPO_ROOT/.env" ]; then
  set -a
  # shellcheck source=/dev/null
  . "$_REPO_ROOT/.env"
  set +a
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

echo "Building $IMAGE_TAG"
echo "  lerobot pin        : $LEROBOT_PIN"
echo "  backbone model     : $BACKBONE_MODEL"
echo "  backbone revision  : $BACKBONE_REVISION"
echo "  dockerfile         : $DOCKERFILE"

# --secret id=hf_token,env=HF_TOKEN, never --build-arg: a --build-arg lands in
# the image history and `docker image inspect` would show it.
DOCKER_BUILDKIT=1 docker build \
  --secret id=hf_token,env=HF_TOKEN \
  --build-arg "LEROBOT_PIN=$LEROBOT_PIN" \
  --build-arg "BACKBONE_MODEL=$BACKBONE_MODEL" \
  --build-arg "BACKBONE_REVISION=$BACKBONE_REVISION" \
  -f "$DOCKERFILE" \
  -t "$IMAGE_TAG" \
  .

echo "Built image: $IMAGE_TAG"
# Plan 06-06 adds the post-build in-image snapshot assertion here (run the image
# and confirm the cached snapshot directory equals $BACKBONE_REVISION), plus the
# entrypoint runtime check that pairs with it.
