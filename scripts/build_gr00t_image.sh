#!/usr/bin/env bash
#
# Pin-enforcing build wrapper for the Isaac-GR00T n1.7-release inference image.
#
# This is a thin delegator: it checks out the upstream clone at a pinned commit,
# fails closed if HEAD does not match the pin (supply-chain guard), then hands
# off to the *unmodified* upstream docker/build.sh. It never forks
# docker/build.sh, never vendors a Dockerfile, and never QEMU cross-builds Orin
# — run it natively on the Orin and the wrapper only passes --profile=orin
# through.
#
# Usage:
#   scripts/build_gr00t_image.sh                 # x86 default  -> image tag 'gr00t'
#   scripts/build_gr00t_image.sh orin            # Orin (native) -> image tag 'gr00t-orin'
#   scripts/build_gr00t_image.sh --profile=orin  # equivalent to the above
#
# Env:
#   GR00T_REPO   Path to the local Isaac-GR00T clone
#                (default: ../Isaac-GR00T relative to the repo root)
#
set -euo pipefail

# Default to a sibling Isaac-GR00T clone next to this repository; override with
# the GR00T_REPO env var for a clone elsewhere.
_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GR00T_REPO="${GR00T_REPO:-$(dirname "$_REPO_ROOT")/Isaac-GR00T}"
# == tag n1.7-release (verified == HEAD). Changing this changes WHICH commit
# builds — it is the reproducibility anchor and must not drift silently.
PIN="23ace64f17aa5015259b8609d371eb61a357c776"

# Normalize the optional first positional arg into the upstream profile flag.
#   ""                 -> x86 default (no flag; upstream image tag 'gr00t')
#   "orin"/"--profile=orin" -> "--profile=orin" (upstream image tag 'gr00t-orin')
PROFILE_ARG=""
case "${1:-}" in
  "")
    PROFILE_ARG=""
    ;;
  orin|--profile=orin)
    PROFILE_ARG="--profile=orin"
    ;;
  *)
    echo "ERROR: unknown argument '${1}' — expected nothing (x86) or 'orin'/'--profile=orin'." >&2
    exit 1
    ;;
esac

# Fail closed if the clone is missing rather than building stale/unknown code.
if [ ! -d "$GR00T_REPO/.git" ]; then
  echo "ERROR: Isaac-GR00T clone not found at GR00T_REPO=$GR00T_REPO — refusing to build." >&2
  exit 1
fi

# Pin enforcement (supply-chain fail-closed).
git -C "$GR00T_REPO" fetch --tags --quiet
git -C "$GR00T_REPO" checkout --quiet "$PIN"
HEAD="$(git -C "$GR00T_REPO" rev-parse HEAD)"
if [ "$HEAD" != "$PIN" ]; then
  echo "ERROR: Isaac-GR00T HEAD $HEAD != pinned $PIN — refusing to build." >&2
  exit 1
fi

# build.sh computes REPO_ROOT="$DIR/.." and uses the clone root as the build
# context, so we must invoke it from inside the clone (not from Dum-E).
cd "$GR00T_REPO"
if [ -n "$PROFILE_ARG" ]; then
  exec bash docker/build.sh "$PROFILE_ARG"
else
  exec bash docker/build.sh
fi
