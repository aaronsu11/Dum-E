#!/usr/bin/env bash
# Complete source build by default. An immutable registry base is optional.
set -euo pipefail
cd "$(dirname "$0")/.."
base="${MODEL_SWAP_BASE_IMAGE:-}"
if [[ -n "$base" ]]; then
  [[ "$base" =~ @sha256:[a-f0-9]{64}$ ]] || { echo "Base override must be a registry digest" >&2; exit 1; }
else
  bash scripts/build_lerobot_policy_image.sh
  base=lerobot-policy
fi
# Read only the token in a child shell; never export the rest of .env to Docker.
if [[ -z "${HF_TOKEN:-}" && "${DUME_NO_DOTENV:-0}" != 1 && -f .env ]]; then
  HF_TOKEN="$(bash -c 'set -a; . "$1" >/dev/null 2>&1; printf "%s" "${HF_TOKEN:-}"' _ "$PWD/.env")"
fi
: "${HF_TOKEN:?HF_TOKEN with accepted Cosmos/PaliGemma terms is required}"
export HF_TOKEN
docker build --build-arg "BASE_IMAGE=$base" --secret id=hf_token,env=HF_TOKEN \
  --label "org.opencontainers.image.revision=$(git rev-parse HEAD)" \
  -f docker/lerobot/Dockerfile.multi_policy -t "${1:-dume-model-swap:local}" .
