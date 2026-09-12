#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
base=lerobot-policy:phase7-stock-tests-20260911
expected=sha256:6758186bd24cd0745b7442dafbb6680cbc2a986fe387eb2de5aaf0b75dce9c98
actual="$(docker image inspect "$base" --format '{{.Id}}')"
[[ "$actual" == "$expected" ]] || { echo "Pinned dependency image mismatch" >&2; exit 1; }
tokenizer="${MODEL_SWAP_TOKENIZER_SOURCE:-$HOME/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c}"
[[ -f "$tokenizer/tokenizer.json" ]] || { echo "Pinned Pi05 tokenizer snapshot required" >&2; exit 1; }
# Copy only tokenizer artifacts into a temporary build context. Snapshot entries
# are cache symlinks; dereference them without including HF credentials/cache.
tokenizer_context="$(mktemp -d)"
trap 'rm -rf "$tokenizer_context"' EXIT
for name in tokenizer.json tokenizer.model tokenizer_config.json special_tokens_map.json; do
  [[ ! -f "$tokenizer/$name" ]] || cp -L "$tokenizer/$name" "$tokenizer_context/$name"
done
docker build --build-arg "BASE_IMAGE=$base" \
  --build-context "tokenizer=$tokenizer_context" \
  --label "org.opencontainers.image.revision=$(git rev-parse HEAD)" \
  -f docker/model-swap/Dockerfile -t "${1:-dume-model-swap:phase8.1}" .
