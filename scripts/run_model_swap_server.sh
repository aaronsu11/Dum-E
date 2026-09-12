#!/usr/bin/env bash
set -euo pipefail
profile="${1:?Usage: run_model_swap_server.sh pi05-base|molmoact2-so101|groot-so101 [image]}"
case "$profile" in pi05-base|molmoact2-so101|groot-so101) ;; *) echo "Unknown profile" >&2; exit 1;; esac
image="${2:-dume-model-swap:phase8.1}"
cache="${MODEL_SWAP_CACHE:-$HOME/.cache/huggingface}"
evidence="${MODEL_SWAP_EVIDENCE:-$PWD/corpus/model-swap-$profile}"
mkdir -p "$cache" "$evidence"
gpu_users="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
[[ -z "$gpu_users" ]] || { echo "GPU already has a compute process; stop the previous model before switching." >&2; exit 1; }
extra=()
if [[ "$profile" == groot-so101 ]]; then
  checkpoint="${MODEL_SWAP_GROOT_CHECKPOINT:-$PWD/checkpoints/GR00T-N1.7-3B-SO101}"
  [[ -f "$checkpoint/model.safetensors.index.json" ]] || { echo "Validated GR00T checkpoint required" >&2; exit 1; }
  # Reuse the image's pinned, pre-existing Cosmos cache entirely offline.
  extra=(-e HF_HOME=/root/.cache/huggingface -e HF_HUB_OFFLINE=1
         -v "$checkpoint:/checkpoints/model:ro")
fi
# host networking keeps the server itself loopback-only on Linux; no -p 0.0.0.0
# publication. On EC2 reach it only with ssh -L 8081:127.0.0.1:8081.
exec docker run --rm --name dume-model-swap --gpus all --network host \
  --memory 24g --memory-swap 26g --shm-size 1g \
  -v "$cache:/cache/huggingface" -v "$evidence:/evidence" \
  "${extra[@]}" \
  "$image" --profile "$profile" --evidence-dir /evidence
