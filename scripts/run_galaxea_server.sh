#!/usr/bin/env bash
set -euo pipefail
image="${1:-dume-g05:phase9}"
checkpoint="${G05_CHECKPOINT_ROOT:-$HOME/.cache/huggingface/hub/models--OpenGalaxea--G05/snapshots/e312be81e90c56a55bcb26b57429bd39a335b449}"
evidence="${G05_EVIDENCE:-$PWD/corpus/phase9-g05-20260912/local-startup}"
# Check before Docker or checkpoint loading; X/desktop graphics alone are allowed.
gpu_users="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
[[ -z "$gpu_users" ]] || {
  echo "GPU occupied by compute PID(s): $gpu_users. Stop the current policy before switching." >&2
  exit 3
}
[[ -f "$checkpoint/g05-so101/checkpoints/model_state_dict.pt" ]] || {
  echo "Pinned G05 SO101 checkpoint is missing" >&2
  exit 2
}
mkdir -p "$evidence"
mounts=(-v "$checkpoint:/checkpoints:ro")
# HF snapshots contain relative symlinks to the repository's blobs directory.
if [[ -d "$checkpoint/../../blobs" ]]; then
  mounts+=(-v "$(realpath "$checkpoint/../../blobs"):/blobs:ro")
fi
exec docker run --rm --name dume-g05 --gpus all --network host \
  --memory 28g --memory-swap 30g --shm-size 1g \
  "${mounts[@]}" -v "$evidence:/evidence" \
  "$image" --evidence-dir /evidence
