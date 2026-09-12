#!/usr/bin/env bash
# Repackage already-local pinned runtimes; no dependency/model downloads.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
recipe_dir="$(mktemp -d /tmp/dume-observed-images.XXXXXX)"
trap 'rm -rf "$recipe_dir"' EXIT
native_base=sha256:e263056fffe7a60a7f48b6309a8b8f2fb3ea9f8f2afa9c94a0105ed5b7d2eeaf
lerobot_base=sha256:6758186bd24cd0745b7442dafbb6680cbc2a986fe387eb2de5aaf0b75dce9c98
for backend in native lerobot; do
  if [[ "$backend" == native ]]; then base="$native_base"; else base="$lerobot_base"; fi
  docker image inspect "$base" >/dev/null
  docker tag "$base" "dume-observer-base-${backend}:local"
  if [[ "$backend" == native ]]; then
    cat > "$recipe_dir/$backend.Dockerfile" <<'DOCKER'
FROM dume-observer-base-native:local
COPY policy_guard/*.py /app/policy_guard/
COPY scripts/serve_observed_native.py scripts/replay_groot_native.py /app/scripts/
ENV PYTHONPATH=/app
LABEL dume.observer.default="lightweight"
ENTRYPOINT ["python"]
CMD ["/app/scripts/serve_observed_native.py", "--model-path", "/checkpoints/model", "--host", "0.0.0.0", "--port", "5555", "--observer-mode", "lightweight"]
DOCKER
  else
    cat > "$recipe_dir/$backend.Dockerfile" <<'DOCKER'
FROM dume-observer-base-lerobot:local
COPY policy_guard/*.py /app/policy_guard/
COPY scripts/serve_observed_lerobot.py /app/scripts/
ENV PYTHONPATH=/app
LABEL dume.observer.default="lightweight"
ENTRYPOINT ["python3"]
CMD ["/app/scripts/serve_observed_lerobot.py", "--host", "0.0.0.0", "--port", "8080"]
DOCKER
  fi
  docker build --network none --pull=false -f "$recipe_dir/$backend.Dockerfile" \
    -t "dume-${backend}-lightweight:20260912" "$project_root"
done
