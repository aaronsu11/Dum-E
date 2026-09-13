# syntax=docker/dockerfile:1.7
# Build the GR00T dependency base with scripts/build_lerobot_policy_image.sh,
# or pass an immutable registry image digest through the wrapper.
ARG BASE_IMAGE=lerobot-policy
FROM ${BASE_IMAGE}
RUN python3 -m pip install --no-cache-dir 'lerobot[pi,molmoact2]==0.6.1'
# Fetch only tokenizer assets at a fixed revision; the token never enters a layer.
RUN --mount=type=secret,id=hf_token HF_HUB_OFFLINE=0 python3 <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download
secret = Path('/run/secrets/hf_token')
snapshot_download('google/paligemma-3b-pt-224',
    revision='35e4f46485b4d07967e7e9935bc3786aad50687c',
    allow_patterns=['tokenizer.json', 'tokenizer.model', 'tokenizer_config.json', 'special_tokens_map.json'],
    local_dir='/opt/pi05-tokenizer', token=secret.read_text().strip() if secret.exists() else None)
assert Path('/opt/pi05-tokenizer/tokenizer.json').is_file()
PY
ENV HF_HUB_OFFLINE=0 PYTHONUNBUFFERED=1 HF_HOME=/cache/huggingface
COPY policy_lab/ /opt/model-swap/policy_lab/
COPY policy_guard/ /app/policy_guard/
COPY docker/lerobot-policy/server.py /app/docker/lerobot-policy/server.py
WORKDIR /opt/model-swap
RUN python3 -c "from lerobot.policies.pi05.modeling_pi05 import PI05Policy; from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy; from policy_lab.runtime import ModelRuntime"
ENTRYPOINT ["python3", "-m", "policy_lab.server"]
