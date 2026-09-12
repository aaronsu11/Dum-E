# syntax=docker/dockerfile:1
# Reuse the proven runtime/dependency image; checkpoints are separately pinned
# and cached at runtime. No HF/AWS credentials or model weights in COPY layers.
ARG BASE_IMAGE=lerobot-policy:phase7-stock-tests-20260911
FROM ${BASE_IMAGE}
ENV HF_HUB_OFFLINE=0 PYTHONUNBUFFERED=1 HF_HOME=/cache/huggingface
COPY --from=tokenizer / /opt/pi05-tokenizer/
COPY policy_lab/ /opt/model-swap/policy_lab/
COPY policy_guard/ /app/policy_guard/
COPY docker/lerobot-policy/server.py /app/docker/lerobot-policy/server.py
WORKDIR /opt/model-swap
RUN python3 -c "from lerobot.policies.pi05.modeling_pi05 import PI05Policy; from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy"
ENTRYPOINT ["python3", "-m", "policy_lab.server"]
