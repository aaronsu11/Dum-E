#!/usr/bin/env bash
set -euo pipefail
region="${AWS_REGION:-$(aws configure get region)}"
: "${region:?AWS region required}"
repo="${MODEL_SWAP_ECR_REPO:-dume/model-swap}"
image="${1:-dume-model-swap:phase8.1}"
tag="${2:?Usage: publish_model_swap_ecr.sh [local-image] immutable-tag}"
[[ "$tag" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$ ]] || { echo "Invalid tag" >&2; exit 1; }
account="$(aws sts get-caller-identity --query Account --output text)"
registry="$account.dkr.ecr.$region.amazonaws.com"
# A lookup failure is not assumed to be absence; create only on not-found.
err="$(mktemp)"
trap 'rm -f "$err"' EXIT
if ! aws ecr describe-repositories --region "$region" --repository-names "$repo" >/dev/null 2>"$err"; then
  if ! rg -q RepositoryNotFoundException "$err"; then cat "$err" >&2; exit 1; fi
  aws ecr create-repository --region "$region" --repository-name "$repo" \
    --image-tag-mutability IMMUTABLE --image-scanning-configuration scanOnPush=true >/dev/null
fi
aws ecr get-login-password --region "$region" |
  docker login --username AWS --password-stdin "$registry" >/dev/null
target="$registry/$repo:$tag"
docker tag "$image" "$target"
docker push "$target"
aws ecr describe-images --region "$region" --repository-name "$repo" \
  --image-ids "imageTag=$tag" --query 'imageDetails[0].{digest:imageDigest,size:imageSizeInBytes,tags:imageTags}' \
  --output json
echo "$target"
