#!/bin/bash

# Get repository name from git or environment
REPO_NAME="${GITHUB_REPOSITORY:-$(basename "$(git rev-parse --show-toplevel 2>/dev/null || echo "hcaptcha-challenger")")}"
IMAGE="hcaptcha-challenger"
REGISTRY="ghcr.io"
FULL_IMAGE="${REGISTRY}/${REPO_NAME,,}"  # Convert to lowercase for GHCR compatibility
TAG="${TAG:-v$(date +'%y%m%d%H')}"

# Determine context and dockerfile paths
if [[ -f "pyproject.toml" ]]; then
  CONTEXT_PATH="."
  DOCKERFILE_PATH="./docker/Dockerfile"
elif [[ -f "../pyproject.toml" ]]; then
  CONTEXT_PATH=".."
  DOCKERFILE_PATH="./docker/Dockerfile"
else
  CONTEXT_PATH="."
  DOCKERFILE_PATH="./docker/Dockerfile"
fi

echo "Building and pushing ${FULL_IMAGE}:${TAG}"
echo "Context: ${CONTEXT_PATH}, Dockerfile: ${DOCKERFILE_PATH}"

# Build with cache support
docker build \
  --cache-from "${FULL_IMAGE}:latest" \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  -t "${FULL_IMAGE}:${TAG}" \
  -t "${FULL_IMAGE}:latest" \
  -f "${DOCKERFILE_PATH}" "${CONTEXT_PATH}"

# Push images
echo "Pushing ${FULL_IMAGE}:${TAG}"
docker push "${FULL_IMAGE}:${TAG}"

echo "Pushing ${FULL_IMAGE}:latest" 
docker push "${FULL_IMAGE}:latest"

echo "Successfully published:"
echo "  ${FULL_IMAGE}:${TAG}"
echo "  ${FULL_IMAGE}:latest"
echo ""
echo "Pull with:"
echo "  docker pull ${FULL_IMAGE}:${TAG}"
echo "  docker pull ${FULL_IMAGE}:latest"