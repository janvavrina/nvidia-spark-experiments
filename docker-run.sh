#!/bin/bash
# Convenience script to run experiments in Docker

set -e

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"
    exit 1
fi

# Check if nvidia-docker is available (for GPU support)
if ! docker info | grep -q nvidia; then
    echo "Warning: NVIDIA Docker runtime may not be configured"
    echo "Make sure you have nvidia-docker2 installed and configured"
fi

# Build the image if it doesn't exist or if --build flag is passed
if [ "$1" == "--build" ] || ! docker images | grep -q "dgx-spark-nlp-experiments"; then
    echo "Building Docker image..."
    docker compose build
fi

# Run the experiments
echo "Starting experiments in Docker container..."
docker compose run --rm dgx-spark-experiments "$@"

