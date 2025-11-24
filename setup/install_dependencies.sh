#!/bin/bash
# Environment setup script for NVIDIA DGX Spark (Grace Blackwell architecture)
# This script installs PyTorch, TensorFlow, JAX, and other dependencies

set -e

echo "Setting up environment for NVIDIA DGX Spark (Grace Blackwell)"

# Detect architecture
ARCH=$(uname -m)
echo "Detected architecture: $ARCH"

# Check if we're on ARM64 (Grace CPU)
if [ "$ARCH" != "aarch64" ]; then
    echo "Warning: Expected ARM64 (aarch64) architecture for Grace CPU, but detected $ARCH"
fi

# Update package lists
echo "Updating package lists..."
sudo apt-get update

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    libopenblas-dev \
    liblapack-dev \
    libffi-dev \
    libssl-dev

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch for ARM64 with CUDA support
echo "Installing PyTorch for ARM64 with CUDA..."
# Note: PyTorch for ARM64 may need to be built from source or use nightly builds
# Check https://pytorch.org/get-started/locally/ for ARM64 CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install TensorFlow
echo "Installing TensorFlow..."
# TensorFlow for ARM64 may require building from source or using specific builds
pip install tensorflow

# Install JAX
echo "Installing JAX..."
# JAX for ARM64 with CUDA support
pip install "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install ML/AI libraries
echo "Installing ML/AI libraries..."
pip install \
    transformers \
    datasets \
    accelerate \
    bitsandbytes \
    peft \
    trl \
    sentencepiece \
    protobuf \
    safetensors \
    huggingface-hub \
    wandb \
    mlflow \
    numpy \
    scipy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    tqdm \
    pyyaml \
    psutil \
    GPUtil \
    nvidia-ml-py3

# Install inference engines
echo "Installing inference engines..."

# vLLM
echo "Installing vLLM..."
pip install vllm

# llama.cpp Python bindings
echo "Installing llama-cpp-python..."
pip install llama-cpp-python

# Install additional utilities
echo "Installing additional utilities..."
pip install \
    requests \
    aiohttp \
    pynvml \
    py3nvml

echo "Environment setup complete!"
echo "To activate the virtual environment, run: source venv/bin/activate"

