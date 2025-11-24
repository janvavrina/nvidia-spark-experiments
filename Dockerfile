# Dockerfile for DGX Spark NLP Experiments
# Optimized for Grace Blackwell architecture (ARM64)
# Note: For ARM64, some packages may need to be built from source

FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# For ARM64, we may need to use a different base or build from source
# Uncomment if you need ARM64-specific base image:
# FROM --platform=linux/arm64 nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    libopenblas-dev \
    liblapack-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# Note: Some packages may need to be installed separately for ARM64 compatibility
RUN pip3 install --no-cache-dir \
    numpy \
    scipy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    tqdm \
    pyyaml \
    psutil \
    requests \
    aiohttp \
    Pillow \
    && pip3 install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    || pip3 install --no-cache-dir torch torchvision torchaudio

# Install ML/AI libraries
RUN pip3 install --no-cache-dir \
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
    GPUtil \
    nvidia-ml-py3 \
    pynvml

# Install inference engines (may need special handling for ARM64)
RUN pip3 install --no-cache-dir vllm || echo "vLLM installation failed, may need manual setup"
RUN pip3 install --no-cache-dir llama-cpp-python || echo "llama-cpp-python installation failed, may need manual setup"

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p results/benchmarks results/finetuning results/multimodal checkpoints mlruns

# Make scripts executable
RUN chmod +x setup/*.sh setup/*.py run_experiments.py reports/generate_report.py

# Set Python path
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Default command
CMD ["python", "run_experiments.py", "--config", "configs/experiments.yaml"]

