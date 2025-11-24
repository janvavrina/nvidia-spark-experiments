# DGX Spark NLP Experiments

Comprehensive experimental framework for testing NVIDIA DGX Spark capabilities across multiple NLP and AI research areas.

## Overview

This repository contains a structured framework for benchmarking and experimenting with large language models, inference engines, training workflows, and multimodal AI on the NVIDIA DGX Spark system with Grace Blackwell architecture.

## Features

- **LLM Inference Benchmarks**: Test inference performance for models sized for 128GB unified memory
- **Inference Engine Comparison**: Compare vLLM, llama.cpp, Lemonade, and Parallax
- **Training & Fine-tuning**: LoRA/QLoRA fine-tuning workflows with performance tracking
- **Financial Domain Fine-tuning**: Fine-tune Granite 4.0, Gemma 3, and Qwen 3 on Financial PhraseBank
- **Framework Comparison**: Compare PyTorch, TensorFlow, and JAX performance
- **Multimodal AI**: Vision-language models and video processing experiments
- **Experiment Orchestration**: Automated sequential execution with dependency resolution
- **Performance Profiling**: NVLink-C2C and unified memory monitoring

## System Requirements

- NVIDIA DGX Spark with Grace Blackwell architecture
- 128GB unified memory
- CUDA 12.1+ support
- Python 3.8+
- ARM64 (aarch64) architecture

## Quick Start

### Option 1: Docker (Recommended)

Docker is recommended to avoid Python version conflicts and ensure consistent environment.

```bash
# Build and run experiments in Docker
./docker-run.sh --build

# Or use docker-compose directly
docker-compose build
docker-compose run --rm dgx-spark-experiments

# Run specific experiment
docker-compose run --rm dgx-spark-experiments python benchmarks/inference.py --model "meta-llama/Llama-3.3-8B-Instruct" --size "8B"

# Access container shell for debugging
docker-compose run --rm dgx-spark-experiments /bin/bash
```

**Docker Requirements:**
- Docker with NVIDIA Container Toolkit
- CUDA 12.1+ compatible drivers
- At least 50GB free disk space for images and models

### Option 2: Native Installation

### 1. Environment Setup

```bash
# Install dependencies
./setup/install_dependencies.sh

# Activate virtual environment
source venv/bin/activate

# Verify environment
python setup/check_environment.py
```

### 2. Run All Experiments

```bash
# Run all experiments sequentially
python run_experiments.py --config configs/experiments.yaml

# Resume from checkpoint if interrupted
python run_experiments.py --config configs/experiments.yaml

# Start fresh (ignore checkpoints)
python run_experiments.py --config configs/experiments.yaml --no-resume
```

### 3. Run Individual Experiments

```bash
# Inference benchmark
python benchmarks/inference.py \
    --model "meta-llama/Llama-3.3-8B-Instruct" \
    --size "8B" \
    --quantization "fp16" \
    --max-tokens 100 \
    --num-runs 5

# Inference engine comparison
python benchmarks/inference_engines.py \
    --model "Qwen/Qwen3-7B-Instruct" \
    --size "7B" \
    --max-tokens 100 \
    --num-runs 5

# Training benchmark
python benchmarks/training.py \
    --model "google/gemma-3-4b-pt" \
    --size "4B" \
    --method "lora" \
    --batch-size 4 \
    --epochs 1 \
    --max-steps 10

# Financial fine-tuning
python experiments/financial_finetuning/finetune_financial.py \
    --model "granite_4" \
    --use-lora \
    --epochs 3 \
    --batch-size 8

# Framework comparison
python benchmarks/framework_comparison.py \
    --model "meta-llama/Llama-3.3-8B-Instruct" \
    --size "8B" \
    --inference-runs 10 \
    --training-steps 10

# Multimodal benchmark
python experiments/multimodal/vision_language.py \
    --model "qwen3_vl" \
    --num-runs 10

# Or test other multimodal models:
# --model "deepseek_vl2"  # DeepSeek-VL2
# --model "gemma3"        # Gemma 3 (multimodal)
```

## Project Structure

```
.
├── setup/                      # Environment setup scripts
│   ├── install_dependencies.sh
│   └── check_environment.py
├── benchmarks/                # Benchmarking suites
│   ├── inference.py           # LLM inference benchmarks
│   ├── inference_engines.py   # Engine comparison (vLLM, llama.cpp, etc.)
│   ├── training.py            # Training performance benchmarks
│   └── framework_comparison.py # PyTorch vs TensorFlow vs JAX
├── experiments/               # Specific experiments
│   ├── financial_finetuning/  # Financial domain fine-tuning
│   │   └── finetune_financial.py
│   └── multimodal/           # Multimodal AI experiments
│       └── vision_language.py
├── orchestrator/             # Experiment orchestration
│   ├── __init__.py
│   └── experiment_runner.py
├── utils/                    # Utility modules
│   ├── profiling.py          # Performance profiling
│   └── experiment_tracking.py # MLflow/W&B integration
├── configs/                  # Configuration files
│   └── experiments.yaml      # Experiment definitions
├── results/                   # Results directory (created at runtime)
│   ├── benchmarks/
│   ├── finetuning/
│   └── multimodal/
├── checkpoints/              # Experiment checkpoints (created at runtime)
├── run_experiments.py        # Main orchestration script
└── README.md
```

## Model Selection for 128GB Unified Memory

Given the 128GB unified memory constraint, models are selected as follows:

- **Full Precision (FP16/BF16)**: 3B-30B models
  - Llama 3.3 8B / Llama 4
  - Qwen 3 7B/14B/32B
  - Gemma 3 4B/13B/27B
  - Granite 4.0 8B/27B
  - DeepSeek R1 7B
  - GPT-OSS 20B

- **Quantized (INT8)**: 30B-70B models
  - Llama 3.3 70B INT8
  - Qwen 3 72B INT8
  - Gemma 3 27B INT8

- **Heavily Quantized (INT4)**: Up to 70B models
  - Llama 3.3 70B INT4
  - Qwen 3 72B INT4

## Experiment Configuration

Edit `configs/experiments.yaml` to:
- Enable/disable specific experiments
- Set timeouts
- Configure dependencies
- Adjust experiment parameters

## Results and Reporting

Results are saved in JSON format in the `results/` directory:
- `results/benchmarks/` - Benchmark results
- `results/finetuning/` - Fine-tuning results
- `results/multimodal/` - Multimodal experiment results

Checkpoints are saved in `checkpoints/` to allow resuming interrupted experiments.

## Performance Profiling

The framework includes utilities for monitoring:
- Unified memory usage
- NVLink-C2C bandwidth
- GPU/CPU memory utilization
- Training and inference performance metrics

## Experiment Tracking

Integrates with:
- **MLflow**: Local tracking (default)
- **Weights & Biases**: Cloud-based tracking (optional)

Set `WANDB_API_KEY` environment variable to enable W&B tracking.

## Docker Setup Details

The Docker setup includes:
- **Base Image**: NVIDIA CUDA 12.1 with Ubuntu 22.04
- **Python**: 3.10 (explicit version to avoid conflicts)
- **All Dependencies**: Pre-installed from requirements.txt
- **Volume Mounts**: 
  - `results/` - Persists all benchmark results
  - `checkpoints/` - Saves experiment checkpoints
  - `~/.cache/huggingface/` - Reuses downloaded models
- **GPU Support**: Uses NVIDIA Container Toolkit

**For ARM64 (Grace Blackwell):**
```bash
# Use the ARM64-optimized Dockerfile
docker build -f Dockerfile.arm64 -t dgx-spark-nlp-experiments:arm64 .
docker run --gpus all --platform linux/arm64 \
    -v $(pwd)/results:/workspace/results \
    dgx-spark-nlp-experiments:arm64
```

### Building Custom Docker Image

If you need to customize the environment:

```bash
# Build with custom arguments
docker build -t dgx-spark-nlp-experiments:custom .

# Run with custom command
docker run --gpus all -v $(pwd)/results:/workspace/results \
    dgx-spark-nlp-experiments:custom \
    python benchmarks/inference.py --model "your-model" --size "8B"
```

## Troubleshooting

### Docker Issues

**NVIDIA Docker not working:**
```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

**Python version conflicts in Docker:**
- The Dockerfile explicitly uses Python 3.10
- If you need a different version, modify the Dockerfile

**Out of Memory Errors**

- Reduce batch size
- Use quantization (INT8/INT4)
- Use smaller models
- Enable gradient checkpointing

### Model Download Issues

- Set `HF_TOKEN` environment variable for gated models
- Use `huggingface-cli login` for authentication
- Check disk space for model cache

### CUDA Errors

- Verify CUDA installation: `nvidia-smi`
- Check CUDA version compatibility
- Ensure proper GPU drivers

## Contributing

When adding new experiments:
1. Create script in appropriate directory
2. Add experiment definition to `configs/experiments.yaml`
3. Ensure proper dependency resolution
4. Add results saving functionality
5. Update this README

## License

This project is for research purposes. Check individual model licenses before use.

## References

- [NVIDIA DGX Spark](https://www.nvidia.com/en-us/data-center/dgx-spark/)
- [Grace Blackwell Architecture](https://www.nvidia.com/en-us/data-center/grace-blackwell/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [vLLM](https://github.com/vllm-project/vllm)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)

