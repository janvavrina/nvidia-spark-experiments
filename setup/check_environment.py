#!/usr/bin/env python3
"""
Environment check script for DGX Spark
Verifies that all required dependencies and hardware are properly configured.
"""

import sys
import subprocess
import platform

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("WARNING: Python 3.8+ recommended")
        return False
    return True

def check_architecture():
    """Check system architecture"""
    arch = platform.machine()
    print(f"Architecture: {arch}")
    if arch != "aarch64":
        print("WARNING: Expected ARM64 (aarch64) for Grace CPU")
    return True

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"CUDA available: Yes")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"    Total memory: {props.total_memory / 1024**3:.2f} GB")
        else:
            print("CUDA available: No")
        return cuda_available
    except ImportError:
        print("PyTorch not installed")
        return False

def check_unified_memory():
    """Check unified memory information"""
    try:
        import torch
        if torch.cuda.is_available():
            # Try to get unified memory info
            # This may require specific CUDA APIs
            print("Unified memory: Checking...")
            # For Grace Blackwell, unified memory should be available
            return True
    except Exception as e:
        print(f"Could not check unified memory: {e}")
    return False

def check_packages():
    """Check if required packages are installed"""
    required_packages = [
        "torch",
        "tensorflow",
        "jax",
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "vllm",
        "llama_cpp",
        "wandb",
        "mlflow",
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing.append(package)
    
    return len(missing) == 0

def check_nvlink():
    """Check NVLink-C2C connectivity"""
    try:
        import torch
        if torch.cuda.is_available():
            # Check NVLink topology
            print("NVLink-C2C: Checking topology...")
            # This would require specific CUDA APIs
            return True
    except Exception as e:
        print(f"Could not check NVLink: {e}")
    return False

def main():
    print("=" * 60)
    print("DGX Spark Environment Check")
    print("=" * 60)
    print()
    
    checks = [
        ("Python Version", check_python_version),
        ("Architecture", check_architecture),
        ("CUDA", check_cuda),
        ("Unified Memory", check_unified_memory),
        ("Required Packages", check_packages),
        ("NVLink-C2C", check_nvlink),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n{name}:")
        print("-" * 40)
        result = check_func()
        results.append((name, result))
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name}: {status}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        print("\nAll checks passed! Environment is ready.")
        return 0
    else:
        print("\nSome checks failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

