"""
Framework Comparison Suite
Compares PyTorch, TensorFlow, and JAX performance on DGX Spark.
"""

import time
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.profiling import MemoryProfiler


@dataclass
class FrameworkMetrics:
    """Metrics for a framework"""
    framework: str
    model_name: str
    inference_time: float
    training_time: float
    memory_used_gb: Optional[float] = None
    throughput_tokens_per_sec: Optional[float] = None
    setup_time: float = 0.0


@dataclass
class FrameworkComparison:
    """Comparison results across frameworks"""
    model_name: str
    model_size: str
    frameworks: List[FrameworkMetrics]
    best_inference: str
    best_training: str
    most_memory_efficient: str


class FrameworkBenchmark:
    """Benchmarks different ML frameworks"""
    
    def __init__(self, output_dir: str = "results/benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def benchmark_pytorch(
        self,
        model_name: str,
        num_inference_runs: int = 10,
        num_training_steps: int = 10,
    ) -> FrameworkMetrics:
        """Benchmark PyTorch"""
        print(f"\nBenchmarking PyTorch with model: {model_name}")
        
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            profiler = MemoryProfiler()
            profiler.start()
            
            setup_start = time.time()
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            model.eval()
            setup_time = time.time() - setup_start
            
            # Inference benchmark
            prompt = "The future of AI is"
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
            
            inference_times = []
            for _ in range(num_inference_runs):
                start = time.time()
                with torch.no_grad():
                    _ = model.generate(inputs.input_ids, max_new_tokens=50, do_sample=False)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                inference_times.append(time.time() - start)
            
            avg_inference_time = statistics.mean(inference_times)
            throughput = 50 / avg_inference_time if avg_inference_time > 0 else 0
            
            # Training benchmark (simplified)
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
            
            training_times = []
            for _ in range(num_training_steps):
                start = time.time()
                optimizer.zero_grad()
                outputs = model(inputs.input_ids, labels=inputs.input_ids)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                training_times.append(time.time() - start)
            
            avg_training_time = statistics.mean(training_times)
            
            metrics = profiler.stop()
            
            return FrameworkMetrics(
                framework="PyTorch",
                model_name=model_name,
                inference_time=avg_inference_time,
                training_time=avg_training_time,
                memory_used_gb=metrics.peak_gpu_memory,
                throughput_tokens_per_sec=throughput,
                setup_time=setup_time,
            )
        except Exception as e:
            print(f"Error benchmarking PyTorch: {e}")
            return None
    
    def benchmark_tensorflow(
        self,
        model_name: str,
        num_inference_runs: int = 10,
        num_training_steps: int = 10,
    ) -> FrameworkMetrics:
        """Benchmark TensorFlow"""
        print(f"\nBenchmarking TensorFlow with model: {model_name}")
        
        try:
            import tensorflow as tf
            from transformers import TFAutoModelForCausalLM, AutoTokenizer
            
            profiler = MemoryProfiler()
            profiler.start()
            
            setup_start = time.time()
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = TFAutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=tf.float16,
            )
            setup_time = time.time() - setup_start
            
            # Inference benchmark
            prompt = "The future of AI is"
            inputs = tokenizer(prompt, return_tensors="tf")
            
            inference_times = []
            for _ in range(num_inference_runs):
                start = time.time()
                _ = model.generate(inputs.input_ids, max_length=50, do_sample=False)
                inference_times.append(time.time() - start)
            
            avg_inference_time = statistics.mean(inference_times)
            throughput = 50 / avg_inference_time if avg_inference_time > 0 else 0
            
            # Training benchmark (simplified)
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
            
            training_times = []
            for _ in range(num_training_steps):
                start = time.time()
                with tf.GradientTape() as tape:
                    outputs = model(inputs.input_ids, training=True)
                    loss = tf.reduce_mean(tf.square(outputs.logits))
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                training_times.append(time.time() - start)
            
            avg_training_time = statistics.mean(training_times)
            
            metrics = profiler.stop()
            
            return FrameworkMetrics(
                framework="TensorFlow",
                model_name=model_name,
                inference_time=avg_inference_time,
                training_time=avg_training_time,
                memory_used_gb=metrics.peak_gpu_memory,
                throughput_tokens_per_sec=throughput,
                setup_time=setup_time,
            )
        except Exception as e:
            print(f"Error benchmarking TensorFlow: {e}")
            return None
    
    def benchmark_jax(
        self,
        model_name: str,
        num_inference_runs: int = 10,
        num_training_steps: int = 10,
    ) -> FrameworkMetrics:
        """Benchmark JAX"""
        print(f"\nBenchmarking JAX with model: {model_name}")
        
        try:
            import jax
            import jax.numpy as jnp
            from transformers import FlaxAutoModelForCausalLM, AutoTokenizer
            
            profiler = MemoryProfiler()
            profiler.start()
            
            setup_start = time.time()
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = FlaxAutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=jnp.float16,
            )
            setup_time = time.time() - setup_start
            
            # Inference benchmark
            prompt = "The future of AI is"
            inputs = tokenizer(prompt, return_tensors="np")
            
            inference_times = []
            for _ in range(num_inference_runs):
                start = time.time()
                _ = model.generate(inputs.input_ids, max_length=50, do_sample=False)
                inference_times.append(time.time() - start)
            
            avg_inference_time = statistics.mean(inference_times)
            throughput = 50 / avg_inference_time if avg_inference_time > 0 else 0
            
            # Training benchmark (simplified)
            from flax import optim
            
            optimizer = optim.Adam(learning_rate=1e-5).create(model.params)
            
            training_times = []
            for _ in range(num_training_steps):
                start = time.time()
                # Simplified training step
                def loss_fn(params):
                    outputs = model.apply(params, inputs.input_ids)
                    return jnp.mean(jnp.square(outputs.logits))
                
                loss, grads = jax.value_and_grad(loss_fn)(optimizer.target)
                optimizer = optimizer.apply_gradient(grads)
                training_times.append(time.time() - start)
            
            avg_training_time = statistics.mean(training_times)
            
            metrics = profiler.stop()
            
            return FrameworkMetrics(
                framework="JAX",
                model_name=model_name,
                inference_time=avg_inference_time,
                training_time=avg_training_time,
                memory_used_gb=metrics.peak_gpu_memory,
                throughput_tokens_per_sec=throughput,
                setup_time=setup_time,
            )
        except Exception as e:
            print(f"Error benchmarking JAX: {e}")
            return None
    
    def compare_frameworks(
        self,
        model_name: str,
        model_size: str,
        num_inference_runs: int = 10,
        num_training_steps: int = 10,
    ) -> FrameworkComparison:
        """Compare all available frameworks"""
        print(f"\n{'='*60}")
        print(f"Framework Comparison")
        print(f"Model: {model_name}")
        print(f"Size: {model_size}")
        print(f"{'='*60}")
        
        frameworks = []
        
        # Benchmark each framework
        pytorch_result = self.benchmark_pytorch(model_name, num_inference_runs, num_training_steps)
        if pytorch_result:
            frameworks.append(pytorch_result)
        
        tensorflow_result = self.benchmark_tensorflow(model_name, num_inference_runs, num_training_steps)
        if tensorflow_result:
            frameworks.append(tensorflow_result)
        
        jax_result = self.benchmark_jax(model_name, num_inference_runs, num_training_steps)
        if jax_result:
            frameworks.append(jax_result)
        
        if not frameworks:
            raise ValueError("No frameworks available for benchmarking")
        
        # Find best performers
        best_inference = min(frameworks, key=lambda f: f.inference_time).framework
        best_training = min(frameworks, key=lambda f: f.training_time).framework
        most_memory_efficient = min(
            frameworks,
            key=lambda f: f.memory_used_gb if f.memory_used_gb else float('inf')
        ).framework
        
        comparison = FrameworkComparison(
            model_name=model_name,
            model_size=model_size,
            frameworks=frameworks,
            best_inference=best_inference,
            best_training=best_training,
            most_memory_efficient=most_memory_efficient,
        )
        
        return comparison
    
    def save_comparison(self, comparison: FrameworkComparison):
        """Save comparison results"""
        filename = f"framework_comparison_{comparison.model_name.replace('/', '_')}.json"
        filepath = self.output_dir / filename
        
        data = asdict(comparison)
        data["frameworks"] = [asdict(f) for f in comparison.frameworks]
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nComparison saved to: {filepath}")
        return filepath


def main():
    """Main framework comparison execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare ML frameworks")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--size", type=str, required=True, help="Model size description")
    parser.add_argument("--inference-runs", type=int, default=10)
    parser.add_argument("--training-steps", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="results/benchmarks")
    
    args = parser.parse_args()
    
    benchmark = FrameworkBenchmark(output_dir=args.output_dir)
    
    comparison = benchmark.compare_frameworks(
        model_name=args.model,
        model_size=args.size,
        num_inference_runs=args.inference_runs,
        num_training_steps=args.training_steps,
    )
    
    benchmark.save_comparison(comparison)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Framework Comparison Summary")
    print(f"{'='*60}")
    for framework in comparison.frameworks:
        print(f"\n{framework.framework}:")
        print(f"  Inference time: {framework.inference_time*1000:.2f} ms")
        print(f"  Training time: {framework.training_time*1000:.2f} ms")
        print(f"  Throughput: {framework.throughput_tokens_per_sec:.2f} tokens/sec" if framework.throughput_tokens_per_sec else "  Throughput: N/A")
        print(f"  Memory: {framework.memory_used_gb:.2f} GB" if framework.memory_used_gb else "  Memory: N/A")
    
    print(f"\nBest Inference: {comparison.best_inference}")
    print(f"Best Training: {comparison.best_training}")
    print(f"Most Memory Efficient: {comparison.most_memory_efficient}")


if __name__ == "__main__":
    main()

