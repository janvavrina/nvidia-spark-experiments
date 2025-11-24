"""
LLM Inference Benchmarking Suite
Tests inference performance for models sized appropriately for 128GB unified memory.
"""

import time
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.profiling import MemoryProfiler, save_metrics


@dataclass
class InferenceMetrics:
    """Metrics for a single inference run"""
    tokens_generated: int
    total_time: float
    tokens_per_second: float
    first_token_latency: float
    avg_token_latency: float
    memory_peak_gb: Optional[float] = None


@dataclass
class BenchmarkResult:
    """Results for a benchmark run"""
    model_name: str
    model_size: str
    quantization: Optional[str]
    batch_size: int
    sequence_length: int
    num_runs: int
    metrics: List[InferenceMetrics]
    avg_tokens_per_second: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    avg_first_token_latency: float
    peak_memory_gb: Optional[float] = None


class InferenceBenchmark:
    """Benchmarks LLM inference performance"""
    
    def __init__(self, output_dir: str = "results/benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self, model_name: str, quantization: Optional[str] = None):
        """Load a model with optional quantization"""
        print(f"Loading model: {model_name} (quantization: {quantization})")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model with appropriate settings
        if quantization == "int8":
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        elif quantization == "int4":
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        else:
            # Full precision or FP16
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        
        model.eval()
        print(f"Model loaded on device: {self.device}")
        
        return model, tokenizer
    
    def generate_text(
        self,
        model,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 100,
        batch_size: int = 1,
    ) -> InferenceMetrics:
        """Generate text and measure performance"""
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Warmup
        with torch.no_grad():
            _ = model.generate(
                inputs.input_ids,
                max_new_tokens=10,
                do_sample=False,
            )
        
        torch.cuda.synchronize() if self.device == "cuda" else None
        
        # Profile memory
        profiler = MemoryProfiler(interval=0.01)
        profiler.start()
        
        # Measure first token latency
        start_time = time.time()
        with torch.no_grad():
            output = model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize() if self.device == "cuda" else None
        end_time = time.time()
        
        metrics = profiler.stop()
        
        # Calculate metrics
        generated_tokens = output.shape[1] - inputs.input_ids.shape[1]
        total_time = end_time - start_time
        tokens_per_second = generated_tokens / total_time if total_time > 0 else 0
        
        # Estimate first token latency (simplified)
        first_token_latency = total_time / generated_tokens if generated_tokens > 0 else 0
        avg_token_latency = total_time / generated_tokens if generated_tokens > 0 else 0
        
        return InferenceMetrics(
            tokens_generated=generated_tokens,
            total_time=total_time,
            tokens_per_second=tokens_per_second,
            first_token_latency=first_token_latency,
            avg_token_latency=avg_token_latency,
            memory_peak_gb=metrics.peak_gpu_memory,
        )
    
    def run_benchmark(
        self,
        model_name: str,
        model_size: str,
        quantization: Optional[str],
        prompts: List[str],
        max_new_tokens: int = 100,
        batch_size: int = 1,
        num_runs: int = 5,
    ) -> BenchmarkResult:
        """Run inference benchmark"""
        print(f"\n{'='*60}")
        print(f"Running inference benchmark")
        print(f"Model: {model_name}")
        print(f"Size: {model_size}")
        print(f"Quantization: {quantization or 'None'}")
        print(f"Batch size: {batch_size}")
        print(f"Number of runs: {num_runs}")
        print(f"{'='*60}")
        
        # Load model
        model, tokenizer = self.load_model(model_name, quantization)
        
        # Run benchmarks
        all_metrics = []
        for i in range(num_runs):
            prompt = prompts[i % len(prompts)]
            print(f"\nRun {i+1}/{num_runs}: Generating {max_new_tokens} tokens...")
            
            metrics = self.generate_text(
                model,
                tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
            )
            
            all_metrics.append(metrics)
            print(f"  Tokens/sec: {metrics.tokens_per_second:.2f}")
            print(f"  First token latency: {metrics.first_token_latency*1000:.2f} ms")
        
        # Calculate statistics
        tokens_per_second_list = [m.tokens_per_second for m in all_metrics]
        first_token_latencies = [m.first_token_latency for m in all_metrics]
        token_latencies = [m.avg_token_latency for m in all_metrics]
        
        result = BenchmarkResult(
            model_name=model_name,
            model_size=model_size,
            quantization=quantization,
            batch_size=batch_size,
            sequence_length=max_new_tokens,
            num_runs=num_runs,
            metrics=all_metrics,
            avg_tokens_per_second=statistics.mean(tokens_per_second_list),
            p50_latency=statistics.median(token_latencies) * 1000,  # ms
            p95_latency=statistics.quantiles(token_latencies, n=20)[18] * 1000 if len(token_latencies) > 1 else 0,
            p99_latency=statistics.quantiles(token_latencies, n=100)[98] * 1000 if len(token_latencies) > 1 else 0,
            avg_first_token_latency=statistics.mean(first_token_latencies) * 1000,  # ms
            peak_memory_gb=max(m.memory_peak_gb for m in all_metrics if m.memory_peak_gb) if any(m.memory_peak_gb for m in all_metrics) else None,
        )
        
        # Cleanup
        del model
        del tokenizer
        torch.cuda.empty_cache() if self.device == "cuda" else None
        
        return result
    
    def save_results(self, result: BenchmarkResult):
        """Save benchmark results to JSON"""
        filename = f"inference_{result.model_name.replace('/', '_')}_{result.quantization or 'fp16'}.json"
        filepath = self.output_dir / filename
        
        data = asdict(result)
        # Convert metrics to dict
        data["metrics"] = [asdict(m) for m in result.metrics]
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nResults saved to: {filepath}")
        return filepath


def main():
    """Main benchmark execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run LLM inference benchmarks")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--size", type=str, required=True, help="Model size description")
    parser.add_argument("--quantization", type=str, choices=["int8", "int4", None], default=None)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="results/benchmarks")
    
    args = parser.parse_args()
    
    # Default prompts for benchmarking
    prompts = [
        "The future of artificial intelligence is",
        "In the field of natural language processing,",
        "Machine learning models have revolutionized",
        "Deep learning architectures such as",
        "The transformer model introduced",
    ]
    
    benchmark = InferenceBenchmark(output_dir=args.output_dir)
    
    result = benchmark.run_benchmark(
        model_name=args.model,
        model_size=args.size,
        quantization=args.quantization,
        prompts=prompts,
        max_new_tokens=args.max_tokens,
        batch_size=args.batch_size,
        num_runs=args.num_runs,
    )
    
    benchmark.save_results(result)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Benchmark Summary")
    print(f"{'='*60}")
    print(f"Average tokens/sec: {result.avg_tokens_per_second:.2f}")
    print(f"P50 latency: {result.p50_latency:.2f} ms")
    print(f"P95 latency: {result.p95_latency:.2f} ms")
    print(f"P99 latency: {result.p99_latency:.2f} ms")
    print(f"Avg first token latency: {result.avg_first_token_latency:.2f} ms")
    if result.peak_memory_gb:
        print(f"Peak memory: {result.peak_memory_gb:.2f} GB")


if __name__ == "__main__":
    main()

