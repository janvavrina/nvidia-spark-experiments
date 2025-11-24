"""
Inference Engine Comparison
Compares vLLM, llama.cpp, Lemonade, and Parallax inference engines.
"""

import subprocess
import json
import time
import statistics
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.profiling import MemoryProfiler


@dataclass
class EngineMetrics:
    """Metrics for an inference engine"""
    engine_name: str
    model_name: str
    throughput_tokens_per_sec: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    first_token_latency_ms: float
    memory_used_gb: Optional[float] = None
    memory_efficiency_gb_per_billion_params: Optional[float] = None
    max_batch_size: Optional[int] = None
    quantization_support: List[str] = None


@dataclass
class EngineComparison:
    """Comparison results across engines"""
    model_name: str
    model_size: str
    engines: List[EngineMetrics]
    best_throughput: str
    best_latency: str
    most_memory_efficient: str


class InferenceEngineBenchmark:
    """Benchmarks different inference engines"""
    
    def __init__(self, output_dir: str = "results/benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def benchmark_vllm(
        self,
        model_name: str,
        prompts: List[str],
        max_tokens: int = 100,
        num_runs: int = 5,
    ) -> EngineMetrics:
        """Benchmark vLLM inference engine"""
        print(f"\nBenchmarking vLLM with model: {model_name}")
        
        try:
            from vllm import LLM, SamplingParams
            
            # Initialize vLLM
            llm = LLM(model=model_name, dtype="half")
            
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=max_tokens,
            )
            
            # Run benchmarks
            latencies = []
            tokens_generated = []
            first_token_times = []
            
            profiler = MemoryProfiler()
            profiler.start()
            
            for i in range(num_runs):
                prompt = prompts[i % len(prompts)]
                start = time.time()
                
                outputs = llm.generate([prompt], sampling_params)
                
                end = time.time()
                latency = (end - start) * 1000  # ms
                latencies.append(latency)
                
                # Get token count
                output = outputs[0]
                tokens = len(output.outputs[0].token_ids)
                tokens_generated.append(tokens)
                
                # Estimate first token latency (simplified)
                first_token_times.append(latency / tokens if tokens > 0 else 0)
            
            metrics = profiler.stop()
            
            throughput = sum(tokens_generated) / sum(latencies) * 1000 if latencies else 0
            
            return EngineMetrics(
                engine_name="vLLM",
                model_name=model_name,
                throughput_tokens_per_sec=throughput,
                latency_p50_ms=statistics.median(latencies),
                latency_p95_ms=statistics.quantiles(latencies, n=20)[18] if len(latencies) > 1 else latencies[0],
                latency_p99_ms=statistics.quantiles(latencies, n=100)[98] if len(latencies) > 1 else latencies[0],
                first_token_latency_ms=statistics.mean(first_token_times),
                memory_used_gb=metrics.peak_gpu_memory,
                quantization_support=["fp16", "int8", "awq", "gptq"],
            )
        except ImportError:
            print("vLLM not available, skipping...")
            return None
        except Exception as e:
            print(f"Error benchmarking vLLM: {e}")
            return None
    
    def benchmark_llama_cpp(
        self,
        model_path: str,
        prompts: List[str],
        max_tokens: int = 100,
        num_runs: int = 5,
    ) -> EngineMetrics:
        """Benchmark llama.cpp inference engine"""
        print(f"\nBenchmarking llama.cpp with model: {model_path}")
        
        try:
            from llama_cpp import Llama
            
            # Load model
            llm = Llama(
                model_path=model_path,
                n_ctx=2048,
                n_gpu_layers=-1,  # Use GPU
            )
            
            latencies = []
            tokens_generated = []
            first_token_times = []
            
            profiler = MemoryProfiler()
            profiler.start()
            
            for i in range(num_runs):
                prompt = prompts[i % len(prompts)]
                start = time.time()
                
                output = llm(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
                
                end = time.time()
                latency = (end - start) * 1000  # ms
                latencies.append(latency)
                
                tokens = len(output["choices"][0]["text"].split())
                tokens_generated.append(tokens)
                first_token_times.append(latency / tokens if tokens > 0 else 0)
            
            metrics = profiler.stop()
            
            throughput = sum(tokens_generated) / sum(latencies) * 1000 if latencies else 0
            
            return EngineMetrics(
                engine_name="llama.cpp",
                model_name=model_path,
                throughput_tokens_per_sec=throughput,
                latency_p50_ms=statistics.median(latencies),
                latency_p95_ms=statistics.quantiles(latencies, n=20)[18] if len(latencies) > 1 else latencies[0],
                latency_p99_ms=statistics.quantiles(latencies, n=100)[98] if len(latencies) > 1 else latencies[0],
                first_token_latency_ms=statistics.mean(first_token_times),
                memory_used_gb=metrics.peak_gpu_memory,
                quantization_support=["fp16", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0"],
            )
        except ImportError:
            print("llama-cpp-python not available, skipping...")
            return None
        except Exception as e:
            print(f"Error benchmarking llama.cpp: {e}")
            return None
    
    def benchmark_lemonade(
        self,
        model_name: str,
        prompts: List[str],
        max_tokens: int = 100,
        num_runs: int = 5,
    ) -> EngineMetrics:
        """Benchmark Lemonade inference engine"""
        print(f"\nBenchmarking Lemonade with model: {model_name}")
        
        # Lemonade may not be available - placeholder implementation
        try:
            # This would require Lemonade to be installed
            # Placeholder for actual implementation
            print("Lemonade not available or not implemented yet")
            return None
        except Exception as e:
            print(f"Error benchmarking Lemonade: {e}")
            return None
    
    def benchmark_parallax(
        self,
        model_name: str,
        prompts: List[str],
        max_tokens: int = 100,
        num_runs: int = 5,
    ) -> EngineMetrics:
        """Benchmark Parallax inference engine"""
        print(f"\nBenchmarking Parallax with model: {model_name}")
        
        # Parallax may not be available - placeholder implementation
        try:
            # This would require Parallax to be installed
            # Placeholder for actual implementation
            print("Parallax not available or not implemented yet")
            return None
        except Exception as e:
            print(f"Error benchmarking Parallax: {e}")
            return None
    
    def compare_engines(
        self,
        model_name: str,
        model_size: str,
        prompts: List[str],
        max_tokens: int = 100,
        num_runs: int = 5,
        llama_cpp_path: Optional[str] = None,
    ) -> EngineComparison:
        """Compare all available inference engines"""
        print(f"\n{'='*60}")
        print(f"Inference Engine Comparison")
        print(f"Model: {model_name}")
        print(f"Size: {model_size}")
        print(f"{'='*60}")
        
        engines = []
        
        # Benchmark each engine
        vllm_result = self.benchmark_vllm(model_name, prompts, max_tokens, num_runs)
        if vllm_result:
            engines.append(vllm_result)
        
        if llama_cpp_path:
            llama_result = self.benchmark_llama_cpp(llama_cpp_path, prompts, max_tokens, num_runs)
            if llama_result:
                engines.append(llama_result)
        
        lemonade_result = self.benchmark_lemonade(model_name, prompts, max_tokens, num_runs)
        if lemonade_result:
            engines.append(lemonade_result)
        
        parallax_result = self.benchmark_parallax(model_name, prompts, max_tokens, num_runs)
        if parallax_result:
            engines.append(parallax_result)
        
        if not engines:
            raise ValueError("No inference engines available for benchmarking")
        
        # Find best performers
        best_throughput = max(engines, key=lambda e: e.throughput_tokens_per_sec).engine_name
        best_latency = min(engines, key=lambda e: e.latency_p50_ms).engine_name
        most_memory_efficient = min(
            engines,
            key=lambda e: e.memory_used_gb if e.memory_used_gb else float('inf')
        ).engine_name
        
        comparison = EngineComparison(
            model_name=model_name,
            model_size=model_size,
            engines=engines,
            best_throughput=best_throughput,
            best_latency=best_latency,
            most_memory_efficient=most_memory_efficient,
        )
        
        return comparison
    
    def save_comparison(self, comparison: EngineComparison):
        """Save comparison results"""
        filename = f"engine_comparison_{comparison.model_name.replace('/', '_')}.json"
        filepath = self.output_dir / filename
        
        data = asdict(comparison)
        data["engines"] = [asdict(e) for e in comparison.engines]
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nComparison saved to: {filepath}")
        return filepath


def main():
    """Main comparison execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare inference engines")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--size", type=str, required=True, help="Model size description")
    parser.add_argument("--llama-cpp-path", type=str, help="Path to GGUF model file for llama.cpp")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="results/benchmarks")
    
    args = parser.parse_args()
    
    prompts = [
        "The future of artificial intelligence is",
        "In the field of natural language processing,",
        "Machine learning models have revolutionized",
        "Deep learning architectures such as",
        "The transformer model introduced",
    ]
    
    benchmark = InferenceEngineBenchmark(output_dir=args.output_dir)
    
    comparison = benchmark.compare_engines(
        model_name=args.model,
        model_size=args.size,
        prompts=prompts,
        max_tokens=args.max_tokens,
        num_runs=args.num_runs,
        llama_cpp_path=args.llama_cpp_path,
    )
    
    benchmark.save_comparison(comparison)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Engine Comparison Summary")
    print(f"{'='*60}")
    for engine in comparison.engines:
        print(f"\n{engine.engine_name}:")
        print(f"  Throughput: {engine.throughput_tokens_per_sec:.2f} tokens/sec")
        print(f"  P50 Latency: {engine.latency_p50_ms:.2f} ms")
        print(f"  Memory: {engine.memory_used_gb:.2f} GB" if engine.memory_used_gb else "  Memory: N/A")
    
    print(f"\nBest Throughput: {comparison.best_throughput}")
    print(f"Best Latency: {comparison.best_latency}")
    print(f"Most Memory Efficient: {comparison.most_memory_efficient}")


if __name__ == "__main__":
    main()

