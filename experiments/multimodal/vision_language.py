"""
Multimodal AI Experiments
Tests vision-language models and video processing on DGX Spark.
"""

import time
import json
import torch
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass, asdict
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.profiling import MemoryProfiler


@dataclass
class MultimodalMetrics:
    """Metrics for multimodal experiments"""
    model_name: str
    task: str  # "vision_language", "video", "image_generation"
    inference_time: float
    memory_used_gb: Optional[float] = None
    throughput_samples_per_sec: Optional[float] = None


class MultimodalBenchmark:
    """Benchmarks multimodal AI models"""
    
    def __init__(self, output_dir: str = "results/multimodal"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def benchmark_qwen3_vl(
        self,
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        num_runs: int = 5,
    ) -> MultimodalMetrics:
        """Benchmark Qwen3-VL model"""
        print(f"\nBenchmarking Qwen3-VL: {model_name}")
        
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            from PIL import Image
            import numpy as np
            
            profiler = MemoryProfiler()
            profiler.start()
            
            # Load model
            processor = AutoProcessor.from_pretrained(model_name)
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            model.eval()
            
            # Create dummy image
            dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            prompt = "What is in this image?"
            
            inference_times = []
            for _ in range(num_runs):
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": dummy_image},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = processor.process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(model.device)
                
                start = time.time()
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=50)
                inference_times.append(time.time() - start)
            
            avg_time = sum(inference_times) / len(inference_times)
            
            metrics = profiler.stop()
            
            return MultimodalMetrics(
                model_name=model_name,
                task="vision_language",
                inference_time=avg_time,
                memory_used_gb=metrics.peak_gpu_memory,
                throughput_samples_per_sec=1.0 / avg_time if avg_time > 0 else 0,
            )
        except Exception as e:
            print(f"Error benchmarking Qwen3-VL: {e}")
            return None
    
    def benchmark_deepseek_vl2(
        self,
        model_name: str = "deepseek-ai/DeepSeek-VL2",
        num_runs: int = 5,
    ) -> MultimodalMetrics:
        """Benchmark DeepSeek-VL2 model"""
        print(f"\nBenchmarking DeepSeek-VL2: {model_name}")
        
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
            from PIL import Image
            import numpy as np
            
            profiler = MemoryProfiler()
            profiler.start()
            
            # Load model
            processor = AutoProcessor.from_pretrained(model_name)
            model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            model.eval()
            
            # Create dummy image
            dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            prompt = "What is in this image?"
            
            inference_times = []
            for _ in range(num_runs):
                inputs = processor(images=dummy_image, text=prompt, return_tensors="pt").to(model.device)
                
                start = time.time()
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=50)
                inference_times.append(time.time() - start)
            
            avg_time = sum(inference_times) / len(inference_times)
            
            metrics = profiler.stop()
            
            return MultimodalMetrics(
                model_name=model_name,
                task="vision_language",
                inference_time=avg_time,
                memory_used_gb=metrics.peak_gpu_memory,
                throughput_samples_per_sec=1.0 / avg_time if avg_time > 0 else 0,
            )
        except Exception as e:
            print(f"Error benchmarking DeepSeek-VL2: {e}")
            return None
    
    def benchmark_gemma3(
        self,
        model_name: str = "google/gemma-3-4b-it",
        num_runs: int = 5,
    ) -> MultimodalMetrics:
        """Benchmark Gemma 3 multimodal model"""
        print(f"\nBenchmarking Gemma 3: {model_name}")
        
        try:
            from transformers import Gemma3ForConditionalGeneration, AutoProcessor
            from PIL import Image
            import numpy as np
            
            profiler = MemoryProfiler()
            profiler.start()
            
            # Load model
            processor = AutoProcessor.from_pretrained(model_name)
            model = Gemma3ForConditionalGeneration.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
            model.eval()
            
            # Create dummy image
            dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            
            inference_times = []
            for _ in range(num_runs):
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "url": dummy_image},
                            {"type": "text", "text": "What is in this image?"},
                        ],
                    }
                ]
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    add_generation_prompt=True,
                ).to(model.device)
                
                start = time.time()
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=50)
                inference_times.append(time.time() - start)
            
            avg_time = sum(inference_times) / len(inference_times)
            
            metrics = profiler.stop()
            
            return MultimodalMetrics(
                model_name=model_name,
                task="vision_language",
                inference_time=avg_time,
                memory_used_gb=metrics.peak_gpu_memory,
                throughput_samples_per_sec=1.0 / avg_time if avg_time > 0 else 0,
            )
        except Exception as e:
            print(f"Error benchmarking Gemma 3: {e}")
            return None
    
    def save_results(self, metrics: MultimodalMetrics):
        """Save multimodal benchmark results"""
        filename = f"multimodal_{metrics.model_name.replace('/', '_')}_{metrics.task}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(asdict(metrics), f, indent=2)
        
        print(f"\nResults saved to: {filepath}")
        return filepath


def main():
    """Main multimodal benchmark execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run multimodal benchmarks")
    parser.add_argument("--model", type=str, choices=["qwen3_vl", "deepseek_vl2", "gemma3"], default="qwen3_vl")
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="results/multimodal")
    
    args = parser.parse_args()
    
    benchmark = MultimodalBenchmark(output_dir=args.output_dir)
    
    if args.model == "qwen3_vl":
        metrics = benchmark.benchmark_qwen3_vl(num_runs=args.num_runs)
    elif args.model == "deepseek_vl2":
        metrics = benchmark.benchmark_deepseek_vl2(num_runs=args.num_runs)
    elif args.model == "gemma3":
        metrics = benchmark.benchmark_gemma3(num_runs=args.num_runs)
    
    if metrics:
        benchmark.save_results(metrics)
        
        print(f"\n{'='*60}")
        print(f"Multimodal Benchmark Summary")
        print(f"{'='*60}")
        print(f"Inference time: {metrics.inference_time*1000:.2f} ms")
        if metrics.throughput_samples_per_sec:
            print(f"Throughput: {metrics.throughput_samples_per_sec:.2f} samples/sec")
        if metrics.memory_used_gb:
            print(f"Memory: {metrics.memory_used_gb:.2f} GB")


if __name__ == "__main__":
    import torch
    main()

