"""
Generate final evaluation report from all benchmark results
"""

import json
from pathlib import Path
from typing import Dict, List
import sys


def load_all_results(results_dir: str = "results") -> Dict:
    """Load all result files"""
    results = {
        "inference": [],
        "inference_engines": [],
        "training": [],
        "framework_comparison": [],
        "finetuning": [],
        "multimodal": [],
    }
    
    results_path = Path(results_dir)
    
    # Load inference benchmarks
    for file in (results_path / "benchmarks").glob("inference_*.json"):
        with open(file) as f:
            results["inference"].append(json.load(f))
    
    # Load inference engine comparisons
    for file in (results_path / "benchmarks").glob("engine_comparison_*.json"):
        with open(file) as f:
            results["inference_engines"].append(json.load(f))
    
    # Load training benchmarks
    for file in (results_path / "benchmarks").glob("training_*.json"):
        with open(file) as f:
            results["training"].append(json.load(f))
    
    # Load framework comparisons
    for file in (results_path / "benchmarks").glob("framework_comparison_*.json"):
        with open(file) as f:
            results["framework_comparison"].append(json.load(f))
    
    # Load fine-tuning results
    for file in (results_path / "finetuning").glob("finetuning_*.json"):
        with open(file) as f:
            results["finetuning"].append(json.load(f))
    
    # Load multimodal results
    for file in (results_path / "multimodal").glob("multimodal_*.json"):
        with open(file) as f:
            results["multimodal"].append(json.load(f))
    
    return results


def generate_markdown_report(results: Dict, output_file: str = "reports/benchmark_summary.md"):
    """Generate markdown report"""
    report = []
    report.append("# DGX Spark NLP Experiments - Final Report\n")
    report.append("Generated from benchmark results\n\n")
    
    # Inference benchmarks
    if results["inference"]:
        report.append("## Inference Benchmarks\n\n")
        report.append("| Model | Size | Quantization | Avg Tokens/sec | P50 Latency (ms) | Peak Memory (GB) |\n")
        report.append("|-------|------|--------------|----------------|------------------|------------------|\n")
        
        for r in results["inference"]:
            report.append(
                f"| {r['model_name']} | {r['model_size']} | {r.get('quantization', 'fp16')} | "
                f"{r['avg_tokens_per_second']:.2f} | {r['p50_latency']:.2f} | "
                f"{r.get('peak_memory_gb', 'N/A')} |\n"
            )
        report.append("\n")
    
    # Inference engine comparison
    if results["inference_engines"]:
        report.append("## Inference Engine Comparison\n\n")
        for comp in results["inference_engines"]:
            report.append(f"### {comp['model_name']}\n\n")
            report.append("| Engine | Throughput (tokens/sec) | P50 Latency (ms) | Memory (GB) |\n")
            report.append("|--------|------------------------|------------------|------------|\n")
            
            for engine in comp["engines"]:
                report.append(
                    f"| {engine['engine_name']} | {engine['throughput_tokens_per_sec']:.2f} | "
                    f"{engine['latency_p50_ms']:.2f} | {engine.get('memory_used_gb', 'N/A')} |\n"
                )
            report.append(f"\n**Best Throughput**: {comp['best_throughput']}\n")
            report.append(f"**Best Latency**: {comp['best_latency']}\n")
            report.append(f"**Most Memory Efficient**: {comp['most_memory_efficient']}\n\n")
    
    # Training benchmarks
    if results["training"]:
        report.append("## Training Benchmarks\n\n")
        report.append("| Model | Method | Samples/sec | Peak Memory (GB) | Final Loss |\n")
        report.append("|-------|--------|-------------|-----------------|------------|\n")
        
        for r in results["training"]:
            report.append(
                f"| {r['model_name']} | {r['training_method']} | {r['samples_per_second']:.2f} | "
                f"{r.get('peak_memory_gb', 'N/A')} | {r.get('final_loss', 'N/A')} |\n"
            )
        report.append("\n")
    
    # Framework comparison
    if results["framework_comparison"]:
        report.append("## Framework Comparison\n\n")
        for comp in results["framework_comparison"]:
            report.append(f"### {comp['model_name']}\n\n")
            report.append("| Framework | Inference Time (ms) | Training Time (ms) | Memory (GB) |\n")
            report.append("|-----------|-------------------|------------------|------------|\n")
            
            for framework in comp["frameworks"]:
                report.append(
                    f"| {framework['framework']} | {framework['inference_time']*1000:.2f} | "
                    f"{framework['training_time']*1000:.2f} | {framework.get('memory_used_gb', 'N/A')} |\n"
                )
            report.append(f"\n**Best Inference**: {comp['best_inference']}\n")
            report.append(f"**Best Training**: {comp['best_training']}\n")
            report.append(f"**Most Memory Efficient**: {comp['most_memory_efficient']}\n\n")
    
    # Fine-tuning results
    if results["finetuning"]:
        report.append("## Financial Domain Fine-tuning\n\n")
        report.append("| Model | Method | Val Accuracy | Peak Memory (GB) |\n")
        report.append("|-------|--------|--------------|-----------------|\n")
        
        for r in results["finetuning"]:
            report.append(
                f"| {r['model_name']} | {r['method']} | {r.get('val_accuracy', 'N/A')} | "
                f"{r.get('peak_memory_gb', 'N/A')} |\n"
            )
        report.append("\n")
    
    # Multimodal results
    if results["multimodal"]:
        report.append("## Multimodal Experiments\n\n")
        report.append("| Model | Task | Inference Time (ms) | Memory (GB) |\n")
        report.append("|-------|------|---------------------|------------|\n")
        
        for r in results["multimodal"]:
            report.append(
                f"| {r['model_name']} | {r['task']} | {r['inference_time']*1000:.2f} | "
                f"{r.get('memory_used_gb', 'N/A')} |\n"
            )
        report.append("\n")
    
    # Save report
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(''.join(report))
    
    print(f"Report generated: {output_file}")


def main():
    """Main report generation"""
    results = load_all_results()
    generate_markdown_report(results)


if __name__ == "__main__":
    main()

