"""
Training Performance Benchmarks
Measures training speed, memory efficiency, and convergence for different model sizes.
"""

import time
import json
import torch
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass, asdict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.profiling import MemoryProfiler, save_metrics


@dataclass
class TrainingMetrics:
    """Metrics for training performance"""
    model_name: str
    model_size: str
    training_method: str  # "full", "lora", "qlora"
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    num_epochs: int
    total_time: float
    time_per_epoch: float
    samples_per_second: float
    peak_memory_gb: Optional[float] = None
    final_loss: Optional[float] = None


class TrainingBenchmark:
    """Benchmarks training performance"""
    
    def __init__(self, output_dir: str = "results/benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def prepare_dataset(self, dataset_name: str = "wikitext", subset: str = "wikitext-2-raw-v1"):
        """Prepare dataset for training"""
        print(f"Loading dataset: {dataset_name}/{subset}")
        dataset = load_dataset(dataset_name, subset, split="train[:1%]")  # Small subset for benchmarking
        return dataset
    
    def tokenize_dataset(self, dataset, tokenizer, max_length: int = 512):
        """Tokenize dataset"""
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
        
        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )
        return tokenized
    
    def benchmark_full_finetuning(
        self,
        model_name: str,
        model_size: str,
        dataset,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 5e-5,
        num_epochs: int = 1,
        max_steps: int = 10,  # Small number for benchmarking
    ) -> TrainingMetrics:
        """Benchmark full fine-tuning"""
        print(f"\nBenchmarking full fine-tuning: {model_name}")
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        
        # Prepare dataset
        tokenized = self.tokenize_dataset(dataset, tokenizer)
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./tmp_training",
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            max_steps=max_steps,
            logging_steps=1,
            save_strategy="no",
            fp16=True,
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=data_collator,
        )
        
        # Profile training
        profiler = MemoryProfiler()
        profiler.start()
        
        start_time = time.time()
        trainer.train()
        end_time = time.time()
        
        metrics = profiler.stop()
        
        total_time = end_time - start_time
        num_samples = len(tokenized) * num_epochs if max_steps == -1 else max_steps * batch_size * gradient_accumulation_steps
        
        return TrainingMetrics(
            model_name=model_name,
            model_size=model_size,
            training_method="full",
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            total_time=total_time,
            time_per_epoch=total_time / num_epochs,
            samples_per_second=num_samples / total_time if total_time > 0 else 0,
            peak_memory_gb=metrics.peak_gpu_memory,
            final_loss=trainer.state.log_history[-1].get("train_loss") if trainer.state.log_history else None,
        )
    
    def benchmark_lora(
        self,
        model_name: str,
        model_size: str,
        dataset,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 1e-4,
        num_epochs: int = 1,
        max_steps: int = 10,
    ) -> TrainingMetrics:
        """Benchmark LoRA fine-tuning"""
        print(f"\nBenchmarking LoRA fine-tuning: {model_name}")
        
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            
            # LoRA configuration
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj"],  # Common for LLaMA models
            )
            
            model = get_peft_model(model, lora_config)
            
            # Prepare dataset
            tokenized = self.tokenize_dataset(dataset, tokenizer)
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
            )
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir="./tmp_training",
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=learning_rate,
                num_train_epochs=num_epochs,
                max_steps=max_steps,
                logging_steps=1,
                save_strategy="no",
                fp16=True,
            )
            
            # Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized,
                data_collator=data_collator,
            )
            
            # Profile training
            profiler = MemoryProfiler()
            profiler.start()
            
            start_time = time.time()
            trainer.train()
            end_time = time.time()
            
            metrics = profiler.stop()
            
            total_time = end_time - start_time
            num_samples = len(tokenized) * num_epochs if max_steps == -1 else max_steps * batch_size * gradient_accumulation_steps
            
            return TrainingMetrics(
                model_name=model_name,
                model_size=model_size,
                training_method="lora",
                batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                total_time=total_time,
                time_per_epoch=total_time / num_epochs,
                samples_per_second=num_samples / total_time if total_time > 0 else 0,
                peak_memory_gb=metrics.peak_gpu_memory,
                final_loss=trainer.state.log_history[-1].get("train_loss") if trainer.state.log_history else None,
            )
        except ImportError:
            print("PEFT not available, skipping LoRA benchmark")
            return None
    
    def save_results(self, metrics: TrainingMetrics):
        """Save training benchmark results"""
        filename = f"training_{metrics.model_name.replace('/', '_')}_{metrics.training_method}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(asdict(metrics), f, indent=2)
        
        print(f"\nResults saved to: {filepath}")
        return filepath


def main():
    """Main training benchmark execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run training benchmarks")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--size", type=str, required=True, help="Model size description")
    parser.add_argument("--method", type=str, choices=["full", "lora", "qlora"], default="lora")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="results/benchmarks")
    
    args = parser.parse_args()
    
    benchmark = TrainingBenchmark(output_dir=args.output_dir)
    
    # Load small dataset
    dataset = benchmark.prepare_dataset()
    
    if args.method == "full":
        metrics = benchmark.benchmark_full_finetuning(
            model_name=args.model,
            model_size=args.size,
            dataset=dataset,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            num_epochs=args.epochs,
            max_steps=args.max_steps,
        )
    elif args.method == "lora":
        metrics = benchmark.benchmark_lora(
            model_name=args.model,
            model_size=args.size,
            dataset=dataset,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            num_epochs=args.epochs,
            max_steps=args.max_steps,
        )
    else:
        print(f"Method {args.method} not yet implemented")
        return
    
    if metrics:
        benchmark.save_results(metrics)
        
        print(f"\n{'='*60}")
        print(f"Training Benchmark Summary")
        print(f"{'='*60}")
        print(f"Total time: {metrics.total_time:.2f} seconds")
        print(f"Time per epoch: {metrics.time_per_epoch:.2f} seconds")
        print(f"Samples per second: {metrics.samples_per_second:.2f}")
        if metrics.peak_memory_gb:
            print(f"Peak memory: {metrics.peak_memory_gb:.2f} GB")
        if metrics.final_loss:
            print(f"Final loss: {metrics.final_loss:.4f}")


if __name__ == "__main__":
    main()

