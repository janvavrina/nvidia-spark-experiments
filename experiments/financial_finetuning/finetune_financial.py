"""
Financial Domain Fine-tuning
Fine-tunes Granite 4.0, Gemma 3, and Qwen 3 on Financial PhraseBank dataset.
"""

import os
import json
import torch
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass, asdict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, TaskType
import sys
from sklearn.metrics import accuracy_score, f1_score, classification_report

sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.profiling import MemoryProfiler


@dataclass
class FinetuningResult:
    """Results from fine-tuning"""
    model_name: str
    base_model: str
    dataset: str
    method: str  # "lora", "qlora", "full"
    num_epochs: int
    batch_size: int
    learning_rate: float
    final_loss: float
    train_accuracy: Optional[float] = None
    val_accuracy: Optional[float] = None
    test_accuracy: Optional[float] = None
    test_f1: Optional[float] = None
    peak_memory_gb: Optional[float] = None


class FinancialFinetuner:
    """Fine-tunes models on Financial PhraseBank dataset"""
    
    def __init__(self, output_dir: str = "results/finetuning"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_financial_phrasebank(self):
        """Load Financial PhraseBank dataset"""
        print("Loading Financial PhraseBank dataset...")
        
        try:
            # Try to load from HuggingFace
            dataset = load_dataset("financial_phrasebank", "sentences_50agree")
        except:
            # If not available, create a placeholder
            print("Financial PhraseBank not found on HuggingFace. Using placeholder data.")
            print("Please download the dataset manually and place it in data/financial_phrasebank/")
            
            # Create placeholder dataset structure
            dataset = {
                "train": Dataset.from_dict({
                    "text": [
                        "The company reported strong quarterly earnings.",
                        "Stock prices fell sharply after the announcement.",
                        "Revenue increased by 15% compared to last year.",
                    ],
                    "label": [1, 0, 1],  # 0: negative, 1: positive, 2: neutral
                })
            }
        
        return dataset
    
    def preprocess_dataset(self, dataset, tokenizer, max_length: int = 512):
        """Preprocess dataset for classification"""
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
        )
        return tokenized
    
    def finetune_granite_4(
        self,
        model_name: str = "ibm/granite-4.0-8b-base",
        use_lora: bool = True,
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-4,
    ) -> FinetuningResult:
        """Fine-tune Granite 4.0 model"""
        print(f"\n{'='*60}")
        print(f"Fine-tuning Granite 4.0: {model_name}")
        print(f"{'='*60}")
        
        # Load dataset
        dataset = self.load_financial_phrasebank()
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # For classification, we need to adapt the model
        # This is a simplified version - actual implementation may vary
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        
        # Add classification head (simplified)
        # In practice, you might want to use AutoModelForSequenceClassification
        
        # Apply LoRA if requested
        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
            )
            model = get_peft_model(model, lora_config)
            method = "lora"
        else:
            method = "full"
        
        # Preprocess dataset
        tokenized = self.preprocess_dataset(dataset["train"], tokenizer)
        
        # Split dataset
        split = tokenized.train_test_split(test_size=0.2)
        train_dataset = split["train"]
        val_dataset = split["test"]
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / "granite_4"),
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            fp16=True,
            load_best_model_at_end=True,
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        # Profile training
        profiler = MemoryProfiler()
        profiler.start()
        
        # Train
        train_result = trainer.train()
        
        metrics = profiler.stop()
        
        # Evaluate
        eval_result = trainer.evaluate()
        
        return FinetuningResult(
            model_name="granite_4",
            base_model=model_name,
            dataset="financial_phrasebank",
            method=method,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            final_loss=train_result.training_loss,
            val_accuracy=eval_result.get("eval_accuracy"),
            peak_memory_gb=metrics.peak_gpu_memory,
        )
    
    def finetune_gemma_3(
        self,
        model_name: str = "google/gemma-3-4b-pt",
        use_lora: bool = True,
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-4,
    ) -> FinetuningResult:
        """Fine-tune Gemma 3 model"""
        print(f"\n{'='*60}")
        print(f"Fine-tuning Gemma 3: {model_name}")
        print(f"{'='*60}")
        
        # Similar to Granite4
        dataset = self.load_financial_phrasebank()
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        
        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
            )
            model = get_peft_model(model, lora_config)
            method = "lora"
        else:
            method = "full"
        
        tokenized = self.preprocess_dataset(dataset["train"], tokenizer)
        split = tokenized.train_test_split(test_size=0.2)
        train_dataset = split["train"]
        val_dataset = split["test"]
        
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / "gemma_3"),
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            fp16=True,
            load_best_model_at_end=True,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        profiler = MemoryProfiler()
        profiler.start()
        
        train_result = trainer.train()
        metrics = profiler.stop()
        eval_result = trainer.evaluate()
        
        return FinetuningResult(
            model_name="gemma_3",
            base_model=model_name,
            dataset="financial_phrasebank",
            method=method,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            final_loss=train_result.training_loss,
            val_accuracy=eval_result.get("eval_accuracy"),
            peak_memory_gb=metrics.peak_gpu_memory,
        )
    
    def finetune_qwen_3(
        self,
        model_name: str = "Qwen/Qwen3-7B-Instruct",
        use_lora: bool = True,
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-4,
    ) -> FinetuningResult:
        """Fine-tune Qwen 3 model"""
        print(f"\n{'='*60}")
        print(f"Fine-tuning Qwen 3: {model_name}")
        print(f"{'='*60}")
        
        dataset = self.load_financial_phrasebank()
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        
        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
            )
            model = get_peft_model(model, lora_config)
            method = "lora"
        else:
            method = "full"
        
        tokenized = self.preprocess_dataset(dataset["train"], tokenizer)
        split = tokenized.train_test_split(test_size=0.2)
        train_dataset = split["train"]
        val_dataset = split["test"]
        
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / "qwen_3"),
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            fp16=True,
            load_best_model_at_end=True,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        profiler = MemoryProfiler()
        profiler.start()
        
        train_result = trainer.train()
        metrics = profiler.stop()
        eval_result = trainer.evaluate()
        
        return FinetuningResult(
            model_name="qwen_3",
            base_model=model_name,
            dataset="financial_phrasebank",
            method=method,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            final_loss=train_result.training_loss,
            val_accuracy=eval_result.get("eval_accuracy"),
            peak_memory_gb=metrics.peak_gpu_memory,
        )
    
    def save_results(self, result: FinetuningResult):
        """Save fine-tuning results"""
        filename = f"finetuning_{result.model_name}_{result.method}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(asdict(result), f, indent=2)
        
        print(f"\nResults saved to: {filepath}")
        return filepath


def main():
    """Main fine-tuning execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune models on Financial PhraseBank")
    parser.add_argument("--model", type=str, choices=["granite_4", "gemma_3", "qwen_3"], required=True)
    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--output-dir", type=str, default="results/finetuning")
    
    args = parser.parse_args()
    
    finetuner = FinancialFinetuner(output_dir=args.output_dir)
    
    if args.model == "granite_4":
        result = finetuner.finetune_granite_4(
            use_lora=args.use_lora,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )
    elif args.model == "gemma_3":
        result = finetuner.finetune_gemma_3(
            use_lora=args.use_lora,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )
    elif args.model == "qwen_3":
        result = finetuner.finetune_qwen_3(
            use_lora=args.use_lora,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )
    
    finetuner.save_results(result)
    
    print(f"\n{'='*60}")
    print(f"Fine-tuning Summary")
    print(f"{'='*60}")
    print(f"Model: {result.model_name}")
    print(f"Method: {result.method}")
    print(f"Final loss: {result.final_loss:.4f}")
    if result.val_accuracy:
        print(f"Validation accuracy: {result.val_accuracy:.4f}")
    if result.peak_memory_gb:
        print(f"Peak memory: {result.peak_memory_gb:.2f} GB")


if __name__ == "__main__":
    main()

