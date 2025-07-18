#!/usr/bin/env python3
import argparse
import json
import os
import time
import torch
import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

import pandas as pd
from datasets import load_dataset
from rich import print as rprint
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import TextStreamer
from transformers.utils.quantization_config import BitsAndBytesConfig

from entropix.config import (
    DynamicThresholdConfig,
    MODEL_CONFIG_OVERRIDES,
    SamplerConfig,
    ThresholdLevel,
    Thresholds,
)
from entropix.model import Model, generate


# Set up multiprocessing for CUDA
mp.set_start_method('spawn', force=True)

# Check if we should disable compilation globally
DISABLE_COMPILATION = os.environ.get('DISABLE_TORCH_COMPILE', '0') == '1'
if DISABLE_COMPILATION:
    rprint("[yellow]Model compilation disabled via DISABLE_TORCH_COMPILE=1[/yellow]")


class OptimizedBaselineTester:
    def __init__(self):
        self.args = self._parse_args()
        self.model_config = self._get_model_config()
        self.df = self._load_dataset_data()
        self.output_dir = self._setup_output_dir()
        self.num_gpus = torch.cuda.device_count()
        
        # Validate GPU availability
        if self.num_gpus == 0:
            rprint("[red]No GPUs detected! This script requires CUDA GPUs.[/red]")
            raise RuntimeError("No CUDA GPUs available")
        
        rprint(f"[green]Detected {self.num_gpus} GPUs[/green]")
        for i in range(self.num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            rprint(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")

    def _parse_args(self):
        parser = argparse.ArgumentParser(
            description="Optimized Multi-GPU Baseline testing: Transformers vs Entropix sampling"
        )
        parser.add_argument(
            "--model",
            type=str,
            required=True,
            choices=["llama-8b", "deepseek-8b", "qwen-4b"],
        )
        parser.add_argument(
            "--dataset", type=str, required=True, choices=["aime", "gsm8k", "math500"]
        )
        parser.add_argument("--limit", type=int, default=10)
        parser.add_argument("--start", type=int, default=0)
        parser.add_argument("--output-dir", type=str, default=None)
        parser.add_argument(
            "--batch-size",
            type=int,
            default=4,
            help="Batch size for processing multiple questions at once"
        )
        parser.add_argument(
            "--use-both-gpus",
            action="store_true",
            help="Use both GPUs - one for standard, one for entropix"
        )
        parser.add_argument(
            "--compile-model",
            action="store_true",
            help="Use torch.compile for faster inference (requires PyTorch 2.0+)"
        )
        parser.add_argument(
            "--no-cuda-graphs",
            action="store_true",
            help="Disable CUDA graphs in torch.compile (fixes some compatibility issues)"
        )
        return parser.parse_args()

    def _get_model_config(self) -> Dict[str, Any]:
        """Get model configuration based on model name."""
        model_configs = {
            "llama-8b": {
                "path": "meta-llama/Llama-3.1-8B-Instruct",
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "min_p": 0.0,
            },
            "deepseek-8b": {
                "path": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "min_p": 0.0,
            },
            "qwen-4b": {
                "path": "Qwen/Qwen3-4B",
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "min_p": 0.0,
            },
        }
        return model_configs[self.args.model]

    def _load_dataset_data(self) -> pd.DataFrame:
        """Load dataset based on the specified name."""
        rprint(f"[bold blue]Loading dataset: {self.args.dataset}[/bold blue]")
        if self.args.dataset == "aime":
            dataset = load_dataset("AI-MO/aimo-validation-aime", split="train")
            df = pd.DataFrame(dataset)
            df = df.rename(columns={"problem": "question"})
        elif self.args.dataset == "gsm8k":
            dataset = load_dataset("openai/gsm8k", "main", split="test")
            df = pd.DataFrame(dataset)
        elif self.args.dataset == "math500":
            dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
            df = pd.DataFrame(dataset)
            df = df.rename(columns={"problem": "question"})
        else:
            raise ValueError(f"Unknown dataset: {self.args.dataset}")

        end_idx = self.args.start + self.args.limit if self.args.limit else len(df)
        df = df.iloc[self.args.start : end_idx]
        rprint(f"[green]Loaded {len(df)} questions[/green]")
        return df

    def _setup_output_dir(self) -> Path:
        if self.args.output_dir:
            output_dir = Path(self.args.output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(
                f"results_{self.args.model}_{self.args.dataset}_{timestamp}"
            )
        output_dir.mkdir(exist_ok=True)
        return output_dir

    def run(self):
        """Main entry point for optimized processing."""
        if self.args.use_both_gpus and self.num_gpus >= 2:
            self._run_dual_gpu_pipeline()
        else:
            self._run_multi_process()

    def _run_dual_gpu_pipeline(self):
        """Run with one GPU for standard generation and one for entropix."""
        rprint("[bold cyan]Running in dual-GPU pipeline mode[/bold cyan]")
        
        # Prepare all questions
        all_questions = [(idx, row) for idx, row in self.df.iterrows()]
        
        # Process using workers that handle model loading once
        with ProcessPoolExecutor(max_workers=2) as executor:
            # Submit both methods to run on different GPUs
            standard_future = executor.submit(
                process_all_questions_gpu,
                all_questions,
                self.model_config,
                0,  # GPU 0 for standard
                "standard",
                self.args,
                self.output_dir
            )
            
            entropix_future = executor.submit(
                process_all_questions_gpu,
                all_questions,
                self.model_config,
                1,  # GPU 1 for entropix
                "entropix",
                self.args,
                self.output_dir
            )
            
            # Collect results
            standard_results = standard_future.result()
            entropix_results = entropix_future.result()
        
        # Merge results
        results = []
        for i, (idx, row) in enumerate(all_questions):
            result = {
                "question_id": idx,
                "question": row["question"],
                "expected_answer": row["answer"],
                "dataset": self.args.dataset,
                "model": self.args.model,
                "standard": standard_results[i],
                "entropix": entropix_results[i]
            }
            results.append(result)
            
            # Save individual result
            with open(self.output_dir / f"q{idx + 1}.json", "w") as f:
                json.dump(result, f, indent=2)
        
        self._save_summary(results)

    def _run_multi_process(self):
        """Run using multiple processes across available GPUs."""
        rprint("[bold cyan]Running in multi-process mode[/bold cyan]")
        
        # Split dataset across GPUs
        gpu_assignments = []
        for i, (idx, row) in enumerate(self.df.iterrows()):
            gpu_id = i % self.num_gpus
            gpu_assignments.append((idx, row, gpu_id))
        
        # Group by GPU
        gpu_batches = {}
        for idx, row, gpu_id in gpu_assignments:
            if gpu_id not in gpu_batches:
                gpu_batches[gpu_id] = []
            gpu_batches[gpu_id].append((idx, row))
        
        results = []
        
        with ProcessPoolExecutor(max_workers=self.num_gpus) as executor:
            futures = {}
            
            # Submit tasks for each GPU
            for gpu_id, questions in gpu_batches.items():
                future = executor.submit(
                    process_all_questions_gpu_both_methods,
                    questions,
                    self.model_config,
                    gpu_id,
                    self.args,
                    self.output_dir
                )
                futures[future] = gpu_id
            
            # Collect results with progress bar
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task("Processing questions...", total=len(self.df))
                
                for future in as_completed(futures):
                    gpu_id = futures[future]
                    gpu_results = future.result()
                    results.extend(gpu_results)
                    progress.update(task, advance=len(gpu_results))
        
        # Sort results by question ID
        results.sort(key=lambda x: x["question_id"])
        self._save_summary(results)

    def _save_summary(self, results):
        """Calculate and save summary statistics."""
        summary = {
            "model": self.args.model,
            "dataset": self.args.dataset,
            "total_questions": len(results),
            "batch_size": self.args.batch_size,
            "num_gpus_used": self.num_gpus,
        }
        
        stats = {
            "standard": {"times": [], "tokens": []},
            "entropix": {"times": [], "tokens": []},
        }

        for result in results:
            for method in ["standard", "entropix"]:
                if method in result and "error" not in result[method]:
                    stats[method]["times"].append(result[method]["generation_time"])
                    stats[method]["tokens"].append(result[method]["total_tokens"])

        for method in ["standard", "entropix"]:
            if stats[method]["times"]:
                summary[f"{method}_avg_time"] = sum(stats[method]["times"]) / len(
                    stats[method]["times"]
                )
                summary[f"{method}_avg_tokens"] = sum(stats[method]["tokens"]) / len(
                    stats[method]["tokens"]
                )
                summary[f"{method}_total_time"] = sum(stats[method]["times"])
            else:
                summary[f"{method}_avg_time"] = 0
                summary[f"{method}_avg_tokens"] = 0
                summary[f"{method}_total_time"] = 0

        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Save all results
        with open(self.output_dir / "all_results.json", "w") as f:
            json.dump(results, f, indent=2)

        rprint(f"\n[bold green]Testing Complete![/bold green]")
        rprint(f"[bold]Results saved to:[/bold] {self.output_dir}")
        rprint(f"\n[bold]Summary:[/bold]")
        rprint(f"Model: {self.args.model}")
        rprint(f"Dataset: {self.args.dataset}")
        rprint(f"Total Questions: {summary['total_questions']}")
        rprint(f"Batch Size: {summary['batch_size']}")
        rprint(f"GPUs Used: {summary['num_gpus_used']}")
        rprint("\n[bold]Performance:[/bold]")
        rprint(f"Standard - Avg Time: {summary.get('standard_avg_time', 0):.2f}s, Total: {summary.get('standard_total_time', 0):.2f}s")
        rprint(f"Entropix - Avg Time: {summary.get('entropix_avg_time', 0):.2f}s, Total: {summary.get('entropix_total_time', 0):.2f}s")
        rprint(f"Standard Avg Tokens: {summary.get('standard_avg_tokens', 0):.1f}")
        rprint(f"Entropix Avg Tokens: {summary.get('entropix_avg_tokens', 0):.1f}")


def setup_model_compilation(model, args):
    """Setup model compilation with proper configuration."""
    if DISABLE_COMPILATION:
        return model
        
    if not args.compile_model or not hasattr(torch, 'compile'):
        return model
    
    # Check if we should disable CUDA graphs
    if args.no_cuda_graphs or args.model == "qwen-4b":
        # Qwen models often have issues with CUDA graphs
        return torch.compile(
            model, 
            mode="reduce-overhead", 
            fullgraph=False, 
        )
    else:
        try:
            # Try with CUDA graphs first
            return torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            print(f"[yellow]Warning: Compilation with CUDA graphs failed, retrying without: {e}[/yellow]")
            # Fallback to no CUDA graphs
            return torch.compile(
                model, 
                mode="reduce-overhead", 
                fullgraph=False, 
                disable_cuda_graphs=True
            )


def load_model_and_tokenizer(model_config, gpu_id, args):
    """Load model and tokenizer on specific GPU."""
    torch.cuda.set_device(gpu_id)
    
    # Load model
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_config["path"],
        device_map={"": gpu_id},
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_config["path"])
    
    # Fix padding side for decoder-only models
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Apply config overrides
    model_name = model_config["path"].split("/")[-1]
    if model_name in MODEL_CONFIG_OVERRIDES:
        config_overrides = MODEL_CONFIG_OVERRIDES[model_name]
        for attr_name, value in config_overrides.items():
            setattr(model.config, attr_name, value)
    
    # Compile model if requested
    model = setup_model_compilation(model, args)
    
    return model, tokenizer


def process_all_questions_gpu(questions, model_config, gpu_id, method, args, output_dir):
    """Process all questions on a specific GPU using specified method."""
    torch.cuda.set_device(gpu_id)
    
    # Load model once
    model, tokenizer = load_model_and_tokenizer(model_config, gpu_id, args)
    
    # Setup entropix if needed
    entropix_model = None
    sampler_cfg = None
    if method == "entropix":
        entropix_model = Model(model, model.config, tokenizer)
        sampler_cfg = SamplerConfig(
            temperature=model_config["temperature"],
            top_p=model_config["top_p"],
            top_k=model_config["top_k"],
            min_p=model_config["min_p"],
            thresholds=Thresholds(
                logit_entropy=ThresholdLevel(low=1.2, medium=3, high=1),
                logit_varentropy=ThresholdLevel(low=3, medium=6.5, high=2),
                dynamic=DynamicThresholdConfig(strategy="static"),
            ),
        )
    
    results = []
    
    # Process in batches
    batch_size = args.batch_size
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        
        if method == "standard":
            batch_results = process_batch_standard(batch, model, tokenizer, model_config)
        else:  # entropix
            batch_results = process_batch_entropix(batch, entropix_model, sampler_cfg)
        
        results.extend(batch_results)
    
    # Clean up
    del model
    if entropix_model:
        del entropix_model
    torch.cuda.empty_cache()
    
    return results


def process_all_questions_gpu_both_methods(questions, model_config, gpu_id, args, output_dir):
    """Process all questions on a specific GPU using both methods."""
    torch.cuda.set_device(gpu_id)
    
    # Load model once
    model, tokenizer = load_model_and_tokenizer(model_config, gpu_id, args)
    
    # Setup entropix
    entropix_model = Model(model, model.config, tokenizer)
    sampler_cfg = SamplerConfig(
        temperature=model_config["temperature"],
        top_p=model_config["top_p"],
        top_k=model_config["top_k"],
        min_p=model_config["min_p"],
        thresholds=Thresholds(
            logit_entropy=ThresholdLevel(low=1.2, medium=3, high=1),
            logit_varentropy=ThresholdLevel(low=3, medium=6.5, high=2),
            dynamic=DynamicThresholdConfig(strategy="static"),
        ),
    )
    
    results = []
    
    # Process in batches
    batch_size = args.batch_size
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        
        # Standard generation
        standard_results = process_batch_standard(batch, model, tokenizer, model_config)
        
        # Entropix generation
        entropix_results = process_batch_entropix(batch, entropix_model, sampler_cfg)
        
        # Combine results
        for j, (idx, row) in enumerate(batch):
            result = {
                "question_id": idx,
                "question": row["question"],
                "expected_answer": row["answer"],
                "dataset": args.dataset,
                "model": args.model,
                "standard": standard_results[j],
                "entropix": entropix_results[j]
            }
            results.append(result)
            
            # Save individual result
            with open(output_dir / f"q{idx + 1}.json", "w") as f:
                json.dump(result, f, indent=2)
    
    # Clean up
    del model
    del entropix_model
    torch.cuda.empty_cache()
    
    return results


def process_batch_standard(batch, model, tokenizer, model_config):
    """Process a batch using standard transformers generation."""
    results = []
    
    # Prepare all prompts
    messages_list = []
    for idx, row in batch:
        messages = [{"role": "user", "content": row["question"]}]
        messages_list.append(messages)
    
    # Tokenize all at once
    prompts = []
    for messages in messages_list:
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, 
                    enable_thinking=False
                )
            except TypeError:
                # Fallback if enable_thinking is not supported
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        else:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        prompts.append(prompt)
    
    # Batch tokenization
    inputs = tokenizer(
        prompts, 
        return_tensors="pt", 
        padding=True,
        truncation=True,
        max_length=4096
    ).to(model.device)
    
    # Generate for entire batch
    with torch.no_grad():
        start_time = time.time()
        outputs = model.generate(
            **inputs,
            temperature=model_config["temperature"],
            top_p=model_config["top_p"],
            top_k=model_config["top_k"],
            do_sample=True,
            max_new_tokens=32768,
            pad_token_id=tokenizer.pad_token_id,
        )
        generation_time = time.time() - start_time
    
    # Decode outputs
    for i, (idx, row) in enumerate(batch):
        input_length = inputs["input_ids"][i].shape[0]
        generated_tokens = outputs[i][input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        results.append({
            "response": generated_text,
            "generation_time": generation_time / len(batch),  # Average time per question
            "total_tokens": len(generated_tokens),
            "method": "standard"
        })
    
    return results


def process_batch_entropix(batch, entropix_model, sampler_cfg):
    """Process a batch using entropix generation."""
    results = []
    
    # Process each question individually (entropix doesn't support batching yet)
    for idx, row in batch:
        messages = [{"role": "user", "content": row["question"]}]
        
        try:
            # Mark CUDA graph step if available (PyTorch 2.1+)
            if hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
                torch.compiler.cudagraph_mark_step_begin()
            
            start_time = time.time()
            gen_data = generate(
                messages,
                entropix_model,
                sampler_cfg,
                stream_output=False,
                enable_thinking=False,
                enable_uncertainty_detection=False,
                enable_insertion=False,
                enable_branching=False,
                max_tokens=32768,
            )
            generation_time = time.time() - start_time
            
            results.append({
                "response": gen_data.response,
                "generation_time": generation_time,
                "total_tokens": len(gen_data.tokens),
                "gen_data": gen_data.to_dict(),
                "method": "entropix"
            })
        except RuntimeError as e:
            if "CUDAGraphs" in str(e):
                # If CUDA graph error, try without compilation
                print(f"[yellow]CUDA graph error detected: {str(e)}[/yellow]")
                results.append({
                    "error": f"CUDA graph error: {str(e)}",
                    "method": "entropix",
                    "generation_time": 0,
                    "total_tokens": 0
                })
            else:
                raise
        except Exception as e:
            print(f"[red]Error in entropix generation: {str(e)}[/red]")
            results.append({
                "error": str(e),
                "method": "entropix",
                "generation_time": 0,
                "total_tokens": 0
            })
    
    return results


if __name__ == "__main__":
    tester = OptimizedBaselineTester()
    tester.run()