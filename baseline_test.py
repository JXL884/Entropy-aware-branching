#!/usr/bin/env python3
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from datasets import load_dataset
from rich import print as rprint
from rich.progress import Progress, SpinnerColumn, TextColumn
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


class BaselineTester:
    def __init__(self):
        self.args = self._parse_args()
        self.model_config = self._get_model_config()
        self.df = self._load_dataset_data()
        self.output_dir = self._setup_output_dir()
        self.results: List[Dict] = []
        self._setup_models()

    def _parse_args(self):
        parser = argparse.ArgumentParser(
            description="Baseline testing: Transformers vs Entropix sampling"
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
            "--stream",
            action="store_true",
            help="Enable real-time streaming and verbose output.",
        )
        parser.add_argument(
            "--num-workers",
            type=int,
            default=4,
            help="Number of parallel workers for processing questions.",
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

    def _setup_models(self):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        rprint(f"[bold blue]Loading model: {self.model_config['path']}[/bold blue]")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_config["path"],
            device_map="auto",
            torch_dtype="auto",
            quantization_config=quantization_config,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config["path"])
        self._apply_config_overrides()
        self.entropix_model = Model(self.base_model, self.base_model.config, self.tokenizer)
        self.sampler_cfg = SamplerConfig(
            temperature=self.model_config["temperature"],
            top_p=self.model_config["top_p"],
            top_k=self.model_config["top_k"],
            min_p=self.model_config["min_p"],
            thresholds=Thresholds(
                logit_entropy=ThresholdLevel(low=1.2, medium=3, high=1),
                logit_varentropy=ThresholdLevel(low=3, medium=6.5, high=2),
                dynamic=DynamicThresholdConfig(strategy="static"),
            ),
        )

    def _apply_config_overrides(self):
        model_name = self.model_config["path"].split("/")[-1]
        if model_name in MODEL_CONFIG_OVERRIDES:
            rprint(f"[green]Applied config overrides for {model_name}[/green]")
            config_overrides = MODEL_CONFIG_OVERRIDES[model_name]
            for attr_name, value in config_overrides.items():
                setattr(self.base_model.config, attr_name, value)
        else:
            rprint(f"[yellow]No config overrides found for {model_name}[/yellow]")

    def _format_prompt(self, question_data) -> List[Dict[str, str]]:
        return [{"role": "user", "content": question_data.question}]

    def _generate_standard_transformers(
        self, messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Generate response using standard Transformers generation."""
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        else:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.base_model.device)
        generation_kwargs = {
            "input_ids": inputs["input_ids"],
            "temperature": self.model_config["temperature"],
            "top_p": self.model_config["top_p"],
            "top_k": self.model_config["top_k"],
            "do_sample": True,
            "max_new_tokens": 32768,
        }
        if self.args.stream:
            streamer = TextStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )
            generation_kwargs["streamer"] = streamer

        start_time = time.time()
        outputs = self.base_model.generate(**generation_kwargs)
        generation_time = time.time() - start_time
        if self.args.stream:
            print()

        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return {
            "response": generated_text,
            "generation_time": generation_time,
            "total_tokens": len(outputs[0]) - len(inputs["input_ids"][0]),
        }

    def _generate_entropix(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Generate response using Entropix sampling."""
        start_time = time.time()
        gen_data = generate(
            messages,
            self.entropix_model,
            self.sampler_cfg,
            stream_output=self.args.stream,
            enable_thinking=False,
            enable_uncertainty_detection=False,
            enable_insertion=False,
            enable_branching=False,
            max_tokens=32768,
        )
        generation_time = time.time() - start_time
        if self.args.stream:
            print()
        return {
            "response": gen_data.response,
            "generation_time": generation_time,
            "total_tokens": len(gen_data.tokens),
            "gen_data": gen_data.to_dict(),
        }

    def _process_question(self, idx: int, question_data):
        messages = self._format_prompt(question_data)
        if self.args.stream:
            rprint(f"\n[bold cyan]Question {idx + 1}:[/bold cyan]")
            rprint(f"[yellow]{messages[-1]['content'][:200]}...[/yellow]")

        question_result = {
            "question_id": idx,
            "question": messages[-1]["content"],
            "expected_answer": question_data.answer,
            "dataset": self.args.dataset,
            "model": self.args.model,
        }

        # Test both generation methods
        question_result["standard"] = self._run_generation(
            "Standard Transformers", "green", self._generate_standard_transformers, messages
        )
        question_result["entropix"] = self._run_generation(
            "Entropix Sampling", "blue", self._generate_entropix, messages
        )

        with open(self.output_dir / f"q{idx + 1}.json", "w") as f:
            json.dump(question_result, f, indent=2)
        return question_result

    def _run_generation(self, name, color, generator_func, messages):
        if self.args.stream:
            rprint(f"[bold {color}]Testing {name}...[/bold {color}]")
            rprint("[dim]" + "─" * 50 + "[/dim]")

        try:
            result = generator_func(messages)
            if self.args.stream:
                rprint(
                    f"[bold {color}]{name} complete in {result['generation_time']:.2f}s[/bold {color}]"
                )
                rprint("[dim]" + "─" * 50 + "[/dim]")
                rprint(
                    f"[{color}]{name.split(' ')[0]} Response:[/{color}] {result['response'][:100]}..."
                )
            return {**result, "method": name}
        except Exception as e:
            rprint(f"[red]Error in {name} generation: {e}[/red]")
            return {"error": str(e)}

    def _calculate_and_save_summary(self):
        summary = {
            "model": self.args.model,
            "dataset": self.args.dataset,
            "total_questions": len(self.results),
        }
        stats = {
            "standard": {"times": [], "tokens": []},
            "entropix": {"times": [], "tokens": []},
        }

        for result in self.results:
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
            else:
                summary[f"{method}_avg_time"] = 0
                summary[f"{method}_avg_tokens"] = 0

        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        rprint(f"\n[bold green]Testing Complete![/bold green]")
        rprint(f"[bold]Results saved to:[/bold] {self.output_dir}")
        rprint(f"\n[bold]Summary:[/bold]")
        rprint(f"Model: {self.args.model}")
        rprint(f"Dataset: {self.args.dataset}")
        rprint(f"Total Questions: {summary['total_questions']}")
        rprint("\n[bold]Performance:[/bold]")
        rprint(f"Standard Avg Time: {summary.get('standard_avg_time', 0):.2f}s")
        rprint(f"Entropix Avg Time: {summary.get('entropix_avg_time', 0):.2f}s")
        rprint(f"Standard Avg Tokens: {summary.get('standard_avg_tokens', 0):.1f}")
        rprint(f"Entropix Avg Tokens: {summary.get('entropix_avg_tokens', 0):.1f}")

    def run(self):
        if self.args.stream:
            rprint(
                f"[bold yellow]Streaming enabled:[/bold yellow] Tokens will be displayed in real-time during generation"
            )
            rprint(
                f"[dim]Note: Streaming shows colored tokens based on sampler states[/dim]\n"
            )
            num_workers = 1
            if self.args.num_workers > 1:
                rprint(
                    f"[yellow]Streaming is enabled, so processing is forced to be sequential (1 worker).[/yellow]"
                )
        else:
            gpu_count = torch.cuda.device_count()
            if gpu_count > 0:
                # Set a sensible max of 4 workers per GPU to avoid bottlenecks
                max_workers = gpu_count * 4
                if self.args.num_workers > max_workers:
                    rprint(f"[yellow]Warning: Number of workers capped at {max_workers} ({gpu_count} GPUs x 4) to prevent GPU bottleneck.[/yellow]")
                    num_workers = max_workers
                else:
                    num_workers = self.args.num_workers
            else:
                rprint("[yellow]Warning: No GPU detected. Running on CPU with 1 worker.[/yellow]")
                num_workers = 1

            if num_workers > 1:
                rprint(f"[bold blue]Running with {num_workers} parallel workers.[/bold blue]")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(self._process_question, idx, question_data): (
                    idx,
                    question_data,
                )
                for idx, (_, question_data) in enumerate(self.df.iterrows())
            }

            progress = None
            task = None
            if not self.args.stream:
                progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    transient=False,
                )
                progress.start()
                task = progress.add_task(
                    "Processing questions...", total=len(futures)
                )

            try:
                for future in as_completed(futures):
                    result = future.result()
                    self.results.append(result)
                    if progress and task:
                        progress.advance(task)
            finally:
                if progress:
                    progress.stop()

        self.results.sort(key=lambda r: r["question_id"])
        self._calculate_and_save_summary()


if __name__ == "__main__":
    tester = BaselineTester()
    tester.run()
