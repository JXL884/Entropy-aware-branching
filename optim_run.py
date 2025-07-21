import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set

import pandas as pd
import torch
import torch.multiprocessing as mp
from datasets import load_dataset
from rich import print as rprint
from rich.progress import (BarColumn, Progress, SpinnerColumn, TextColumn,
                           TimeRemainingColumn)
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig

from entropix.config import (DynamicThresholdConfig, MODEL_CONFIG_OVERRIDES,
                            SamplerConfig, ThresholdLevel, Thresholds)
from entropix.model import Model, generate

mp.set_start_method('spawn', force=True)

# --- Configuration and Setup Functions ---
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["llama-8b", "deepseek-8b", "qwen-4b", "qwen-1.7b", "qwen-14b"])
    parser.add_argument("--dataset", type=str, required=True, choices=["aime", "gsm8k", "math500", "l1", "l2"])
    parser.add_argument("--method", type=str, default="both", choices=["standard", "entropix", "both"])
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--force-restart", action="store_true", help="Ignore existing results and restart from scratch")
    return parser.parse_args()

def get_model_config(model_name: str) -> Dict[str, Any]:
    """Retrieves the configuration for a given model name."""
    common_hparams = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0}
    model_paths = {
        "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
        "deepseek-8b": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        "qwen-4b": "Qwen/Qwen3-4B",
        "qwen-1.7b": "Qwen/Qwen3-1.7B",
        "qwen-14b": "Qwen/Qwen3-14B",
    }
    config = {"path": model_paths[model_name]}
    config.update(common_hparams)
    return config

def setup_output_directory(args: argparse.Namespace) -> Path:
    """Creates and returns the output directory path."""
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results_{args.model}_{args.dataset}_{timestamp}")
    output_dir.mkdir(exist_ok=True, parents=True)
    rprint(f"[bold]Results will be saved to:[/bold] {output_dir.resolve()}")
    return output_dir

def validate_gpus():
    """Checks for available GPUs and prints their information."""
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        rprint("[bold red]No GPUs detected! This script requires CUDA GPUs.[/bold red]")
        raise RuntimeError("No CUDA GPUs available")

    rprint(f"[bold green]Detected {num_gpus} GPUs. Work will be distributed across all of them.[/bold green]")
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        rprint(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    return num_gpus

def apply_config_overrides(model: Any, config_name: str, config_overrides: Dict):
    """Applies model-specific configuration overrides for Entropix."""
    if config_name not in config_overrides:
        rprint(f"[yellow]Warning: No config overrides found in MODEL_CONFIG_OVERRIDES for '{config_name}'. "
               f"Using default model config for Entropix.[/yellow]")
        return

    overrides = config_overrides[config_name]
    rprint(f"[bold cyan]Applying Entropix config overrides for '{config_name}': {overrides}[/bold cyan]")
    for attr_name, value in overrides.items():
        setattr(model.config, attr_name, value)


# --- Resume and Checkpointing Functions ---
def is_question_complete(result_path: Path, methods_to_check: List[str]) -> bool:
    """
    Checks if a question has been successfully processed for all required methods.
    
    A question is considered complete if:
    - The JSON file exists
    - For each method to check:
      - The method key exists in the JSON
      - The response field is not empty
      - There's no error field or the error field is None/empty
    """
    if not result_path.exists():
        return False
    
    try:
        with open(result_path, 'r') as f:
            data = json.load(f)
        
        for method in methods_to_check:
            if method not in data:
                return False
            
            method_data = data[method]
            
            # Check if response exists and is not empty
            if 'response' not in method_data or not method_data['response']:
                return False
            
            # Check if there's an error
            if 'error' in method_data and method_data['error']:
                return False
        
        return True
    
    except (json.JSONDecodeError, KeyError, TypeError):
        # If we can't read or parse the file properly, consider it incomplete
        return False

def get_completed_questions(output_dir: Path, methods_to_check: List[str]) -> Set[int]:
    """
    Scans the output directory and returns a set of question IDs that have been
    successfully completed for all required methods.
    """
    completed = set()
    
    # Look for all q_*.json files
    for json_file in output_dir.glob("q_*.json"):
        try:
            # Extract question ID from filename
            question_id = int(json_file.stem.split('_')[1])
            
            if is_question_complete(json_file, methods_to_check):
                completed.add(question_id)
        
        except (ValueError, IndexError):
            # Skip files that don't match the expected pattern
            continue
    
    return completed

def check_resume_status(output_dir: Path, total_questions: int, methods_to_check: List[str]) -> Tuple[Set[int], bool]:
    """
    Checks if there are existing results in the output directory and returns
    the set of completed questions and whether we're resuming.
    """
    if not output_dir.exists():
        return set(), False
    
    # Count existing result files
    existing_files = list(output_dir.glob("q_*.json"))
    if not existing_files:
        return set(), False
    
    completed = get_completed_questions(output_dir, methods_to_check)
    
    if completed:
        rprint(f"[bold yellow]Found existing results in {output_dir}[/bold yellow]")
        rprint(f"[yellow]Completed questions: {len(completed)}/{total_questions}[/yellow]")
        rprint(f"[yellow]Resuming from previous run...[/yellow]")
        return completed, True
    
    return set(), False


# --- Data Loading and Formatting Functions ---
def format_prompt(dataset_name: str, row: pd.Series) -> List[Dict[str, str]]:
    """Formats the prompt based on the dataset type."""
    if dataset_name in ("l1", "l2"):
        system_prompt = (
            "You are an expert financial analyst.\n"
            "You are given questions about various financial topics, from quantitative analysis to portfolio management to ethics of being a chartered financial analyst (CFA).\n"
            "Each question includes 3 potential answers, A B and C, one of which is correct (or in some cases, more correct than the others).\n"
            "Indicate the correct answer: A, B, or C."
        )
        user_content = f"{row.prompt_question}\nA. {row.choice_a}\nB. {row.choice_b}\nC. {row.choice_c}"
        if dataset_name == 'l2' and 'case' in row and pd.notna(row['case']):
            user_content = f"{row.case}\n\n{user_content}"
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]

    elif dataset_name in ("aime", "gsm8k", "math500"):
        return [{"role": "user", "content": row.prompt_question}]

    else:
        raise ValueError(f"Unknown dataset for prompt formatting: {dataset_name}")

def load_and_prepare_dataset(args: argparse.Namespace, completed_questions: Set[int] = None) -> pd.DataFrame:
    """Loads a dataset from Hugging Face Hub or local files and prepares it for benchmarking."""
    rprint(f"[bold blue]Loading dataset: {args.dataset}[/bold blue]")

    if args.dataset in ("l1", "l2"):
        script_dir = Path(__file__).parent
        file_path = script_dir / "data" / f"{args.dataset}_exams.json"
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found at {file_path}")
        df = pd.read_json(file_path)
        df['question_id'] = df.index
        df = df.rename(columns={"question": "prompt_question", "answer": "expected_answer"})
    elif args.dataset == "aime":
        dataset = load_dataset("AI-MO/aimo-validation-aime", split="train")
        df = pd.DataFrame(dataset).rename(columns={"problem": "prompt_question", "answer": "expected_answer"})
        df['question_id'] = df.index
    elif args.dataset == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        df = pd.DataFrame(dataset).rename(columns={"question": "prompt_question", "answer": "expected_answer"})
        df['question_id'] = df.index
    elif args.dataset == "math500":
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        df = pd.DataFrame(dataset).rename(columns={"problem": "prompt_question", "answer": "expected_answer"})
        df['question_id'] = df.index
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    limit = args.limit if args.limit > 0 else len(df)
    df = df.iloc[:limit].copy()

    # Filter out completed questions if resuming
    if completed_questions:
        df = df[~df['question_id'].isin(completed_questions)].copy()
        rprint(f"[green]Filtered out {len(completed_questions)} completed questions.[/green]")

    df['full_prompt'] = df.apply(lambda row: format_prompt(args.dataset, row), axis=1)

    rprint(f"[green]Loaded and pre-formatted {len(df)} questions for processing.[/green]")
    return df

# --- Model Loading and Generation Functions ---
def load_model_and_tokenizer(model_path: str, gpu_id: int) -> Tuple[Any, Any]:
    """Loads a quantized model and its tokenizer onto a specific GPU."""
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": gpu_id},
        torch_dtype=torch.bfloat16,
        quantization_config=quant_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def _prepare_prompts_from_chat_templates(batch: List[Dict], tokenizer: Any) -> List[str]:
    """Helper to prepare prompts from pre-formatted chat templates."""
    prompts = []
    for item in batch:
        messages = item['full_prompt']
        try:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
    return prompts

def process_batch_standard(batch: List[Dict], model: Any, tokenizer: Any, model_config: Dict, max_new_tokens: int) -> List[Dict]:
    """Processes a batch of questions using standard generation."""
    prompts = _prepare_prompts_from_chat_templates(batch, tokenizer)
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(model.device)

    start_time = time.time()
    outputs = model.generate(
        **inputs,
        temperature=model_config["temperature"],
        top_p=model_config["top_p"],
        top_k=model_config["top_k"],
        do_sample=True,
        max_new_tokens=max_new_tokens,
    )
    gen_time = time.time() - start_time

    results = []
    for i in range(len(batch)):
        input_len = torch.sum(inputs.attention_mask[i]).item()
        gen_tokens = outputs[i][input_len:]
        results.append({
            "response": tokenizer.decode(gen_tokens, skip_special_tokens=True),
            "prompt": prompts[i],
            "generation_time": gen_time / len(batch),
            "total_tokens": len(gen_tokens),
            "method": "standard"
        })
    return results

def process_batch_entropix(batch: List[Dict], entropix_model: Model, sampler_cfg: SamplerConfig, max_tokens: int) -> List[Dict]:
    """Processes a batch of questions using Entropix generation (one by one)."""
    results = []
    for item in batch:
        messages = item['full_prompt']
        try:
            start_time = time.time()
            gen_data = generate(messages, entropix_model, sampler_cfg, stream_output=False, max_tokens=max_tokens)
            generation_time = time.time() - start_time
            results.append({
                "response": gen_data.response,
                "prompt": messages,
                "generation_time": generation_time,
                "total_tokens": len(gen_data.tokens),
                "method": "entropix"
            })
        except Exception as e:
            rprint(f"[red]Error in entropix generation for q{item['question_id']}: {e}[/red]")
            results.append({
                "response": f"ERROR: {e}",
                "prompt": messages,
                "error": str(e),
                "method": "entropix",
                "generation_time": 0,
                "total_tokens": 0
            })
    return results

# --- Main Worker and Orchestration ---
def run_benchmark_on_gpu(task: Dict) -> int:
    """
    A unified worker function to run on a single GPU. It processes its assigned
    questions and saves the results to disk incrementally after each batch.
    """
    gpu_id = task['gpu_id']
    questions = task['questions']
    methods_to_run = task['methods_to_run']
    model_config = task['model_config']
    args = task['args']
    output_dir = task['output_dir']

    torch.cuda.set_device(gpu_id)
    model, tokenizer = load_model_and_tokenizer(model_config["path"], gpu_id)

    entropix_model, sampler_cfg = None, None
    if 'entropix' in methods_to_run:
        config_name = model_config["path"].split("/")[-1]
        apply_config_overrides(model, config_name, MODEL_CONFIG_OVERRIDES)
        entropix_model = Model(model, model.config, tokenizer)
        sampler_cfg = SamplerConfig(
            temperature=model_config["temperature"], top_p=model_config["top_p"],
            top_k=model_config["top_k"], min_p=model_config["min_p"],
            thresholds=Thresholds(
                logit_entropy=ThresholdLevel(low=1.2, medium=3, high=1),
                logit_varentropy=ThresholdLevel(low=3, medium=6.5, high=2),
                dynamic=DynamicThresholdConfig(strategy="static"),
            )
        )

    processed_count = 0
    batch_size = args.batch_size
    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i : i + batch_size]

        # Process each question in the batch
        for question_data in batch_questions:
            output_path = output_dir / f"q_{question_data['question_id']}.json"
            
            # Load existing data if file exists
            existing_data = {}
            if output_path.exists():
                try:
                    with open(output_path, 'r') as f:
                        existing_data = json.load(f)
                except (json.JSONDecodeError, IOError):
                    existing_data = {}
            
            # Update question data with existing results
            question_data.update(existing_data)

        # Process standard method if needed
        if 'standard' in methods_to_run:
            # Filter questions that need standard processing
            questions_needing_standard = [
                q for q in batch_questions 
                if not is_question_complete(output_dir / f"q_{q['question_id']}.json", ['standard'])
            ]
            
            if questions_needing_standard:
                standard_results = process_batch_standard(
                    questions_needing_standard, model, tokenizer, 
                    model_config, args.max_new_tokens
                )
                for j, res in enumerate(standard_results):
                    questions_needing_standard[j]['standard'] = res

        # Process entropix method if needed
        if 'entropix' in methods_to_run:
            # Filter questions that need entropix processing
            questions_needing_entropix = [
                q for q in batch_questions 
                if not is_question_complete(output_dir / f"q_{q['question_id']}.json", ['entropix'])
            ]
            
            if questions_needing_entropix:
                entropix_results = process_batch_entropix(
                    questions_needing_entropix, entropix_model, 
                    sampler_cfg, args.max_new_tokens
                )
                for j, res in enumerate(entropix_results):
                    questions_needing_entropix[j]['entropix'] = res

        # Save all results
        for result_data in batch_questions:
            output_path = output_dir / f"q_{result_data['question_id']}.json"
            with open(output_path, "w") as f:
                json.dump(result_data, f, indent=2)
        
        processed_count += len(batch_questions)

    del model, tokenizer, entropix_model
    torch.cuda.empty_cache()

    return processed_count

def main():
    args = parse_arguments()
    model_config = get_model_config(args.model)
    output_dir = setup_output_directory(args)
    num_gpus = validate_gpus()
    
    # Determine which methods we'll be running
    methods_to_run = ['standard', 'entropix'] if args.method == 'both' else [args.method]
    
    # Check for existing results and resume capability
    completed_questions = set()
    if not args.force_restart:
        # First load dataset to get total count
        temp_df = load_and_prepare_dataset(args, completed_questions=set())
        total_questions = len(temp_df)
        
        completed_questions, is_resuming = check_resume_status(
            output_dir, total_questions, methods_to_run
        )
    
    # Load dataset, filtering out completed questions
    df = load_and_prepare_dataset(args, completed_questions)

    if df.empty:
        if completed_questions:
            rprint("[bold green]All questions have been processed! Nothing left to do.[/bold green]")
        else:
            rprint("[yellow]No questions to process based on the provided arguments. Exiting.[/yellow]")
        return

    rprint(f"[bold cyan]Starting benchmark for method(s): {', '.join(methods_to_run)}[/bold cyan]")
    rprint(f"[cyan]Distributing {len(df)} questions across {num_gpus} GPUs.[/cyan]")

    all_questions = df.to_dict('records')
    tasks = []
    for i in range(num_gpus):
        gpu_questions = all_questions[i::num_gpus]
        if gpu_questions:
            tasks.append({
                'gpu_id': i,
                'questions': gpu_questions,
                'methods_to_run': methods_to_run,
                'model_config': model_config,
                'args': args,
                'output_dir': output_dir
            })

    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        future_to_gpu = {
            executor.submit(run_benchmark_on_gpu, task): task['gpu_id']
            for task in tasks
        }

        progress_columns = [
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        ]
        with Progress(*progress_columns) as progress:
            # Show total progress including already completed questions
            total_to_process = len(df) + len(completed_questions)
            p_task = progress.add_task(
                "Processing questions...", 
                total=total_to_process,
                completed=len(completed_questions)
            )
            
            for future in as_completed(future_to_gpu):
                try:
                    num_processed = future.result()
                    progress.update(p_task, advance=num_processed)
                except Exception as e:
                    gpu_id = future_to_gpu[future]
                    rprint(f"[bold red]An error occurred in the worker for GPU {gpu_id}: {e}[/bold red]")
                    import traceback
                    traceback.print_exc()

    rprint(f"\n[bold green]Benchmark Complete![/bold green]")
    
    # Final summary
    final_completed = get_completed_questions(output_dir, methods_to_run)
    rprint(f"[bold green]Total questions processed: {len(final_completed)}[/bold green]")
    
    if len(final_completed) < total_to_process:
        remaining = total_to_process - len(final_completed)
        rprint(f"[yellow]Warning: {remaining} questions may have failed or encountered errors.[/yellow]")
        rprint(f"[yellow]You can run the script again to retry failed questions.[/yellow]")

if __name__ == "__main__":
    main()