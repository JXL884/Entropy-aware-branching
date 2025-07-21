import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dspy
from rich.console import Console
from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                           SpinnerColumn, TextColumn, TimeRemainingColumn)
from rich.table import Table

from math_verify import parse, verify
from math_verify.parser import (ExprExtractionConfig, LatexExtractionConfig,
                                StringExtractionConfig)

# --- dspy/LLM Signatures ---
class GradeMathResponse(dspy.Signature):
    """
    You are an expert mathematics grader.
    You will be given a math problem, the known correct answer, and a student's response.
    Your task is to determine if the student's response is correct and provide detailed feedback.
    For numerical answers, check if the final answer matches the expected answer.
    """
    question: str = dspy.InputField(description="The math problem")
    correct_answer: str = dspy.InputField(description="The correct final answer")
    student_response: str = dspy.InputField(description="The student's response to grade (may be a truncated final part of their full response)")

    extracted_answer: str = dspy.OutputField(description="The final answer from the student's response")
    feedback: str = dspy.OutputField(description="Short feedback on the student's response indicating what they did wrong if they got the answer wrong. Omit if correct.")
    is_correct: bool = dspy.OutputField(description="Whether the student's answer is correct")

class GradeCFAResponse(dspy.Signature):
    """
    You are an expert CFA (Chartered Financial Analyst).
    You will be given a CFA exam question with multiple choice answers (A, B, C) and a student's response.
    Your task is to determine if the student's response is correct.

    The student's response may contain reasoning followed by a final answer.
    Look for the final answer choice (A, B, or C) in the response and evaluate its correctness.
    """
    question: str = dspy.InputField(description="The CFA exam question with choices A, B, C")
    correct_answer: str = dspy.InputField(description="The correct answer (A, B, or C)")
    correct_explanation: str = dspy.InputField(description="The correct answer explanation")
    student_response: str = dspy.InputField(description="The student's response to grade")

    extracted_answer: str = dspy.OutputField(description="The answer choice (A, B, or C) extracted from the student's response")
    feedback: str = dspy.OutputField(description="Short feedback on the student's response indicating what they did wrong if they got the answer wrong. Omit if correct.")
    is_correct: bool = dspy.OutputField(description="Whether the student's answer is correct")

# --- Dataset Configuration ---
DATASET_CONFIGS = {
    "gsm8k": {
        "eval_type": "Math",
        "gold_config": [ExprExtractionConfig(try_extract_without_anchor=True)],
        "pred_config": [LatexExtractionConfig(), ExprExtractionConfig()],
        "llm_grader": GradeMathResponse,
        "llm_kwargs_builder": lambda item, resp: {"question": item["question"], "correct_answer": item["expected_answer"], "student_response": resp[-500:]},
    },
    "aime": {
        "eval_type": "Math",
        "gold_config": [ExprExtractionConfig(try_extract_without_anchor=True)],
        "pred_config": [LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()],
        "llm_grader": GradeMathResponse,
        "llm_kwargs_builder": lambda item, resp: {"question": item["question"], "correct_answer": item["expected_answer"], "student_response": resp[-500:]},
    },
    "cfa-l1": {
        "eval_type": "CFA",
        "gold_config": [StringExtractionConfig(strings=("A", "B", "C"), lowercase=True, try_extract_without_anchor=True)],
        "pred_config": [StringExtractionConfig(strings=("A", "B", "C"), lowercase=True)],
        "gold_preprocessor": lambda ans: ans.split('_')[-1].upper() if '_' in ans else ans.upper(),
        "llm_grader": GradeCFAResponse,
        "llm_kwargs_builder": lambda item, resp: {"question": item["prompt_question"], "correct_answer": item["expected_answer"], "correct_explanation": item.get("explanation", ""), "student_response": resp},
    },
}

DATASET_CONFIGS["math500"] = DATASET_CONFIGS["gsm8k"]
DATASET_CONFIGS["cfa-l2"] = DATASET_CONFIGS["cfa-l1"]

def setup_dspy(model_name: str, max_tokens: int = 4096) -> None:
    """Configures dspy with the specified OpenRouter model."""
    if "OPENROUTER_API_KEY" not in os.environ:
        raise EnvironmentError("The --use-llm or --llm-only flag requires the OPENROUTER_API_KEY environment variable.")
    lm = dspy.LM(f"openrouter/{model_name}", max_tokens=max_tokens)
    dspy.configure(lm=lm)

def grade_with_llm(grader_signature: dspy.Signature, **kwargs) -> Dict[str, Any]:
    """Generic function to grade with a dspy signature."""
    try:
        grader = dspy.ChainOfThought(grader_signature)
        result = grader(**kwargs)
        return {"is_correct": result.is_correct, "feedback": result.feedback, "extracted_answer": result.extracted_answer}
    except Exception as e:
        return {"is_correct": False, "feedback": f"LLM_GRADING_ERROR: {str(e)}", "extracted_answer": ""}

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True, help="Path to the directory with result files (*.json).")
    parser.add_argument("--dataset", type=str, required=True, choices=DATASET_CONFIGS.keys(), help="Name of the dataset being evaluated.")
    parser.add_argument("--output-file", type=Path, default=None, help="Path to save the detailed evaluation JSON file. Defaults to <input-dir>/evaluation_summary.json.")
    parser.add_argument("--error-dir", type=Path, default=None, help="Directory to save detailed error analysis files. Defaults to <input-dir>/error_analysis.")
    parser.add_argument("--use-llm", action="store_true", help="Enable LLM-based grading for answers that fail the initial check.")
    parser.add_argument("--llm-only", action="store_true", help="Bypass the parser and use LLM-based grading for all answers. Implies --use-llm.")
    parser.add_argument("--llm-model", type=str, default="google/gemini-2.5-flash", help="Model for LLM grading via OpenRouter.")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of parallel workers for LLM calls (default: 10)")
    return parser.parse_args()

def serialize_object(obj: Any) -> str:
    """Convert a parsed object (e.g., from sympy) to a string for JSON."""
    if obj is None: return ""
    if isinstance(obj, list): return ", ".join(map(str, obj)) if obj else "[]"
    return str(obj)

def _evaluate_single_method(
    response: str,
    parsed_gold: Any,
    item_data: Dict,
    args: argparse.Namespace,
    config: Dict
) -> Tuple[bool, bool, Dict]:
    """
    Evaluates a single response (e.g., 'standard' or 'entropix').
    Returns: (is_correct_base, is_correct_final, details_dict)
    """
    if not response:
        return False, False, {}

    # --- LLM-Only Mode: Skip parser and go straight to LLM grading ---
    if args.llm_only:
        llm_kwargs = config["llm_kwargs_builder"](item_data, response)
        llm_result = grade_with_llm(config["llm_grader"], **llm_kwargs)
        is_correct = llm_result.get("is_correct", False) if llm_result else False

        details = {
            "response": response,
            "parsed_prediction": "SKIPPED (LLM-only mode)",
            "correct_base": False,  # Base parser check was skipped
            "llm_fallback_result": llm_result, # This is the primary result
            "correct_final": is_correct,
        }
        # For llm-only, base correctness is considered False, final correctness is the LLM's verdict.
        return False, is_correct, details

    # --- Hybrid/Parser-Only Mode Logic ---
    parsed_pred = parse(response, extraction_config=config["pred_config"])
    is_correct_base = verify(parsed_gold, parsed_pred)
    is_correct_final = is_correct_base
    llm_fallback_details = None

    if args.use_llm and not is_correct_base:
        llm_kwargs = config["llm_kwargs_builder"](item_data, response)
        llm_result = grade_with_llm(config["llm_grader"], **llm_kwargs)
        if llm_result and llm_result.get("is_correct"):
            is_correct_final = True
        llm_fallback_details = llm_result

    details = {
        "response": response,
        "parsed_prediction": serialize_object(parsed_pred),
        "correct_base": is_correct_base,
        "llm_fallback_result": llm_fallback_details,
        "correct_final": is_correct_final,
    }
    return is_correct_base, is_correct_final, details

def _process_single_item(item: Dict, args: argparse.Namespace, config: Dict) -> Dict:
    """
    Process a single item (question) and return evaluation results.
    This function is designed to be called in parallel.
    """
    # Pre-process gold answer if a preprocessor is defined
    gold_str_raw = item.get("expected_answer", item.get("gold_answer", ""))
    gold_preprocessor = config.get("gold_preprocessor", lambda x: x)
    gold_str = gold_preprocessor(gold_str_raw)
    item["expected_answer"] = gold_str  # Update item for llm_kwargs_builder

    parsed_gold = parse(gold_str, extraction_config=config["gold_config"], extraction_mode="any_match")

    # Evaluate both methods
    is_std_correct_base, is_std_correct_final, std_details = _evaluate_single_method(
        item.get("standard", {}).get("response", ""), parsed_gold, item, args, config
    )
    is_ent_correct_base, is_ent_correct_final, ent_details = _evaluate_single_method(
        item.get("entropix", {}).get("response", ""), parsed_gold, item, args, config
    )

    # Store results
    result_details = {
        "question_id": item.get("question_id", item.get("id")),
        "question": item.get("question", item.get("prompt_question")),
        "gold_answer": gold_str,
        "parsed_gold": serialize_object(parsed_gold),
        "standard": std_details,
        "entropix": ent_details,
        "is_std_correct_base": is_std_correct_base,
        "is_std_correct_final": is_std_correct_final,
        "is_ent_correct_base": is_ent_correct_base,
        "is_ent_correct_final": is_ent_correct_final,
    }
    
    return result_details

def run_evaluation_parallel(args: argparse.Namespace, console: Console, all_results: List[Dict], config: Dict):
    """Runs the main evaluation loop with parallel processing for LLM calls."""
    error_dir = args.error_dir or args.input_dir / "evaluation" / "error_analysis"
    error_dir.parent.mkdir(parents=True, exist_ok=True)
    error_dir.mkdir(exist_ok=True)

    evaluation_data = []
    correct_counts_base = {"standard": 0, "entropix": 0}
    correct_counts_final = {"standard": 0, "entropix": 0}

    mode_desc = ""
    if args.llm_only:
        mode_desc = " with LLM-Only Grading"
    elif args.use_llm:
        mode_desc = " with LLM Fallback"
    progress_desc = f"Evaluating {config['eval_type']}{mode_desc}"

    # Determine if we need parallel processing
    use_parallel = args.use_llm and len(all_results) > 1
    
    with Progress(
        SpinnerColumn(), 
        TextColumn("[progress.description]{task.description}"), 
        BarColumn(), 
        MofNCompleteColumn(), 
        TimeRemainingColumn(), 
        console=console
    ) as progress:
        task = progress.add_task(f"{progress_desc}...", total=len(all_results))

        if use_parallel:
            # Parallel processing with ThreadPoolExecutor
            console.print(f"[cyan]Using {args.num_workers} parallel workers for LLM calls...[/cyan]")
            
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                # Submit all tasks
                future_to_item = {
                    executor.submit(_process_single_item, item, args, config): item 
                    for item in all_results
                }
                
                # Process completed tasks as they finish
                for future in as_completed(future_to_item):
                    try:
                        result_details = future.result()
                        
                        # Update counts
                        correct_counts_base["standard"] += result_details["is_std_correct_base"]
                        correct_counts_final["standard"] += result_details["is_std_correct_final"]
                        correct_counts_base["entropix"] += result_details["is_ent_correct_base"]
                        correct_counts_final["entropix"] += result_details["is_ent_correct_final"]
                        
                        # Remove the temporary count fields
                        for key in ["is_std_correct_base", "is_std_correct_final", 
                                   "is_ent_correct_base", "is_ent_correct_final"]:
                            result_details.pop(key, None)
                        
                        evaluation_data.append(result_details)
                        
                        # Save error file if either method failed
                        if not result_details["standard"]["correct_final"] or not result_details["entropix"]["correct_final"]:
                            with open(error_dir / f"error_q{result_details['question_id']}.json", "w") as f_err:
                                json.dump(result_details, f_err, indent=2)
                        
                        progress.update(task, advance=1)
                        
                    except Exception as e:
                        console.print(f"[red]Error processing item: {e}[/red]")
                        progress.update(task, advance=1)
        else:
            # Sequential processing (original logic)
            for item in all_results:
                result_details = _process_single_item(item, args, config)
                
                # Update counts
                correct_counts_base["standard"] += result_details["is_std_correct_base"]
                correct_counts_final["standard"] += result_details["is_std_correct_final"]
                correct_counts_base["entropix"] += result_details["is_ent_correct_base"]
                correct_counts_final["entropix"] += result_details["is_ent_correct_final"]
                
                # Remove the temporary count fields
                for key in ["is_std_correct_base", "is_std_correct_final", 
                           "is_ent_correct_base", "is_ent_correct_final"]:
                    result_details.pop(key, None)
                
                evaluation_data.append(result_details)
                
                # Save error file if either method failed
                if not result_details["standard"]["correct_final"] or not result_details["entropix"]["correct_final"]:
                    with open(error_dir / f"error_q{result_details['question_id']}.json", "w") as f_err:
                        json.dump(result_details, f_err, indent=2)
                
                progress.update(task, advance=1)

    # Sort evaluation data by question_id for consistent output
    evaluation_data.sort(key=lambda x: x["question_id"])
    
    display_summary_table(console, config['eval_type'], args.dataset, len(all_results), 
                         correct_counts_base, correct_counts_final if args.use_llm else None, args)
    
    return evaluation_data

def display_summary_table(console: Console, eval_type: str, dataset_name: str, total: int, base_counts: dict, final_counts: Optional[dict], args: argparse.Namespace):
    """Creates and prints a summary table for the evaluation results."""
    title = f"{eval_type} Evaluation Summary for {dataset_name.upper()}"
    base_method_name = "math-verify" if eval_type == "Math" else "String Match"

    summary = Table(title=title, header_style="bold magenta", show_header=True)
    summary.add_column("Method", style="cyan", no_wrap=True)
    summary.add_column("Correct", style="green")
    summary.add_column("Total", style="blue")
    summary.add_column("Accuracy", style="yellow")

    for method in ["standard", "entropix"]:
        if method == "entropix" and total > 0: summary.add_section()
        method_name_display = method.capitalize()

        if args.llm_only:
            # In LLM-only mode, we only show the final LLM result.
            final_acc = (final_counts[method] / total) * 100 if total > 0 else 0
            summary.add_row(f"{method_name_display} (LLM-Only)", str(final_counts[method]), str(total), f"{final_acc:.2f}%")
        else:
            # Original hybrid mode display logic
            base_acc = (base_counts[method] / total) * 100 if total > 0 else 0
            summary.add_row(f"{method_name_display} ({base_method_name})", str(base_counts[method]), str(total), f"{base_acc:.2f}%")

            if final_counts:
                final_acc = (final_counts[method] / total) * 100 if total > 0 else 0
                delta = final_acc - base_acc
                delta_color = "green" if delta >= 0 else "red"
                delta_sign = "+" if delta >= 0 else ""
                summary.add_row(f"  + LLM Fallback", str(final_counts[method]), str(total), f"{final_acc:.2f}% ([bold {delta_color}]{delta_sign}{delta:.2f}%[/])")

    console.print(summary)

def main():
    args = _parse_args()
    console = Console()

    # If llm-only is specified, it automatically enables use-llm
    if args.llm_only:
        args.use_llm = True
        console.print("[bold yellow]LLM-only mode enabled. The parser will be skipped for all evaluations.[/bold yellow]")

    try:
        if args.use_llm:
            setup_dspy(args.llm_model)
    except (EnvironmentError, ImportError) as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        return

    console.print(f"[cyan]Scanning for result files (*.json) in '{args.input_dir}'...[/cyan]")
    json_files = sorted(list(args.input_dir.glob("*.json")), key=lambda p: p.stem)
    if not json_files:
        console.print(f"[red]Error: No result files found in {args.input_dir}. Exiting.[/red]")
        return
    console.print(f"[green]Found {len(json_files)} result files. Loading...[/green]")

    all_results = []
    for p in json_files:
        try:
            with p.open('r', encoding='utf-8') as f:
                all_results.append(json.load(f))
        except json.JSONDecodeError:
            console.print(f"[yellow]Warning: Could not decode JSON from {p.name}. Skipping.[/yellow]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not read file {p.name} due to {e}. Skipping.[/yellow]")

    if not all_results:
        console.print("[red]Error: No valid result files could be loaded. Exiting.[/red]")
        return

    dataset_config = DATASET_CONFIGS[args.dataset]
    console.print(f"[cyan]Using configuration for dataset: '{args.dataset}' (Type: {dataset_config['eval_type']})[/cyan]")

    # Run evaluation with parallel processing
    evaluation_data = run_evaluation_parallel(args, console, all_results, dataset_config)

    # Save summary
    if evaluation_data:
        output_path = args.output_file or args.input_dir / "evaluation" / "evaluation_summary.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding='utf-8') as f_out:
            json.dump(evaluation_data, f_out, indent=2, ensure_ascii=False)
        console.print(f"\n[bold green]Detailed evaluation report saved to:[/bold green] {output_path}")

if __name__ == "__main__":
    main()