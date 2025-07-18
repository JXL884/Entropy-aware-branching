#!/usr/bin/env python3
"""
Results Analysis Script for Baseline Testing

This script analyzes and compares results from baseline testing runs.
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
from rich import print as rprint
from rich.table import Table
from rich.console import Console

console = Console()

def load_results(results_dir: Path) -> Dict[str, Any]:
    """Load results from a results directory."""
    summary_file = results_dir / "summary.json"
    
    if not summary_file.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_file}")
    
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    # Load individual question results
    question_results = []
    for q_file in sorted(results_dir.glob("q*.json")):
        with open(q_file, 'r') as f:
            question_results.append(json.load(f))
    
    return {
        "summary": summary,
        "questions": question_results
    }

def compare_results(result_dirs: List[Path]) -> pd.DataFrame:
    """Compare results from multiple test runs."""
    comparisons = []
    
    for result_dir in result_dirs:
        try:
            results = load_results(result_dir)
            summary = results["summary"]
            
            # Extract model and dataset from directory name
            dir_parts = result_dir.name.split("_")
            if len(dir_parts) >= 3:
                model = dir_parts[1]
                dataset = dir_parts[2]
            else:
                model = summary.get("model", "unknown")
                dataset = summary.get("dataset", "unknown")
            
            comparison = {
                "model": model,
                "dataset": dataset,
                "total_questions": summary["total_questions"],
                "standard_accuracy": summary["standard_correct"] / summary["total_questions"] * 100,
                "entropix_accuracy": summary["entropix_correct"] / summary["total_questions"] * 100,
                "standard_avg_time": summary["standard_avg_time"],
                "entropix_avg_time": summary["entropix_avg_time"],
                "standard_avg_tokens": summary["standard_avg_tokens"],
                "entropix_avg_tokens": summary["entropix_avg_tokens"],
                "accuracy_improvement": (summary["entropix_correct"] - summary["standard_correct"]) / summary["total_questions"] * 100,
                "time_ratio": summary["entropix_avg_time"] / summary["standard_avg_time"] if summary["standard_avg_time"] > 0 else float('inf'),
                "results_dir": str(result_dir)
            }
            
            comparisons.append(comparison)
            
        except Exception as e:
            rprint(f"[red]Error loading results from {result_dir}: {e}[/red]")
    
    return pd.DataFrame(comparisons)

def create_comparison_table(df: pd.DataFrame) -> Table:
    """Create a rich table for comparison results."""
    table = Table(title="Baseline Testing Results Comparison")
    
    table.add_column("Model", style="cyan")
    table.add_column("Dataset", style="magenta")
    table.add_column("Standard Acc (%)", style="green")
    table.add_column("Entropix Acc (%)", style="blue")
    table.add_column("Improvement (%)", style="yellow")
    table.add_column("Time Ratio", style="red")
    table.add_column("Questions", style="white")
    
    for _, row in df.iterrows():
        improvement_color = "green" if row["accuracy_improvement"] > 0 else "red"
        time_color = "green" if row["time_ratio"] < 1.5 else "yellow" if row["time_ratio"] < 2.0 else "red"
        
        table.add_row(
            row["model"],
            row["dataset"],
            f"{row['standard_accuracy']:.1f}",
            f"{row['entropix_accuracy']:.1f}",
            f"[{improvement_color}]{row['accuracy_improvement']:+.1f}[/{improvement_color}]",
            f"[{time_color}]{row['time_ratio']:.2f}x[/{time_color}]",
            str(row["total_questions"])
        )
    
    return table

def create_visualizations(df: pd.DataFrame, output_dir: Path):
    """Create visualization plots."""
    output_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Accuracy comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Standard vs Entropix accuracy
    x = range(len(df))
    width = 0.35
    
    axes[0].bar([i - width/2 for i in x], df['standard_accuracy'], width, label='Standard', alpha=0.8)
    axes[0].bar([i + width/2 for i in x], df['entropix_accuracy'], width, label='Entropix', alpha=0.8)
    axes[0].set_xlabel('Test Configuration')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Accuracy Comparison: Standard vs Entropix')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{row['model']}\n{row['dataset']}" for _, row in df.iterrows()], rotation=45)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy improvement
    colors = ['green' if x > 0 else 'red' for x in df['accuracy_improvement']]
    axes[1].bar(x, df['accuracy_improvement'], color=colors, alpha=0.8)
    axes[1].set_xlabel('Test Configuration')
    axes[1].set_ylabel('Accuracy Improvement (%)')
    axes[1].set_title('Accuracy Improvement: Entropix - Standard')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{row['model']}\n{row['dataset']}" for _, row in df.iterrows()], rotation=45)
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Performance comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Generation time comparison
    axes[0].bar([i - width/2 for i in x], df['standard_avg_time'], width, label='Standard', alpha=0.8)
    axes[0].bar([i + width/2 for i in x], df['entropix_avg_time'], width, label='Entropix', alpha=0.8)
    axes[0].set_xlabel('Test Configuration')
    axes[0].set_ylabel('Average Generation Time (s)')
    axes[0].set_title('Generation Time Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{row['model']}\n{row['dataset']}" for _, row in df.iterrows()], rotation=45)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Time ratio
    time_colors = ['green' if x < 1.5 else 'yellow' if x < 2.0 else 'red' for x in df['time_ratio']]
    axes[1].bar(x, df['time_ratio'], color=time_colors, alpha=0.8)
    axes[1].set_xlabel('Test Configuration')
    axes[1].set_ylabel('Time Ratio (Entropix/Standard)')
    axes[1].set_title('Time Ratio: Entropix vs Standard')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{row['model']}\n{row['dataset']}" for _, row in df.iterrows()], rotation=45)
    axes[1].axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Equal Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Model-wise analysis
    if len(df['model'].unique()) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy by model
        model_accuracy = df.groupby('model')[['standard_accuracy', 'entropix_accuracy']].mean()
        model_accuracy.plot(kind='bar', ax=axes[0])
        axes[0].set_title('Average Accuracy by Model')
        axes[0].set_ylabel('Accuracy (%)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Time ratio by model
        model_time = df.groupby('model')['time_ratio'].mean()
        model_time.plot(kind='bar', ax=axes[1], color='orange')
        axes[1].set_title('Average Time Ratio by Model')
        axes[1].set_ylabel('Time Ratio')
        axes[1].axhline(y=1, color='black', linestyle='--', alpha=0.5)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'model_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze baseline testing results")
    parser.add_argument("results_dirs", nargs="+", help="Directories containing results")
    parser.add_argument("--output-dir", type=str, default="analysis_output", help="Output directory for analysis")
    parser.add_argument("--save-csv", action="store_true", help="Save comparison data as CSV")
    
    args = parser.parse_args()
    
    # Convert to Path objects
    result_dirs = [Path(d) for d in args.results_dirs]
    
    # Load and compare results
    rprint("[bold blue]Loading and comparing results...[/bold blue]")
    df = compare_results(result_dirs)
    
    if df.empty:
        rprint("[red]No valid results found![/red]")
        return
    
    # Display comparison table
    table = create_comparison_table(df)
    console.print(table)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save CSV if requested
    if args.save_csv:
        csv_file = output_dir / "comparison_results.csv"
        df.to_csv(csv_file, index=False)
        rprint(f"[green]Comparison data saved to: {csv_file}[/green]")
    
    # Create visualizations
    rprint("[bold blue]Creating visualizations...[/bold blue]")
    create_visualizations(df, output_dir)
    rprint(f"[green]Visualizations saved to: {output_dir}[/green]")
    
    # Print summary statistics
    rprint("\n[bold]Summary Statistics:[/bold]")
    rprint(f"Total test configurations: {len(df)}")
    rprint(f"Average accuracy improvement: {df['accuracy_improvement'].mean():.2f}%")
    rprint(f"Average time ratio: {df['time_ratio'].mean():.2f}x")
    rprint(f"Configurations with accuracy improvement: {(df['accuracy_improvement'] > 0).sum()}/{len(df)}")
    rprint(f"Configurations with acceptable time ratio (<2x): {(df['time_ratio'] < 2.0).sum()}/{len(df)}")

if __name__ == "__main__":
    main() 