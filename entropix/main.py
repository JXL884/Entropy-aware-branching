import csv
import os
from datetime import datetime

import jax
import pandas as pd
import torch
import tyro

from entropix.config import CLIConfig
from entropix.smollm_tokenizer import download_tokenizer
from entropix.utils import validate_csv
from entropix.weight import download_weights
from entropix.model import EntropixModel

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()
torch.set_float32_matmul_precision('high')

def initialize_model():
    download_weights()
    _ = download_tokenizer()
    jax.clear_caches()
    torch.cuda.empty_cache()
    global entropix_model
    entropix_model = EntropixModel()
    print("Model initialized and ready to use!")

def generate_text() -> None:
    """Generate text using the model with the given configuration."""
    global entropix_model
    if 'entropix_model' not in globals():
        print("Model not initialized. Please run initialize_model() first.")
        return

    # Handle CSV input if provided
    if config.csv_file:
        if not validate_csv(config.csv_file):
            return

        df = pd.read_csv(config.csv_file)
        total_prompts = len(df)

        print(f"Processing {total_prompts} prompts from CSV file...")

        # Create output CSV file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"generated_responses_{timestamp}.csv"

        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['prompts', 'response'])

            for idx, row in df.iterrows():
                prompt = row['prompts'].strip()
                print(f"\nProcessing prompt {idx + 1}/{total_prompts}:")
                print(f"Prompt: {prompt}\n")

                if config.stream:
                    response = ""
                    print("Response: ", end='', flush=True)
                    for token in entropix_model.generate_stream(prompt, config.max_tokens, config.debug, batch=True):
                        print(token, end='', flush=True)
                        response += token
                    print()  # Final newline
                else:
                    response = entropix_model.generate(prompt, config.max_tokens, config.debug, batch=True)
                    print(f"Response: {response}\n")

                writer.writerow([prompt, response])

        print(f"\nAll responses have been saved to {output_file}")

    else:
        # Original single prompt behavior
        if config.stream:
            response = ""
            for token in entropix_model.generate_stream(config.prompt, config.max_tokens, config.debug):
                print(token, end='', flush=True)
                response += token
            print()  # Final newline
        else:
            response = entropix_model.generate(config.prompt, config.max_tokens, config.debug)
            print(response)

if __name__ == '__main__':
    torch.cuda.empty_cache()
    initialize_model()
    config = tyro.cli(CLIConfig)
    main(config)
