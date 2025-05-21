from typing import Literal
from entropix.config import SamplerConfig, Thresholds, ThresholdLevel, Branching
from entropix.model import generate, Model
from entropix.tokenizer import Message
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from accelerate import Accelerator
import torch
from datasets import load_dataset

import pandas as pd

import tyro

# some config parameters are not compatible with the old ones, only part we need to change
config_overrides = {
"Qwen_1B": {
    "head_dim": 128,
    "use_scaled_rope": False,
    "n_layers": 28,
    "n_local_kv_heads": 2,
    "n_local_heads": 12
},
"Qwen_3B": {
    "head_dim": 128,
    "use_scaled_rope": False,
    "n_layers": 36,
    "n_local_kv_heads": 2,
    "n_local_heads": 16
},
"Qwen_7B": {
    "head_dim": 128,
    "use_scaled_rope": False,
    "n_layers": 28,
    "n_local_kv_heads": 4,
    "n_local_heads": 28
}
}

# Function to apply config overrides
def apply_config_overrides(model, config_name, config_overrides):
    if config_name not in config_overrides.keys():
        raise ValueError(f"Config {config_name} not found!")
    
    config_overrides = config_overrides[config_name]
    for attr_name, value in config_overrides.items():
        setattr(model.config, attr_name, value)


def main(save_dir: str, allow_branching: bool, random_select: bool, dataset: Literal["MATH","GSM"], feedback: Literal["llama3.3", "PRM"], model_name: str, num_samples: int, starting: int, do_insert: bool, insert_text: str, logit_entropy_high: float = 4.0, logit_varentropy_high: float = 10.5):
    SYS_PROMPT = "You are an math expert, you can reason over various math topics and answer the questions. Now let's think step by step to solve the problem. "

    if dataset == "MATH":
        df = load_dataset("HuggingFaceH4/MATH-500")
        df = df['test']
        df = df.rename_column("problem", "question")
    elif dataset == "GSM":
        df = load_dataset("openai/gsm8k", "main")
        df = df['test']
    else:
        ValueError("Wrong dataset name, should pick from MATH, GSM")

    thresholds = Thresholds(
        logit_entropy=ThresholdLevel(low=0.6, medium=1.584, high=logit_entropy_high),
        logit_varentropy=ThresholdLevel(low=1.584, medium=3.28, high=logit_varentropy_high)
    )

    branching = Branching(num_samples = num_samples)

    sampler_cfg = SamplerConfig(
        thresholds=thresholds,
        branching=branching
    )

    print(sampler_cfg.thresholds)
    print(sampler_cfg.branching)


    if model_name == "Qwen_7B":
        MODEL = "Qwen/Qwen2.5-7B-Instruct"
    elif model_name == "Qwen_3B":
        MODEL = "Qwen/Qwen2.5-3B-Instruct"
    elif model_name == "Qwen_1B":
        MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
    else:
        raise ValueError("Wrong model name, should pick from Qwen_7B, Qwen_3B, Qwen_1B")

    base_model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = base_model.config

    apply_config_overrides(base_model, model_name, config_overrides)
    config = base_model.config
    # Now model.config has the old key names (n_layer, n_embd, etc.)
    print(config)

    model = Model(base_model, config, tokenizer)

    score_model_name = 'RLHFlow/Llama3.1-8B-PRM-Deepseek-Data'
    accelerator = Accelerator()
    local_rank = accelerator.local_process_index
    score_tokenizer = AutoTokenizer.from_pretrained(score_model_name)
    score_model_params = AutoModelForCausalLM.from_pretrained(score_model_name, torch_dtype=torch.bfloat16).to(local_rank).eval()

    score_tokenizer.padding_side = "right"
    score_tokenizer.pad_token = score_tokenizer.eos_token
    score_model_params.config.pad_token_id = score_model_params.config.eos_token_id

    score_model = Model(None, score_model_params, score_tokenizer)
    for i, data in enumerate(df['question']):
        if i >= starting:
            while True:  # Keep retrying until successful
                try:
                    content = " \n The question is : " + data
                    # Prepare the content and messages
                    messages = [Message(role="system", content=SYS_PROMPT), Message(role="user", content=content)]

                    print(f"\n{messages}\n", flush=True)

                    # Generate data
                    gen_data = generate(
                        messages, 
                        model, 
                        score_model, 
                        sampler_cfg, 
                        print_stream=True, 
                        max_tokens=2048, 
                        feedback_provider=feedback, 
                        allow_branching=allow_branching, 
                        random_select=random_select,
                        do_insert=False,
                        insert_text=insert_text,
                    )
                    
                    # Save the generated data
                    gen_data.save(f"{save_dir}/{model_name}-{dataset}-{i}.json")

                    # Free memory after processing each question
                    del messages
                    torch.cuda.empty_cache()
                    
                    break  # Exit the retry loop if successful
                
                except Exception as e:
                    print(f"Error encountered for index {i}: {e}. Retrying...", flush=True)

if __name__ == "__main__":
    tyro.cli(main)
