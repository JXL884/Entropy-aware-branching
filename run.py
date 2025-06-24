import torch
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from entropix.config import (
    SamplerConfig,
    ThresholdLevel,
    Thresholds,
    MODEL_CONFIG_OVERRIDES,
)
from entropix.model import Model, generate
from entropix.plot import plot2d, plot3d

def apply_config_overrides(model, config_name, config_overrides):
    if config_name not in config_overrides.keys():
        raise ValueError(f"Config {config_name} not found!")

    config_overrides = config_overrides[config_name]
    for attr_name, value in config_overrides.items():
        setattr(model.config, attr_name, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="Qwen/Qwen3-8B", help="Path to the model"
    )
    parser.add_argument(
        "--model_name", type=str, default="Qwen3-8B", help="Name of the model config to use",
    )
    parser.add_argument(
        #"--prompt", type=str, default="Find the number of ordered pairs $(x,y)$, where both $x$ and $y$ are integers between $-100$ and $100$, inclusive, such that $12x^{2}-xy-6y^{2}=0$." #answer: 117
        "--prompt", type=str, default="Which number is larger, 9.9 or 9.11?" 
    )
    parser.add_argument(
        "--use_prm_model", action="store_true", help="Use PRM model for scoring"
    )
    args = parser.parse_args()

    messages = [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": args.prompt},
    ]

    thresholds = Thresholds(
        logit_entropy=ThresholdLevel(low=1.2, medium=3, high=1),
        logit_varentropy=ThresholdLevel(low=3, medium=6.5, high=2),
    )

    sampler_cfg = SamplerConfig(thresholds=thresholds)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype="auto",
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    apply_config_overrides(base_model, args.model_name, MODEL_CONFIG_OVERRIDES)

    config = base_model.config
    model = Model(base_model, config, tokenizer)

    print(f"{args.model_name} output:\n")
    print("\n\n", "-"*50)
    gen_data = generate(
        messages,
        model,
        sampler_cfg,
        stream_output=True,
        enable_thinking=False,
        enable_uncertainty_detection=True,
        enable_insertion=True,
        insert_at_start=False,
        insert_at_end=False,
        insertion_text="\n\nWait,",  # "Considering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>.\n\n")
    )
    print("\n\n", "-"*50)
    print("Saving to output folder...")
    gen_data.save(f"output/{config.model_type}_gen_data.json")
    plot2d(gen_data, out=f"output/{config.model_type}_2d_plot.html")
    plot3d(gen_data, out=f"output/{config.model_type}_3d_plot.html")

if __name__ == "__main__":
    main()
