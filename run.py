from entropix.config import SamplerConfig, Thresholds, ThresholdLevel, Branching
from entropix.model import generate, Model
from entropix.tokenizer import Tokenizer
from transformers import AutoTokenizer, Qwen2ForCausalLM
from transformers import AutoModelForCausalLM
from entropix.plot import plot3d, plot2d
from transformers import AutoTokenizer
from accelerate import Accelerator
import torch
from typing import *

messages = [
    {"role": "system", "content": "You are a super intelligent assistant."},
    {"role": "user", "content": "Which number is larger, 9.9 or 9.11?"},
]

messages = [
    {"role": "system", "content": "You are an expert financial analyst. "
    " You are given questions about various financial topics, from quantitative analysis to portfolio management to ethics of being a chartered financial analyst (CFA). "
    "Each question includes 3 potential answers, A B and C, one of which is correct (or in some cases, more correct than the others). "
    "Think step-by-step through the process of solving the question, definining relevant terms/formulas before applying them to the case at hand. "
    "Finally, indicate the correct answer: A, B, or C."},
    {"role": "user", "content": "<p>A random sample of 50 CFA exam candidates was found to have an average IQ of 130. The standard deviation among candidates is known (approximately 20). Assuming that IQs follow a normal distribution, the 2-sided 95% confidence interval for the mean IQ of CFA candidates is <em>closest to</em>:</p> "

    "A. [124.5; 135.5]. "
    "B. [125;135]. "
    "C. [130; 135.5]."},
]

thresholds = Thresholds(
    logit_entropy=ThresholdLevel(low=1.2, medium=3, high=2),
    logit_varentropy=ThresholdLevel(low=3, medium=6.5, high=4)
)

branching = Branching(num_samples = 1)

sampler_cfg = SamplerConfig(
    thresholds=thresholds,
    branching=branching
)

# MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# Load the model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
config = base_model.config



# some config parameters are not compatible with the old ones, only part we need to change
config_overrides = {
    "head_dim": 128,
    "use_scaled_rope": False,
    "n_layers": 28,
    "n_local_kv_heads": 2,
    "n_local_heads": 12
}

# Update the config dynamically if the attribute exists
for attr_name, value in config_overrides.items():
    setattr(base_model.config, attr_name, value)


config = base_model.config
# Now model.config has the old key names (n_layer, n_embd, etc.)
print(config)

model = Model(base_model, config, tokenizer)

# PRM model, COMMENT OUT FOR NOW!!!!!!!!!!!!!!!!!!!!!!    
# score_model_name = 'RLHFlow/Llama3.1-8B-PRM-Deepseek-Data'
# accelerator = Accelerator()
# local_rank = accelerator.local_process_index
# score_tokenizer = AutoTokenizer.from_pretrained(score_model_name)
# score_model_params = AutoModelForCausalLM.from_pretrained(score_model_name, torch_dtype=torch.bfloat16).to(local_rank).eval()

# score_tokenizer.padding_side = "right"
# score_tokenizer.pad_token = score_tokenizer.eos_token
# score_model_params.config.pad_token_id = score_model_params.config.eos_token_id

# score_model = Model(None, score_model_params, score_tokenizer)
 
print(f"\nUSER: {messages[1]['content']}")

# feedback_provider should "PRM" or "llama3.3"
gen_data = generate(messages, model, model, sampler_cfg, feedback_provider="llama3.3", print_stream=True, random_select = False, do_insert = True, insert_text=" To answer this question, let me think step by step. ")

gen_data.save(f"{config.model_type}_gen_data.json") # can load output file in entropix-dashboard

print()
# plot2d(gen_data, out=f"{model_params.name}_2d_plot.html")
# plot3d(gen_data, out=f"{model_params.name}_3d_plot.html")
