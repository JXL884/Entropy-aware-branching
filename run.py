from entropix.config import SamplerConfig
from entropix.models import LLAMA_1B, LLAMA_3B, LLAMA_8B, SMOLLM_360M, download_weights
from entropix.model import load_weights, generate, Model
from entropix.tokenizer import Tokenizer
from entropix.plot import plot3d, plot2d
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from accelerate import Accelerator
import torch

messages = [
    {"role": "system", "content": "You are a super intelligent assistant."},
    {"role": "user", "content": "Which number is larger, 9.9 or 9.11?"},
]
sampler_cfg = SamplerConfig() # using default config

#for model_params in (LLAMA_1B, SMOLLM_360M):
model_params = LLAMA_8B
print()
print("=" * 80)
print(model_params.name)
print("=" * 80)

#download_weights(model_params)

weights_path = f"weights/{model_params.name}"  # default location weights get saved to
tokenizer_path = f"weights/tokenizers/{model_params.name}.json"  # default location tokenizer gets saved to

tokenizer = Tokenizer(tokenizer_path)
weights = load_weights(weights_path, model_params)
model = Model(weights, model_params, tokenizer)

# PRM model
score_model_name = 'RLHFlow/Llama3.1-8B-PRM-Deepseek-Data'
accelerator = Accelerator()
local_rank = accelerator.local_process_index
score_tokenizer = AutoTokenizer.from_pretrained(score_model_name)
score_model_params = AutoModelForCausalLM.from_pretrained(score_model_name, torch_dtype=torch.bfloat16).to(local_rank).eval()

score_tokenizer.padding_side = "right"
score_tokenizer.pad_token = score_tokenizer.eos_token
score_model_params.config.pad_token_id = score_model_params.config.eos_token_id

score_model = Model(None, score_model_params, score_tokenizer)

print(f"\nUSER: {messages[1]['content']}")

# feedback_provider should "PRM" or "llama3.3"
gen_data = generate(messages, model, score_model, sampler_cfg, print_stream=True, random_select = True)

gen_data.save(f"{model_params.name}_gen_data.json") # can load output file in entropix-dashboard

print()
# plot2d(gen_data, out=f"{model_params.name}_2d_plot.html")
# plot3d(gen_data, out=f"{model_params.name}_3d_plot.html")
