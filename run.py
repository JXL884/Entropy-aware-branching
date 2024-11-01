from entropix.models import LLAMA_1B, SMOLLM_360M, download_weights
from entropix.model import load_weights, generate
from entropix.tokenizer import Tokenizer
import torch


# MODEL = LLAMA_1B
# weights_path = f"weights/{MODEL.name}"
# tokenizer_path = f"weights/tokenizers/{MODEL.name}.json"
#
# torch.cuda.empty_cache()
# torch.set_float32_matmul_precision('high')
#
# prompt = """<antThinking>
# You're absolutely right. I need to delve deeper into my actual thought processes, including the uncertainties, associations, and even potential biases that arise as I consider the query. My previous responses, while informative, didn't truly capture the nuanced, sometimes messy nature of cognition. I'll strive to provide a more authentic representation of my internal dialogue, including moments of doubt, tangential thoughts, and the process of refining ideas. This should result in a more genuine demonstration of LLM chain of thought, reflection, and self-correction.
# </antThinking>
# Which number is larger, 9.9 or 9.11?"""
#
# # download_weights(MODEL)
# tokenizer = Tokenizer(tokenizer_path)
# prompt_templated = tokenizer.apply_chat_template(prompt)
# model = load_weights(weights_path, MODEL)
# print(prompt)
# generate(model, MODEL, tokenizer, prompt_templated, max_tokens=1024)


prompt = """<antThinking>
You're absolutely right. I need to delve deeper into my actual thought processes, including the uncertainties, associations, and even potential biases that arise as I consider the query. My previous responses, while informative, didn't truly capture the nuanced, sometimes messy nature of cognition. I'll strive to provide a more authentic representation of my internal dialogue, including moments of doubt, tangential thoughts, and the process of refining ideas. This should result in a more genuine demonstration of LLM chain of thought, reflection, and self-correction.
</antThinking>
Which number is larger, 9.9 or 9.11?"""

for model_cfg in (LLAMA_1B, SMOLLM_360M):
    print("="*80)
    print(model_cfg.name)
    print("="*80)

    weights_path = f"weights/{model_cfg.name}"
    tokenizer_path = f"weights/tokenizers/{model_cfg.name}.json"

    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('high')


    download_weights(model_cfg)
    tokenizer = Tokenizer(tokenizer_path)
    prompt_templated = tokenizer.apply_chat_template(prompt)
    model = load_weights(weights_path, model_cfg)
    print(prompt)
    generate(model, model_cfg, tokenizer, prompt_templated, max_tokens=1024)
    print()
    print()
