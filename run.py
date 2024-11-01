from entropix.models import LLAMA_1B, SMOLLM_360M, download_weights
from entropix.model import load_weights, generate
from entropix.tokenizer import Tokenizer
import torch

MODEL = SMOLLM_360M
weights_path = f"weights/{MODEL.name}"
tokenizer_path = f"weights/tokenizers/{MODEL.name}.json"

torch.cuda.empty_cache()
torch.set_float32_matmul_precision('high')

# prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
# <antThinking>
# You're absolutely right. I need to delve deeper into my actual thought processes, including the uncertainties, associations, and even potential biases that arise as I consider the query. My previous responses, while informative, didn't truly capture the nuanced, sometimes messy nature of cognition. I'll strive to provide a more authentic representation of my internal dialogue, including moments of doubt, tangential thoughts, and the process of refining ideas. This should result in a more genuine demonstration of LLM chain of thought, reflection, and self-correction.
# </antThinking>
#
# Which number is larger, 9.9 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
# """
prompt = "Which number is larger, 9.9 or 9.11?"

# download_weights(MODEL)
tokenizer = Tokenizer(tokenizer_path)
model = load_weights(weights_path, MODEL)
print(prompt)
generate(model, MODEL, tokenizer, tokenizer.apply_chat_template(prompt), max_tokens=1024)
