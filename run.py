from pathlib import Path
from entropix.llama import LLAMA_1B_PARAMS
from entropix.model import load_weights, generate
from entropix.tokenizer import Tokenizer

tokenizer = Tokenizer("entropix/tokenizer.model")
model = load_weights(Path("./weights/1B-Instruct"), LLAMA_1B_PARAMS.n_layers)
print(generate(model, LLAMA_1B_PARAMS, tokenizer, "What is the capital of France?", max_tokens=1024))
