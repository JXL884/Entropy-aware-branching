from entropix.models import LLAMA_1B, SMOLLM_360M, download_weights
from entropix.model import load_weights, generate
from entropix.tokenizer import Tokenizer

prompt = "Which number is larger, 9.9 or 9.11?"

for model_cfg in (LLAMA_1B, SMOLLM_360M):
    print("="*80)
    print(model_cfg.name)
    print("="*80)

    weights_path = f"weights/{model_cfg.name}"
    tokenizer_path = f"weights/tokenizers/{model_cfg.name}.json"

    download_weights(model_cfg)
    tokenizer = Tokenizer(tokenizer_path)
    prompt_templated = tokenizer.apply_chat_template(prompt)
    model = load_weights(weights_path, model_cfg)
    print(prompt)
    generate(model, model_cfg, tokenizer, prompt_templated, max_tokens=1024)
    print()
    print()
