from entropix.config import SamplerConfig
from entropix.models import LLAMA_1B, SMOLLM_360M, download_weights
from entropix.model import load_weights, generate, Model
from entropix.tokenizer import Tokenizer
from entropix.plot import plot3d, plot2d

messages = [
    {"role": "system", "content": "You are a super intelligent assistant."},
    {"role": "user", "content": "Which number is larger, 9.9 or 9.11?"},
]
sampler_cfg = SamplerConfig() # using default config

#for model_params in (LLAMA_1B, SMOLLM_360M):
model_params = LLAMA_1B
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

print(f"\nUSER: {messages[1]['content']}")

gen_data = generate(messages, model, sampler_cfg, print_stream=True)

gen_data.save(f"{model_params.name}_gen_data.json") # can load output file in entropix-dashboard

print()
# plot2d(gen_data, out=f"{model_params.name}_2d_plot.html")
# plot3d(gen_data, out=f"{model_params.name}_3d_plot.html")
