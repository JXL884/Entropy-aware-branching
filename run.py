from entropix.config import SamplerConfig
from entropix.models import LLAMA_1B, SMOLLM_360M, download_weights
from entropix.model import load_weights, generate, Model
from entropix.tokenizer import Tokenizer
from entropix.plot import plot_entropy, plot_sampler

prompt = "Which number is larger, 9.9 or 9.11?"
sampler_cfg = SamplerConfig()  # using the default sampler thresholds, can specify inline to change
# e.g. to use all defaults except logit entropy thresholds
# sampler_cfg = SamplerConfig(
#     low_logits_entropy_threshold=0.3,
#     medium_logits_entropy_threshold=1.0,
#     high_logits_entropy_threshold=2.0,
# )

for model_params in (LLAMA_1B, SMOLLM_360M):
    print()
    print("=" * 80)
    print(model_params.name)
    print("=" * 80)

    download_weights(model_params)

    weights_path = f"weights/{model_params.name}"  # default location weights get saved to
    tokenizer_path = f"weights/tokenizers/{model_params.name}.json"  # default location tokenizer gets saved to

    tokenizer = Tokenizer(tokenizer_path)
    weights = load_weights(weights_path, model_params)
    model = Model(weights, model_params, tokenizer)

    print(f"\nUSER: {prompt}\n")

    # Don't forget to apply the chat template if using an instruct model
    prompt = tokenizer.apply_chat_template(prompt)

    gen_data = generate(prompt, model, print_stream=True)

    print()
    plot_sampler(gen_data, out=f"{model_params.name}_sampler_plot.html")
    plot_entropy(gen_data, sampler_cfg, out=f"{model_params.name}_entropy_plot.html")
