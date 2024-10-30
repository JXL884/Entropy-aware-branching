import torch
from typing import NamedTuple
from enum import Enum
from dataclasses import dataclass
from typing import Optional

DEFAULT_MASK_VALUE = -0.7 * float(torch.finfo(torch.float32).max)

@dataclass
class CLIConfig:
    """Configuration for text generation parameters.

    Attributes:
        prompt (str): The input text to generate from.
        max_tokens (int, optional): Maximum number of tokens to generate.
            Defaults to 600. Range: 1-2048.
        debug (bool, optional): Enable debug output during generation.
            Defaults to True.
        stream (bool, optional): Stream tokens as they're generated.
            Defaults to True.
        prompt_file (str, optional): Path to CSV file containing prompts.
            Defaults to None.
    """
    prompt: Optional[str] = None
    model: str = "llama-3.2-1b-instruct"
    max_tokens: Optional[int] = 600
    debug: bool = True
    stream: bool = True
    prompt_file: Optional[str] = None

    def __post_init__(self):
        """Validate inputs after initialization."""
        if self.prompt is None and self.prompt_file is None: raise ValueError("Either prompt or prompt_file must be provided")
        if self.prompt_file is None:
            if not isinstance(self.prompt, str): raise ValueError("prompt must be a string")
            if not self.prompt.strip(): raise ValueError("prompt cannot be empty")

        if self.max_tokens is not None:
            if not isinstance(self.max_tokens, int): raise ValueError("max_tokens must be an integer")
            if self.max_tokens < 1 or self.max_tokens > 2048:
                raise ValueError("max_tokens must be between 1 and 2048")


class EntropixConfig:
    def __init__(self):
        # Sampler state toggles

        # Adaptive state dynamic top_p, top_k, min_p adjustment toggles (old)
        '''self.state_dynamic_top_p = True
        self.state_dynamic_top_k = True
        self.state_dynamic_min_p = True'''

# params = {
#     "dim": 960,
#     "n_layers": 32,
#     "n_heads": 15,
#     "n_kv_heads": 5,
#     "vocab_size": 49152,
#     "norm_eps": 1e-05,
#     "rope_theta": 10000.0,
#     "use_scaled_rope": False,  # Inferred from "rope_scaling": null
#     "max_seq_len": 2048,  # Inferred from "max_position_embeddings"
# }

class ModelParams(NamedTuple):
    n_layers: int
    n_local_heads: int
    n_local_kv_heads: int
    head_dim: int
    max_seq_len: int
    rope_theta: float
    use_scaled_rope: bool

class LayerWeights(NamedTuple):
    wq: torch.Tensor
    wk: torch.Tensor
    wv: torch.Tensor
    wo: torch.Tensor
    w1: torch.Tensor
    w2: torch.Tensor
    w3: torch.Tensor
    ffn_norm: torch.Tensor
    attention_norm: torch.Tensor

class XfmrWeights(NamedTuple):
    tok_embeddings: torch.Tensor
    norm: torch.Tensor
    output: torch.Tensor
    layer_weights: list[LayerWeights]

# Experimental custom config to trigger different sampler states
class SamplerConfig:
    def __init__(self):
        self.states = {
            # Low Entropy, Low Varentropy: "flowing with unspoken intent"
            "flowing": True,
            # High Entropy, Low Varentropy: "treading carefully, asking clarifying questions"
            "treading": True,
            # Low Entropy, High Varentropy: "exploring forks in the path"
            "exploring": True,
            # High Entropy, High Varentropy: "resampling in the mist"
            "resampling": True,
            # extras
            "agreement": False,
            "interaction_strength": False,
        }

        # Sampler state extras
        self.temperature = 0.666
        self.top_p = 0.90
        self.top_k = 27
        self.min_p = 0.03

        self.low_logits_entropy_threshold = 0.6
        self.medium_logits_entropy_threshold = 1.584
        self.high_logits_entropy_threshold = 2.17

        self.low_logits_varentropy_threshold = 3.28
        self.medium_logits_varentropy_threshold = 3.85
        self.high_logits_varentropy_threshold = 6.18

        self.low_attention_entropy_threshold = 8.989
        self.medium_attention_entropy_threshold = 8.99
        self.high_attention_entropy_threshold = 8.991

        self.low_attention_varentropy_threshold = 5.212
        self.medium_attention_varentropy_threshold = 5.9125
        self.high_attention_varentropy_threshold = 6.92

        self.low_agreement_threshold = 2e-06
        self.medium_agreement_threshold = 4e-06
        self.high_agreement_threshold = 5e-06

        self.low_interaction_strength_threshold = 0.2
        self.medium_interaction_strength_threshold = 0.247
        self.high_interaction_strength_threshold = 0.264

        self.high_entropy_attention_offset = 1.3
        self.high_entropy_attention_coefficient = 0.2

        self.low_entropy_interaction_strength_offset = 1.2
        self.low_entropy_interaction_strength_coefficient = 0.3

        self.high_entropy_varentropy_attention_offset = 2.0
        self.high_entropy_varentropy_attention_coefficient = 0.5

        self.n_adaptive_samples = 5

        self.adaptive_temperature_logits_coefficient = 0.3
        self.adaptive_temperature_attention_coefficient = 0.2
        self.adaptive_temperature_agreement_coefficient = 0.2
        self.adaptive_top_p_coefficient = 0.1
        self.adaptive_top_k_interaction_coefficient = 0.3
        self.adaptive_top_k_agreement_coefficient = 0.2
        self.adaptive_min_p_coefficient = 0.5
        self.adaptive_score_logits_entropy_coefficient = 0.1
        self.adaptive_score_attention_entropy_coefficient = 0.2
        self.adaptive_score_logits_varentropy_coefficient = 0.3
        self.adaptive_score_attention_varentropy_coefficient = 0.4
        self.adaptive_score_agreement_coefficient = 0.5
        self.adaptive_score_interaction_strength_coefficient = 0.6

class SamplerState(Enum):
    FLOWING = "Flowing with unspoken intent"
    TREADING = "Treading carefully, asking clarifying questions"
    EXPLORING = "Exploring forks in the path"
    RESAMPLING = "Resampling in the mist"
    ADAPTIVE = "Adaptive Sampling"
