from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, NamedTuple, Optional
import torch

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

@dataclass
class ModelConfig:
    name: str
    dim: int
    n_layers: int
    n_local_heads: int
    n_local_kv_heads: int
    head_dim: int
    max_seq_len: int
    rope_theta: float
    use_scaled_rope: bool
    hf_id: str | None = None

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

@dataclass
class SamplerConfig:
    states = {
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
    temperature = 0.666
    top_p = 0.90
    top_k = 27
    min_p = 0.03

    low_logits_entropy_threshold = 0.6
    medium_logits_entropy_threshold = 1.584
    high_logits_entropy_threshold = 2.17

    low_logits_varentropy_threshold = 3.28
    medium_logits_varentropy_threshold = 3.85
    high_logits_varentropy_threshold = 6.18

    low_attention_entropy_threshold = 8.989
    medium_attention_entropy_threshold = 8.99
    high_attention_entropy_threshold = 8.991

    low_attention_varentropy_threshold = 5.212
    medium_attention_varentropy_threshold = 5.9125
    high_attention_varentropy_threshold = 6.92

    low_agreement_threshold = 2e-06
    medium_agreement_threshold = 4e-06
    high_agreement_threshold = 5e-06

    low_interaction_strength_threshold = 0.2
    medium_interaction_strength_threshold = 0.247
    high_interaction_strength_threshold = 0.264

    high_entropy_attention_offset = 1.3
    high_entropy_attention_coefficient = 0.2

    low_entropy_interaction_strength_offset = 1.2
    low_entropy_interaction_strength_coefficient = 0.3

    high_entropy_varentropy_attention_offset = 2.0
    high_entropy_varentropy_attention_coefficient = 0.5

    n_adaptive_samples = 5

    adaptive_temperature_logits_coefficient = 0.3
    adaptive_temperature_attention_coefficient = 0.2
    adaptive_temperature_agreement_coefficient = 0.2
    adaptive_top_p_coefficient = 0.1
    adaptive_top_k_interaction_coefficient = 0.3
    adaptive_top_k_agreement_coefficient = 0.2
    adaptive_min_p_coefficient = 0.5
    adaptive_score_logits_entropy_coefficient = 0.1
    adaptive_score_attention_entropy_coefficient = 0.2
    adaptive_score_logits_varentropy_coefficient = 0.3
    adaptive_score_attention_varentropy_coefficient = 0.4
    adaptive_score_agreement_coefficient = 0.5
    adaptive_score_interaction_strength_coefficient = 0.6

class SamplerState(Enum):
    FLOWING = "Flowing with unspoken intent"
    TREADING = "Treading carefully, asking clarifying questions"
    EXPLORING = "Exploring forks in the path"
    RESAMPLING = "Resampling in the mist"
    ADAPTIVE = "Adaptive Sampling"
