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
class SamplerConfig:
    states: dict[str, bool] = field(
        default_factory=lambda: {
            # Low Entropy, Low Varentropy: "flowing with unspoken intent"
            "flowing": True,
            # High Entropy, Low Varentropy: "treading carefully, asking clarifying questions"
            "treading": True,
            # Low Entropy, High Varentropy: "exploring forks in the path"
            "exploring": True,
            "branching": True,
            # High Entropy, High Varentropy: "resampling in the mist"
            "resampling": True,
            # extras
            "agreement": False,
            "interaction_strength": False,
        }
    )

    # Sampler state extras
    temperature: float = 0.666
    top_p: float = 0.90
    top_k: float = 27
    min_p: float = 0.03

    low_logits_entropy_threshold: float = 0.6
    medium_logits_entropy_threshold: float = 1.584
    high_logits_entropy_threshold: float = 2.17

    low_logits_varentropy_threshold: float = 3.28
    medium_logits_varentropy_threshold: float = 3.85
    high_logits_varentropy_threshold: float = 6.18

    low_attention_entropy_threshold: float = 8.989
    medium_attention_entropy_threshold: float = 8.99
    high_attention_entropy_threshold: float = 8.991

    low_attention_varentropy_threshold: float = 5.212
    medium_attention_varentropy_threshold: float = 5.9125
    high_attention_varentropy_threshold: float = 6.92

    low_agreement_threshold: float = 2e-06
    medium_agreement_threshold: float = 4e-06
    high_agreement_threshold: float = 5e-06

    low_interaction_strength_threshold: float = 0.2
    medium_interaction_strength_threshold: float = 0.247
    high_interaction_strength_threshold: float = 0.264

    high_entropy_attention_offset: float = 1.3
    high_entropy_attention_coefficient: float = 0.2

    low_entropy_interaction_strength_offset: float = 1.2
    low_entropy_interaction_strength_coefficient: float = 0.3

    high_entropy_varentropy_attention_offset: float = 2.0
    high_entropy_varentropy_attention_coefficient: float = 0.5

    n_adaptive_samples: float = 5

    adaptive_temperature_logits_coefficient: float = 0.3
    adaptive_temperature_attention_coefficient: float = 0.2
    adaptive_temperature_agreement_coefficient: float = 0.2
    adaptive_top_p_coefficient: float = 0.1
    adaptive_top_k_interaction_coefficient: float = 0.3
    adaptive_top_k_agreement_coefficient: float = 0.2
    adaptive_min_p_coefficient: float = 0.5
    adaptive_score_logits_entropy_coefficient: float = 0.1
    adaptive_score_attention_entropy_coefficient: float = 0.2
    adaptive_score_logits_varentropy_coefficient: float = 0.3
    adaptive_score_attention_varentropy_coefficient: float = 0.4
    adaptive_score_agreement_coefficient: float = 0.5
    adaptive_score_interaction_strength_coefficient: float = 0.6

class SamplerState(Enum):
    FLOWING = "Flowing with unspoken intent"
    TREADING = "Treading carefully, asking clarifying questions"
    EXPLORING = "Exploring forks in the path"
    BRANCHING = "Branch thinking in parallel"
    RESAMPLING = "Resampling in the mist"
    ADAPTIVE = "Adaptive Sampling"
