from dataclasses import dataclass, field
from types import SimpleNamespace
from enum import Enum
from typing import Callable, NamedTuple, Optional, TypedDict
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


# TODO: change these
class SamplerState(Enum):
    FLOWING = "Flowing with unspoken intent"
    TREADING = "Treading carefully, asking clarifying questions"
    EXPLORING = "Exploring forks in the path"
    RESAMPLING = "Resampling in the mist"
    ADAPTIVE = "Adaptive Sampling"

@dataclass
class SamplerConfig:
    temperature: float = 0.666
    top_p: float = 0.90
    top_k: float = 27
    min_p: float = 0.03

    n_adaptive_samples: int = 5

    class ThresholdLevel(NamedTuple):
        low: float
        medium: float
        high: float

    class Thresholds(NamedTuple):
        entropy: "SamplerConfig.ThresholdLevel"
        varentropy: "SamplerConfig.ThresholdLevel"
        attn_entropy: "SamplerConfig.ThresholdLevel"
        attn_varentropy: "SamplerConfig.ThresholdLevel"
        agreement: "SamplerConfig.ThresholdLevel"
        interaction_strength: "SamplerConfig.ThresholdLevel"

    thresholds = Thresholds(
            entropy=ThresholdLevel(low=0.6, medium=1.584, high=2.17),
            varentropy=ThresholdLevel(low=3.28, medium=3.85, high=6.18),
            attn_entropy=ThresholdLevel(low=8.989, medium=8.99, high=8.991),
            attn_varentropy=ThresholdLevel(low=5.212, medium=5.9125, high=6.92),
            agreement=ThresholdLevel(low=2e-06, medium=4e-06, high=5e-06),
            interaction_strength=ThresholdLevel(low=0.2, medium=0.247, high=0.264),
    )


    # TODO: revisit all below params

    # high_entropy_attention_offset: float = 1.3
    # high_entropy_attention_coefficient: float = 0.2
    #
    # low_entropy_interaction_strength_offset: float = 1.2
    # low_entropy_interaction_strength_coefficient: float = 0.3
    #
    # high_entropy_varentropy_attention_offset: float = 2.0
    # high_entropy_varentropy_attention_coefficient: float = 0.5


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

