import json
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple, Optional

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

class SamplerState(Enum):
    ARGMAX = "Argmax"
    ADAPTIVE = "Adaptive sampling"
    TEMPERATURE = "Temperature sampling"
    PAUSE = "Pausing to think"
    BRANCHING = "Branching"

STATE_COLOR_MAP = {
    SamplerState.ARGMAX: '#FF8C9F',  # pink
    SamplerState.TEMPERATURE: '#FFA500',  # orange
    SamplerState.ADAPTIVE: '#800080',  # purple
    SamplerState.PAUSE: '#90EE90',  # lightgreen
    SamplerState.BRANCHING: '#ADD8E6',  # lightblue
}

@dataclass
class SamplerConfig:
    temperature: float = 0.666
    top_p: float = 0.90
    top_k: float = 27
    min_p: float = 0.03

    class ThresholdLevel(NamedTuple):
        low: float
        medium: float
        high: float

    class Thresholds(NamedTuple):
        logit_entropy: "SamplerConfig.ThresholdLevel"
        logit_varentropy: "SamplerConfig.ThresholdLevel"
        attn_entropy: "SamplerConfig.ThresholdLevel"
        attn_varentropy: "SamplerConfig.ThresholdLevel"
        agreement: "SamplerConfig.ThresholdLevel"
        interaction_strength: "SamplerConfig.ThresholdLevel"

    thresholds = Thresholds(
        logit_entropy=ThresholdLevel(low=0.6, medium=1.584, high=2.17),
        # logit_varentropy=ThresholdLevel(low=3.28, medium=3.85, high=6.18), # original
        logit_varentropy=ThresholdLevel(low=1.584, medium=3.28, high=5.50), # lowered
        attn_entropy=ThresholdLevel(low=8.989, medium=8.99, high=8.991),
        attn_varentropy=ThresholdLevel(low=5.212, medium=5.9125, high=6.92),
        agreement=ThresholdLevel(low=2e-06, medium=4e-06, high=5e-06),
        interaction_strength=ThresholdLevel(low=0.2, medium=0.247, high=0.264),
    )

    class AdaptiveCoefficients(NamedTuple):
        logit_entropy: float = 0.0
        logit_varentropy: float = 0.0
        attn_entropy: float = 0.0
        attn_varentropy: float = 0.0
        agreement: float = 0.0
        interaction_strength: float = 0.0

    class Adaptive(NamedTuple):
        n_samples: int
        temperature: "SamplerConfig.AdaptiveCoefficients"
        top_p: "SamplerConfig.AdaptiveCoefficients"
        top_k: "SamplerConfig.AdaptiveCoefficients"
        min_p: "SamplerConfig.AdaptiveCoefficients"
        score: "SamplerConfig.AdaptiveCoefficients"

    adaptive = Adaptive(
        n_samples=5,
        temperature=AdaptiveCoefficients(logit_entropy=0.3, attn_entropy=0.2, agreement=0.2),
        top_p=AdaptiveCoefficients(attn_varentropy=0.1),
        top_k=AdaptiveCoefficients(interaction_strength=0.3, agreement=0.2),
        min_p=AdaptiveCoefficients(logit_varentropy=0.5),
        score=AdaptiveCoefficients(logit_entropy=0.1, attn_entropy=0.2, logit_varentropy=0.3, attn_varentropy=0.4, agreement=0.5, interaction_strength=0.6),
    )

    class Offsets(NamedTuple):
        high_entropy_attn: float = 1.3
        low_entropy_interaction_strength: float = 1.2
        high_entropy_varentropy_attn: float = 2.0

    class Coefficients(NamedTuple):
        high_entropy_attn: float = 0.2
        low_entropy_interaction_strength: float = 0.3
        high_entropy_varentropy_attn: float = 0.5

    offsets = Offsets()
    coefficients = Coefficients()

    class Branching(NamedTuple):
        num_samples: int = 5

    branching = Branching()

    @classmethod
    def load(cls, path: str):
        with open(path, "r") as f:
            config = json.load(f)
        return cls(**config)
