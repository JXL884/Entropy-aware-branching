import json
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple, Optional
from pydantic import BaseModel, field_validator, model_validator

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

class ThresholdLevel(BaseModel):
    low: float
    medium: float
    high: float

class Thresholds(BaseModel):
    logit_entropy: ThresholdLevel = ThresholdLevel(low=0.6, medium=1.584, high=2.17)
    logit_varentropy: ThresholdLevel = ThresholdLevel(low=1.584, medium=3.28, high=5.50)
    # logit_entropy: ThresholdLevel = ThresholdLevel(low=1.08 * 1.2, medium=2.85 * 1.2, high=3.91 * 1.2)
    # logit_varentropy: ThresholdLevel = ThresholdLevel(low=2.85 * 1.2, medium=5.90 * 1.2, high=9.5 * 1.2)
    attn_entropy: ThresholdLevel = ThresholdLevel(low=8.989, medium=8.99, high=8.991)
    attn_varentropy: ThresholdLevel = ThresholdLevel(low=5.212, medium=5.9125, high=6.92)
    agreement: ThresholdLevel = ThresholdLevel(low=2e-06, medium=4e-06, high=5e-06)
    interaction_strength: ThresholdLevel = ThresholdLevel(low=0.2, medium=0.247, high=0.264)

class AdaptiveCoefficients(BaseModel):
    logit_entropy: float = 0.0
    logit_varentropy: float = 0.0
    attn_entropy: float = 0.0
    attn_varentropy: float = 0.0
    agreement: float = 0.0
    interaction_strength: float = 0.0

class Adaptive(BaseModel):
    n_samples: int = 5
    temperature: AdaptiveCoefficients = AdaptiveCoefficients(logit_entropy=0.3, attn_entropy=0.2, agreement=0.2)
    top_p: AdaptiveCoefficients = AdaptiveCoefficients(attn_varentropy=0.1)
    top_k: AdaptiveCoefficients = AdaptiveCoefficients(interaction_strength=0.3, agreement=0.2)
    min_p: AdaptiveCoefficients = AdaptiveCoefficients(logit_varentropy=0.5)
    score: AdaptiveCoefficients = AdaptiveCoefficients(
        logit_entropy=0.1, attn_entropy=0.2, logit_varentropy=0.3, attn_varentropy=0.4, agreement=0.5, interaction_strength=0.6
    )

class Offsets(BaseModel):
    high_entropy_attn: float = 1.3
    low_entropy_interaction_strength: float = 1.2
    high_entropy_varentropy_attn: float = 2.0

class Coefficients(BaseModel):
    high_entropy_attn: float = 0.2
    low_entropy_interaction_strength: float = 0.3
    high_entropy_varentropy_attn: float = 0.5

class Branching(BaseModel):
    num_samples: int = 5
    max_len: int = 5

# Main SamplerConfig Model
class SamplerConfig(BaseModel):
    temperature: float = 0.666
    top_p: float = 0.90
    top_k: int = 27
    min_p: float = 0.03
    thresholds: Thresholds = Thresholds()
    adaptive: Adaptive = Adaptive()
    offsets: Offsets = Offsets()
    coefficients: Coefficients = Coefficients()
    branching: Branching = Branching()
    self_feedback: bool = False

    @model_validator(mode='before')
    def validate_nested_models(cls, values):
        if isinstance(values.get('thresholds'), dict):
            current = Thresholds().model_dump()
            cls._deep_update(current, values['thresholds'])
            values['thresholds'] = Thresholds.model_validate(current)

        if isinstance(values.get('adaptive'), dict):
            current = Adaptive().model_dump()
            cls._deep_update(current, values['adaptive'])
            values['adaptive'] = Adaptive.model_validate(current)

        if isinstance(values.get('offsets'), dict):
            current = Offsets().model_dump()
            cls._deep_update(current, values['offsets'])
            values['offsets'] = Offsets.model_validate(current)

        if isinstance(values.get('coefficients'), dict):
            current = Coefficients().model_dump()
            cls._deep_update(current, values['coefficients'])
            values['coefficients'] = Coefficients.model_validate(current)

        if isinstance(values.get('branching'), dict):
            current = Branching().model_dump()
            cls._deep_update(current, values['branching'])
            values['branching'] = Branching.model_validate(current)

        return values

    @staticmethod
    def _deep_update(current: dict, updates: dict):
        for k, v in updates.items():
            if isinstance(v, dict) and k in current and isinstance(current[k], dict):
                SamplerConfig._deep_update(current[k], v)
            else:
                current[k] = v

    @classmethod
    def load(cls, path: str) -> 'SamplerConfig':
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls.model_validate(config_dict)

    @classmethod
    def from_dict(cls, config: dict):
        return cls.model_validate(config)

    def update(self, updates: dict) -> None:
        for key, value in updates.items():
            if hasattr(self, key):
                current_attr = getattr(self, key)
                if isinstance(current_attr, BaseModel) and isinstance(value, dict):
                    current_attr = current_attr.model_copy(update=value)
                    setattr(self, key, current_attr)
                else:
                    setattr(self, key, value)

    def to_dict(self) -> dict:
        return self.model_dump()
