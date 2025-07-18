import json
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Literal, Dict

import torch
from pydantic import BaseModel, field_validator, model_validator

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
        if self.prompt is None and self.prompt_file is None:
            raise ValueError("Either prompt or prompt_file must be provided")
        if self.prompt_file is None:
            if not isinstance(self.prompt, str):
                raise ValueError("prompt must be a string")
            if not self.prompt.strip():
                raise ValueError("prompt cannot be empty")

        if self.max_tokens is not None:
            if not isinstance(self.max_tokens, int):
                raise ValueError("max_tokens must be an integer")
            if self.max_tokens < 1 or self.max_tokens > 2048:
                raise ValueError("max_tokens must be between 1 and 2048")


class SamplerState(Enum):
    ADAPTIVE = "Adaptive sampling"
    TEMPERATURE = "Temperature sampling"
    PAUSE = "Pausing to think"


STATE_COLOR_MAP = {
    SamplerState.TEMPERATURE: "#FFA500",  # orange
    SamplerState.ADAPTIVE: "#800080",  # purple
    SamplerState.PAUSE: "#90EE90",  # lightgreen
}


class ThresholdLevel(BaseModel):
    low: float
    medium: float
    high: float


class EWMAConfig(BaseModel):
    """Configuration for Exponentially Weighted Moving Average thresholding."""
    alpha: float = 0.1                  # Smoothing factor (0 < alpha < 1)
    min_samples: int = 30               # Minimum samples before using EWMA
    initial_multiplier: float = 1.0     # Multiplier for initial threshold values
    decay_factor: float = 0.95          # Decay factor for threshold adjustment


class DynamicThresholdConfig(BaseModel):
    """Configuration for dynamic thresholding strategies."""
    strategy: Literal["static", "ewma"] = "static"
    ewma: EWMAConfig = EWMAConfig()


class DynamicThresholdManager:
    """Manages dynamic thresholding strategies for entropy-based sampling."""
    
    def __init__(self, thresholds: 'Thresholds', config: DynamicThresholdConfig):
        self.base_thresholds = thresholds
        self.config = config
        self.strategy = config.strategy
        
        if self.strategy == "ewma":
            self._init_ewma_state()
    
    def _init_ewma_state(self):
        """Initialize EWMA state variables."""
        self.ewma_state = {
            'logit_entropy': {'low': None, 'medium': None, 'high': None},
            'logit_varentropy': {'low': None, 'medium': None, 'high': None},
            'attn_entropy': {'low': None, 'medium': None, 'high': None},
            'attn_varentropy': {'low': None, 'medium': None, 'high': None},
            'agreement': {'low': None, 'medium': None, 'high': None},
            'interaction_strength': {'low': None, 'medium': None, 'high': None}
        }
        self.sample_count = 0
        self.ewma_config = self.config.ewma
    
    def get_thresholds(self, current_metrics: Dict[str, float]) -> 'Thresholds':
        """Get current thresholds based on the selected strategy."""
        if self.strategy == "static":
            return self.base_thresholds
        elif self.strategy == "ewma":
            return self._get_ewma_thresholds(current_metrics)
        else:
            raise ValueError(f"Unknown thresholding strategy: {self.strategy}")
    
    def _get_ewma_thresholds(self, current_metrics: Dict[str, float]) -> 'Thresholds':
        """Calculate EWMA-based dynamic thresholds."""
        self.sample_count += 1
        
        # Initialize EWMA values if this is the first sample
        if self.sample_count == 1:
            self._initialize_ewma_values()
        
        # Update EWMA values for each metric
        for metric_name, metric_value in current_metrics.items():
            if metric_name in self.ewma_state:
                self._update_ewma_metric(metric_name, metric_value)
        
        # Create dynamic thresholds based on EWMA values
        dynamic_thresholds = {}
        for metric_name, levels in self.ewma_state.items():
            dynamic_thresholds[metric_name] = ThresholdLevel(
                low=self._get_ewma_threshold(metric_name, 'low'),
                medium=self._get_ewma_threshold(metric_name, 'medium'),
                high=self._get_ewma_threshold(metric_name, 'high')
            )
        
        # Create new Thresholds object with dynamic values
        return Thresholds(**dynamic_thresholds)
    
    def _initialize_ewma_values(self):
        """Initialize EWMA values with base thresholds."""
        for metric_name in self.ewma_state.keys():
            base_threshold = getattr(self.base_thresholds, metric_name)
            for level in ['low', 'medium', 'high']:
                base_value = getattr(base_threshold, level)
                self.ewma_state[metric_name][level] = base_value * self.ewma_config.initial_multiplier
    
    def _update_ewma_metric(self, metric_name: str, current_value: float):
        """Update EWMA values for a specific metric."""
        if metric_name not in self.ewma_state:
            return
        
        # Only start using EWMA after minimum samples
        if self.sample_count < self.ewma_config.min_samples:
            return
        
        # Update each threshold level based on current metric value
        for level in ['low', 'medium', 'high']:
            current_ewma = self.ewma_state[metric_name][level]
            if current_ewma is not None:
                # Calculate adaptive threshold based on current metric value
                adaptive_threshold = current_value * self._get_level_multiplier(level)
                
                # Apply EWMA update
                new_ewma = (self.ewma_config.alpha * adaptive_threshold + 
                           (1 - self.ewma_config.alpha) * current_ewma)
                
                # Apply decay factor to prevent thresholds from growing too large
                self.ewma_state[metric_name][level] = new_ewma * self.ewma_config.decay_factor
    
    def _get_level_multiplier(self, level: str) -> float:
        """Get multiplier for different threshold levels."""
        multipliers = {
            'low': 0.8,
            'medium': 1.0,
            'high': 1.2
        }
        return multipliers.get(level, 1.0)
    
    def _get_ewma_threshold(self, metric_name: str, level: str) -> float:
        """Get current EWMA threshold value."""
        if metric_name not in self.ewma_state or level not in self.ewma_state[metric_name]:
            # Fallback to base threshold
            base_threshold = getattr(self.base_thresholds, metric_name)
            return getattr(base_threshold, level)
        
        ewma_value = self.ewma_state[metric_name][level]
        if ewma_value is None:
            # Fallback to base threshold
            base_threshold = getattr(self.base_thresholds, metric_name)
            return getattr(base_threshold, level)
        
        return ewma_value
    
    def reset(self):
        """Reset the dynamic threshold manager state."""
        if self.strategy == "ewma":
            self._init_ewma_state()


class Thresholds(BaseModel):
    logit_entropy: ThresholdLevel = ThresholdLevel(low=0.6, medium=1.584, high=2.17)
    logit_varentropy: ThresholdLevel = ThresholdLevel(low=1.584, medium=3.28, high=5.50)
    attn_entropy: ThresholdLevel = ThresholdLevel(low=8.989, medium=8.99, high=8.991)
    attn_varentropy: ThresholdLevel = ThresholdLevel(low=5.212, medium=5.9125, high=6.92)
    agreement: ThresholdLevel = ThresholdLevel(low=2e-06, medium=4e-06, high=5e-06)
    interaction_strength: ThresholdLevel = ThresholdLevel(low=0.2, medium=0.247, high=0.264)
    dynamic: DynamicThresholdConfig = DynamicThresholdConfig()


class AdaptiveCoefficients(BaseModel):
    logit_entropy: float = 0.0
    logit_varentropy: float = 0.0
    attn_entropy: float = 0.0
    attn_varentropy: float = 0.0
    agreement: float = 0.0
    interaction_strength: float = 0.0


class Adaptive(BaseModel):
    n_samples: int = 5
    temperature: AdaptiveCoefficients = AdaptiveCoefficients(
        logit_entropy=0.3, attn_entropy=0.2, agreement=0.2
    )
    top_p: AdaptiveCoefficients = AdaptiveCoefficients(attn_varentropy=0.1)
    top_k: AdaptiveCoefficients = AdaptiveCoefficients(
        interaction_strength=0.3, agreement=0.2
    )
    min_p: AdaptiveCoefficients = AdaptiveCoefficients(logit_varentropy=0.5)
    score: AdaptiveCoefficients = AdaptiveCoefficients(
        logit_entropy=0.1,
        attn_entropy=0.2,
        logit_varentropy=0.3,
        attn_varentropy=0.4,
        agreement=0.5,
        interaction_strength=0.6,
    )


class Offsets(BaseModel):
    high_entropy_attn: float = 1.3
    low_entropy_interaction_strength: float = 1.2
    high_entropy_varentropy_attn: float = 2.0


class Coefficients(BaseModel):
    high_entropy_attn: float = 0.2
    low_entropy_interaction_strength: float = 0.3
    high_entropy_varentropy_attn: float = 0.5


# Main SamplerConfig Model
class SamplerConfig(BaseModel):
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.03
    thresholds: Thresholds = Thresholds()
    adaptive: Adaptive = Adaptive()
    offsets: Offsets = Offsets()
    coefficients: Coefficients = Coefficients()
    self_feedback: bool = False
    cooldown_length: int = 30

    @model_validator(mode="before")
    def validate_nested_models(cls, values):
        if isinstance(values.get("thresholds"), dict):
            current = Thresholds().model_dump()
            cls._deep_update(current, values["thresholds"])
            values["thresholds"] = Thresholds.model_validate(current)

        if isinstance(values.get("adaptive"), dict):
            current = Adaptive().model_dump()
            cls._deep_update(current, values["adaptive"])
            values["adaptive"] = Adaptive.model_validate(current)

        if isinstance(values.get("offsets"), dict):
            current = Offsets().model_dump()
            cls._deep_update(current, values["offsets"])
            values["offsets"] = Offsets.model_validate(current)

        if isinstance(values.get("coefficients"), dict):
            current = Coefficients().model_dump()
            cls._deep_update(current, values["coefficients"])
            values["coefficients"] = Coefficients.model_validate(current)

        return values

    @staticmethod
    def _deep_update(current: dict, updates: dict):
        for k, v in updates.items():
            if isinstance(v, dict) and k in current and isinstance(current[k], dict):
                SamplerConfig._deep_update(current[k], v)
            else:
                current[k] = v

    @classmethod
    def load(cls, path: str) -> "SamplerConfig":
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


MODEL_CONFIG_OVERRIDES = {
    "Qwen2.5_1B": {
        "head_dim": 128,
        "use_scaled_rope": False,
        "n_layers": 28,
        "n_local_kv_heads": 2,
        "n_local_heads": 12,
    },
    "Qwen2.5_3B": {
        "head_dim": 128,
        "use_scaled_rope": False,
        "n_layers": 36,
        "n_local_kv_heads": 2,
        "n_local_heads": 16,
    },
    "Qwen2.5_7B": {
        "head_dim": 128,
        "use_scaled_rope": False,
        "n_layers": 28,
        "n_local_kv_heads": 4,
        "n_local_heads": 28,
    },
    "deepseek": {
        "head_dim": 128,
        "use_scaled_rope": True,
        "n_layers": 36,
        "n_local_kv_heads": 8,
        "n_local_heads": 32,
    },
    "Qwen3-1.7B": {
        "head_dim": 128,
        "use_scaled_rope": False,
        "n_layers": 28,
        "n_local_kv_heads": 8,
        "n_local_heads": 16,
    },
    "Qwen3-4B": {
        "head_dim": 128,
        "use_scaled_rope": False,
        "n_layers": 36,
        "n_local_kv_heads": 8,
        "n_local_heads": 32,
    },
    "Qwen3-8B": {
        "head_dim": 128,
        "use_scaled_rope": False,
        "n_layers": 36,
        "n_local_kv_heads": 8,
        "n_local_heads": 32,
    },
    "Qwen3-14B": {
        "head_dim": 128,
        "use_scaled_rope": False,
        "n_layers": 40,
        "n_local_kv_heads": 8,
        "n_local_heads": 40,
    },
    "Qwen3-32B": {
        "head_dim": 128,
        "use_scaled_rope": False,
        "n_layers": 64,
        "n_local_kv_heads": 8,
        "n_local_heads": 64,
    },
}
