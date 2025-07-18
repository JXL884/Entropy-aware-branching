import torch
import torch.nn.functional as F
from typing import Tuple, Optional

from entropix.metrics import TokenMetrics
from entropix.config import SamplerState, SamplerConfig, DynamicThresholdManager

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def multinomial_sample_one(probs_sort: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    """Samples one token from a multinomial distribution with sorted probabilities."""
    q = torch.rand(probs_sort.shape, generator=generator, device=probs_sort.device)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(torch.int32)

def temperature_sample(logits: torch.Tensor, temperature: float, num_samples=1, generator: torch.Generator | None = None) -> torch.Tensor:
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=num_samples, generator=generator).to(torch.int32)

def top_p_sample(logits: torch.Tensor, top_p: float, num_samples=1, generator: torch.Generator | None = None) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    # Create a mask for probs that exceed the cumulative threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    # Scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    probs = probs.masked_fill(indices_to_remove, 0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=num_samples, generator=generator).to(torch.int32)

def top_k_sample(logits: torch.Tensor, top_k: int, num_samples=1, generator: torch.Generator | None = None) -> torch.Tensor:
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits[indices_to_remove] = float('-inf')
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=num_samples, generator=generator).to(torch.int32)

def min_p_sample(logits: torch.Tensor, min_p: float, num_samples=1, generator: torch.Generator | None = None) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)

    max_prob = torch.max(probs, dim=-1, keepdim=True).values  # noqa: PD011
    min_threshold = max_prob * min_p
    mask = probs < min_threshold

    # Set probabilities below the threshold to 0
    filtered_probs = probs.masked_fill(mask, 0.0)
    # Renormalize the remaining probabilities
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)

    return torch.multinomial(filtered_probs, num_samples=num_samples, generator=generator).to(torch.int32)

def quadratic_sample(logits: torch.Tensor, factor: float, num_samples=1, generator: torch.Generator | None = None) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    transformed_probs = probs**(1 + factor)
    transformed_probs = transformed_probs / transformed_probs.sum(dim=-1, keepdim=True)
    return torch.multinomial(transformed_probs, num_samples=num_samples, generator=generator).to(torch.int32)

def adaptive_sample(
    logits: torch.Tensor,
    metrics: TokenMetrics,
    cfg: SamplerConfig,
    threshold_manager: Optional[DynamicThresholdManager] = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    # Get current thresholds (static or dynamic)
    if threshold_manager is not None:
        current_thresholds = threshold_manager.get_thresholds({
            'logit_entropy': metrics.logit_entropy,
            'logit_varentropy': metrics.logit_varentropy,
            'attn_entropy': metrics.attn_entropy,
            'attn_varentropy': metrics.attn_varentropy,
            'agreement': metrics.agreement,
            'interaction_strength': metrics.interaction_strength
        })
    else:
        current_thresholds = cfg.thresholds
    
    # calculate adaptive sampling parameters
    temperature = cfg.temperature * (
        1 \
        + metrics.logit_entropy * cfg.adaptive.temperature.logit_entropy \
        + metrics.attn_entropy * cfg.adaptive.temperature.attn_entropy \
        - metrics.agreement * cfg.adaptive.temperature.agreement
    )
    top_p = torch.clamp(torch.tensor(cfg.top_p * (1 + metrics.attn_varentropy * cfg.adaptive.top_p.attn_varentropy)), 0.1, 1.0)
    top_k = int(
        torch.clamp(
            torch.round(
                torch.tensor(cfg.top_k) *
                (1 + (metrics.interaction_strength * cfg.adaptive.top_k.interaction_strength - metrics.agreement * cfg.adaptive.top_k.agreement))
            ),
            min=1,
            max=100
        ).item()
    )
    min_p = torch.clamp(torch.tensor((cfg.min_p * (1 - metrics.logit_varentropy * cfg.adaptive.min_p.logit_varentropy))), 0.01, 0.5)

    def _adaptive_sample():
        """Temperature -> min_p -> top_k -> top_p"""
        bsz = logits.shape[0]
        logit = logits[:, -1]
        probs = F.softmax(logit / temperature, dim=-1)

        # Apply min_p sampling
        if min_p > 0.0:
            p_max = torch.max(probs, dim=-1, keepdim=True).values  # noqa: PD011
            indices_to_remove = probs < (min_p * p_max)
            logit = torch.where(indices_to_remove, torch.full_like(logit, float('-inf')), logit)
            probs = F.softmax(logit, dim=-1)

        # Apply top-k sampling
        top_k_probs, top_k_indices = torch.topk(probs, k=min(top_k, probs.shape[-1]))
        probs_sort = torch.flip(top_k_probs, dims=[-1])
        probs_idx = torch.flip(top_k_indices, dims=[-1])
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        # Apply top-p sampling
        mask = torch.where(probs_sum - probs_sort > top_p, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
        probs_sort = probs_sort * (1 - mask)
        probs_sort = probs_sort / torch.sum(probs_sort, dim=-1, keepdim=True)

        next_token = multinomial_sample_one(probs_sort, generator)
        # next_tokens = torch.multinomial(probs_sort, num_samples=cfg.adaptive.n_samples, replacement=True, generator=generator)

        # Convert next_token to int64 before using it in gather
        next_token_g = torch.gather(probs_idx, -1, next_token.reshape(bsz, 1).to(torch.int64))
        # next_tokens_g = torch.gather(probs_idx.unsqueeze(1).expand(-1, cfg.adaptive.n_samples, -1), -1, next_tokens.to(torch.int64))

        return next_token_g.to(torch.int32)

    samples = [_adaptive_sample() for _ in range(cfg.adaptive.n_samples)]
    # print(f" [considering {len(set(s.item() for s in samples))} unique options]", end="")

    def score_sample(sample):
        # Ensure sample is a 1D tensor of indices
        sample_indices = sample.view(-1).to(torch.long)

        # Create one-hot encoding
        one_hot = F.one_hot(sample_indices, num_classes=logits.shape[-1])

        # Calculate log probability
        log_probs = F.log_softmax(logits[:, -1], dim=-1)
        log_prob = torch.sum(log_probs * one_hot, dim=-1)

        # fmt: off
        confidence_score = sum((
                (1 - metrics.logit_entropy / current_thresholds.logit_entropy.high) * cfg.adaptive.score.logit_entropy,
                (1 - metrics.attn_entropy / current_thresholds.attn_entropy.high) * cfg.adaptive.score.attn_entropy,
                (1 - metrics.logit_varentropy / current_thresholds.logit_varentropy.high) * cfg.adaptive.score.logit_varentropy,
                (1 - metrics.attn_varentropy / current_thresholds.attn_varentropy.high) * cfg.adaptive.score.attn_varentropy,
                (metrics.agreement / current_thresholds.agreement.high) * cfg.adaptive.score.agreement,
                (metrics.interaction_strength / current_thresholds.interaction_strength.high) * cfg.adaptive.score.interaction_strength
            ))
        # fmt: on

        return log_prob + confidence_score

    sample_scores = torch.stack([score_sample(sample) for sample in samples])
    best_sample_idx = torch.argmax(sample_scores)
    sampled_token = samples[best_sample_idx]
    return sampled_token

def sample(
    logits: torch.Tensor,
    attention_scores: torch.Tensor,
    metrics: TokenMetrics,
    cfg: SamplerConfig,
    threshold_manager: Optional[DynamicThresholdManager] = None,
    enable_uncertainty_detection: bool = False,
    generator: torch.Generator = torch.Generator(device=device).manual_seed(1337),
    current_step: int = 0,
    last_pause_step: int = -9999
) -> Tuple[torch.Tensor, SamplerState]:
    """
    Main sampling function that determines the sampling strategy based on entropy metrics.
    
    State Logic:
    - ADAPTIVE: Default state using adaptive sampling with temperature, top-p, top-k, and min-p
    - PAUSE: Triggered when entropy and varentropy are high (uncertainty detected)
    - TEMPERATURE: Available for future temperature-only sampling
    
    Args:
        logits: Model output logits
        attention_scores: Attention scores (currently unused)
        metrics: Token-level entropy and variance metrics
        cfg: Sampler configuration
        threshold_manager: Optional dynamic threshold manager for adaptive thresholds
        enable_uncertainty_detection: Whether to detect high uncertainty and trigger PAUSE state
        generator: Random generator for reproducibility
        current_step: Current generation step
        last_pause_step: Last step when pause was triggered (for cooldown)
    
    Returns:
        Tuple of (sampled_token, sampler_state)
    """
    # Get current thresholds (static or dynamic)
    if threshold_manager is not None:
        current_thresholds = threshold_manager.get_thresholds({
            'logit_entropy': metrics.logit_entropy,
            'logit_varentropy': metrics.logit_varentropy,
            'attn_entropy': metrics.attn_entropy,
            'attn_varentropy': metrics.attn_varentropy,
            'agreement': metrics.agreement,
            'interaction_strength': metrics.interaction_strength
        })
    else:
        current_thresholds = cfg.thresholds
    
    # Check if we should trigger pause/uncertainty detection logic
    if enable_uncertainty_detection and (
        metrics.logit_entropy > current_thresholds.logit_entropy.high
        and metrics.logit_varentropy > current_thresholds.logit_varentropy.high 
        and current_step > 30
    ):
        # Check if we're still on cooldown
        if (current_step - last_pause_step) < cfg.cooldown_length:
            # Too soon since last PAUSE => use adaptive sampling
            sampler_state = SamplerState.ADAPTIVE
            sampled_token = adaptive_sample(logits, metrics, cfg, threshold_manager, generator=generator)
            return sampled_token, sampler_state
        else:
            # Allowed to pause
            sampler_state = SamplerState.PAUSE
            sampled_token = adaptive_sample(logits, metrics, cfg, threshold_manager, generator=generator)
            return sampled_token, sampler_state
    else:
        # Normal flow => use adaptive sampling
        sampler_state = SamplerState.ADAPTIVE
        sampled_token = adaptive_sample(logits, metrics, cfg, threshold_manager, generator=generator)
        return sampled_token, sampler_state