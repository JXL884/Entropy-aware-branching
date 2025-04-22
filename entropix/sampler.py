import torch
import torch.nn.functional as F
from typing import Tuple

from entropix.metrics import TokenMetrics
from entropix.config import SamplerState, SamplerConfig

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
    generator: torch.Generator | None = None,
) -> torch.Tensor:
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
                (1 - metrics.logit_entropy / cfg.thresholds.logit_entropy.high) * cfg.adaptive.score.logit_entropy,
                (1 - metrics.attn_entropy / cfg.thresholds.attn_entropy.high) * cfg.adaptive.score.attn_entropy,
                (1 - metrics.logit_varentropy / cfg.thresholds.logit_varentropy.high) * cfg.adaptive.score.logit_varentropy,
                (1 - metrics.attn_varentropy / cfg.thresholds.attn_varentropy.high) * cfg.adaptive.score.attn_varentropy,
                (metrics.agreement / cfg.thresholds.agreement.high) * cfg.adaptive.score.agreement,
                (metrics.interaction_strength / cfg.thresholds.interaction_strength.high) * cfg.adaptive.score.interaction_strength
            ))
        # fmt: on

        return log_prob + confidence_score

    sample_scores = torch.stack([score_sample(sample) for sample in samples])
    best_sample_idx = torch.argmax(sample_scores)
    sampled_token = samples[best_sample_idx]
    return sampled_token

def branching_sample(logits: torch.Tensor, metrics: TokenMetrics, cfg: SamplerConfig, generator: torch.Generator | None = None) -> torch.Tensor:
    """
    Samples multiple tokens from the given logits using temperature sampling.

    Args:
        logits: Tensor of shape [vocab_size].
        metrics: TokenMetrics object containing entropy and variance metrics.
        cfg: SamplerConfig object containing sampling parameters.
        generator: Optional random generator for reproducibility.

    Returns:
        Tensor of shape [num_samples] containing the sampled token indices.
    """

    # TODO: should we set temperature differently?
    temp_adj = cfg.offsets.low_entropy_interaction_strength + cfg.coefficients.low_entropy_interaction_strength * metrics.interaction_strength
    temperature = min(1.5, cfg.temperature * temp_adj)

    # NOTE: only using temperature sampling in branches right now
    # TODO: cleanup / setup AB tests to find best branch sampling method

    device = logits.device
    logits = logits[:, -1]

    # # Convert logits to probabilities
    probs = F.softmax(logits, dim=-1)
 
    num_available_tokens = int((probs > 0).sum().item())

    # Adjust num_samples if necessary
    num_samples_to_draw = min(cfg.branching.num_samples, num_available_tokens)

    if num_samples_to_draw == 0:
        raise ValueError("No tokens available to sample after filtering. Adjust the sampling parameters.")

    # currently the shape is [[num_samples]] help me flatten it to just [num_samples]
    sampled_tokens = temperature_sample(logits, temperature=temperature, num_samples=num_samples_to_draw, generator=generator)
    # sampled_tokens = sampled_tokens.squeeze(0)  # Remove the extra dimension
    return sampled_tokens.to(torch.int32)  

def sample(
    logits: torch.Tensor,
    attention_scores: torch.Tensor,
    metrics: TokenMetrics,
    cfg: SamplerConfig,
    can_branch: bool = False,
    generator: torch.Generator = torch.Generator(device=device).manual_seed(1337),
    current_step: int = 0,
) -> Tuple[torch.Tensor, SamplerState]:
    # Low Entropy, Low Varentropy
    # if metrics.logit_entropy < cfg.thresholds.logit_entropy.low and metrics.logit_varentropy < cfg.thresholds.logit_varentropy.low:
    #     sampler_state = SamplerState.ARGMAX
    #     sampled_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True).to(torch.int32)
    #     return sampled_token, sampler_state

    if can_branch and (
        metrics.logit_entropy > cfg.thresholds.logit_entropy.high
        and metrics.logit_varentropy > cfg.thresholds.logit_varentropy.high and current_step > 20
    ):
        # (A) Check if we're still on cooldown
        if (current_step - cfg.last_pause_step) < cfg.cooldown_length:
            # Too soon since last PAUSE => skip pause
            sampler_state = SamplerState.ARGMAX
            sampled_token = adaptive_sample(logits, metrics, cfg, generator=generator)
            return sampled_token, sampler_state
        else:
            # Allowed to pause
            cfg.last_pause_step = current_step  # record we triggered pause now
            sampler_state = SamplerState.PAUSE
            sampled_token = adaptive_sample(logits, metrics, cfg, generator=generator)
            return sampled_token, sampler_state

    # Otherwise, normal flow => Argmax or other sampling
    sampler_state = SamplerState.ARGMAX
    sampled_token = adaptive_sample(logits, metrics, cfg, generator=generator)
    return sampled_token, sampler_state