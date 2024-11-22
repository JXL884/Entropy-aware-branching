import torch
import torch.nn.functional as F
from typing import Tuple

from entropix.stats import TokenMetrics, calculate_metrics
from entropix.config import SamplerState, SamplerConfig

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def multinomial_sample_one(probs_sort: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    """Samples one token from a multinomial distribution with sorted probabilities."""
    q = torch.rand(probs_sort.shape, generator=generator, device=probs_sort.device)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(torch.int32)

def _sample(logits: torch.Tensor, temperature: float, top_p: float, top_k: float, min_p: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """Temperature -> min_p -> top_k -> top_p"""
    bsz = logits.shape[0]
    logit = logits[:, -1]
    probs = F.softmax(logit / temperature, dim=-1)

    # Apply min_p sampling
    if min_p > 0.0:
        p_max = torch.max(probs, dim=-1, keepdim=True).values
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
    # Convert next_token to int64 before using it in gather
    next_token_g = torch.gather(probs_idx, -1, next_token.reshape(bsz, 1).to(torch.int64))
    return next_token_g.to(torch.int32)

def temperature_sample(logits: torch.Tensor, temperature: float, generator: torch.Generator | None = None) -> torch.Tensor:
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).to(torch.int32)

def top_p_sample(logits: torch.Tensor, top_p: float, generator: torch.Generator | None = None) -> torch.Tensor:
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
    return torch.multinomial(probs, num_samples=1, generator=generator).to(torch.int32)

def top_k_sample(logits: torch.Tensor, top_k: int, generator: torch.Generator | None = None) -> torch.Tensor:
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits[indices_to_remove] = float('-inf')
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).to(torch.int32)

def min_p_sample(logits: torch.Tensor, min_p: float, generator: torch.Generator | None = None) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)

    max_prob = torch.max(probs, dim=-1, keepdim=True).values  # noqa: PD011
    min_threshold = max_prob * min_p
    mask = probs < min_threshold

    # Set probabilities below the threshold to 0
    filtered_probs = probs.masked_fill(mask, 0.0)
    # Renormalize the remaining probabilities
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)

    return torch.multinomial(filtered_probs, num_samples=1, generator=generator).to(torch.int32)

def quadratic_sample(logits: torch.Tensor, factor: float, generator: torch.Generator | None = None) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    transformed_probs = probs ** (1 + factor)
    transformed_probs = transformed_probs / transformed_probs.sum(dim=-1, keepdim=True)
    return torch.multinomial(transformed_probs, num_samples=1, generator=generator).to(torch.int32)


def adaptive_sample(logits: torch.Tensor, temperature: float, epsilon: float = 0.01, generator: torch.Generator | None = None) -> torch.Tensor:
    """
    Perform adaptive sampling by dynamically adjusting the candidate set size based on entropy and varentropy.
    """
    bsz = logits.shape[0]
    logit = logits[:, -1]
    probs = F.softmax(logit / temperature, dim=-1)

    # Sort tokens by probability
    sorted_probs, sorted_indices = torch.topk(probs, k=probs.shape[-1], dim=-1)

    # Initialize candidate set size
    candidate_mask = torch.zeros_like(sorted_probs, dtype=torch.bool, device=logits.device)
    cumulative_entropy = torch.zeros(bsz, device=logits.device)
    cumulative_varentropy = torch.zeros(bsz, device=logits.device)
    # Initial entropy calculation
    previous_entropy = -torch.sum(sorted_probs[0] * torch.log2(torch.clamp(sorted_probs[0], 1e-10, 1.0)))

    i = 0
    while i < sorted_probs.shape[-1]:
        current_prob = sorted_probs[:, i]

        # Update entropy and varentropy with current token
        current_entropy = -torch.sum(current_prob * torch.log2(torch.clamp(current_prob, 1e-10, 1.0)))
        current_varentropy = torch.sum(current_prob * (torch.log2(torch.clamp(current_prob, 1e-10, 1.0)) + cumulative_entropy.unsqueeze(-1))**2)

        entropy_reduction = cumulative_entropy - current_entropy
        varentropy_reduction = cumulative_varentropy - current_varentropy

        # Update mask where entropy reduction is sufficient
        candidate_mask[:, i] = entropy_reduction >= epsilon

        # Update cumulative values
        cumulative_entropy = torch.where(entropy_reduction >= epsilon, cumulative_entropy.clone(), current_entropy)
        cumulative_varentropy = torch.where(entropy_reduction >= epsilon, cumulative_varentropy.clone(), current_varentropy)

        # Check continuation condition
        if not torch.any(entropy_reduction >= epsilon) or i >= sorted_probs.shape[-1] - 1:
            break

        i += 1

    # Mask out tokens not in the candidate set
    candidate_probs = sorted_probs * candidate_mask.float()
    candidate_probs = candidate_probs / torch.sum(candidate_probs, dim=-1, keepdim=True)

    # Sample from the final candidate set
    next_token = multinomial_sample_one(candidate_probs, generator)
    next_token_g = torch.gather(sorted_indices, -1, next_token.to(torch.int64))

    return next_token_g.to(torch.int32)

def sample(
    gen_tokens: torch.Tensor,  # tokens generated so far
    logits: torch.Tensor,  # logits (distribution over all possible choices) of the next token
    attention_scores: torch.Tensor,  # internal attention scores (Q⋅Kᵀ)/√d
    metrics: TokenMetrics,
    cfg: SamplerConfig,
    clarifying_question_token: int = 2564,
    # generator: torch.Generator = torch.Generator(device=device).manual_seed(1337),
) -> Tuple[torch.Tensor, SamplerState]:
    # metrics = calculate_metrics(logits, attention_scores)
    logit_entropy, logit_varentropy = metrics.logits_entropy, metrics.logits_varentropy
    # NOTE: not using rn
    attn_entropy, attn_varentropy = metrics.attn_entropy, metrics.attn_varentropy
    agreement = metrics.agreement
    interaction_strength = metrics.interaction_strength

    # NOTE: previously a param, now hardcoded
    generator: torch.Generator = torch.Generator(device=device).manual_seed(1337)

    # Low Entropy, Low Varentropy
    if logit_entropy < cfg.thresholds.entropy.low and logit_varentropy < cfg.thresholds.entropy.low:
        sampler_state = SamplerState.FLOWING
        sampled_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True).to(torch.int32)
        return sampled_token, sampler_state

    # High Entropy, Low Varentropy TODO: inject "wait..." or something like that
    # NOTE: should either dynamically find the token from the tokenizer here, or accept it as a param and do so in generate

    # elif logit_entropy > cfg.thresholds.entropy.high and logit_varentropy < cfg.thresholds.varentropy.low:
    #     sampler_state = SamplerState.TREADING
    #     # TODO: change how we insert thinking tokens
    #     # Insert a clarifying question token if not already present
    #     if not torch.isin(gen_tokens[:, -1], torch.tensor([clarifying_question_token], device=device, dtype=gen_tokens.dtype)).any():
    #         sampled_token = torch.tensor([[clarifying_question_token]], dtype=torch.int32, device=device)
    #         return sampled_token, sampler_state
    #     else:
    #         # TODO: need a better way to check for this?
    #         pass
    #         # If we've just asked a question, sample with slightly higher temperature
    #         # temp_adj = cfg.high_entropy_attention_offset + cfg.high_entropy_attention_coefficient * attn_entropy
    #         sampled_token = _sample(
    #             # WARNING: hardcoded temporarily
    #             logits,
    #             temperature=min(1.5, cfg.temperature * 1.5),
    #             top_p=cfg.top_p,
    #             top_k=cfg.top_k,
    #             min_p=cfg.min_p,
    #             generator=generator
    #         )
    #         return sampled_token, sampler_state


    # TODO: branching, other sampler choices
    
    else: # All other cases: use adaptive sampling
        # TODO: break this out to its own function, revist how we are doing "adaptive sampling" **OR** just use a simpler sampler method
        sampler_state = SamplerState.ADAPTIVE
        logits_uncertainty = logit_entropy + logit_varentropy
        attn_uncertainty = attn_entropy + attn_varentropy

        # NOTE: adaptive temperature, not using config
        temperature = cfg.temperature * (
            1 + cfg.adaptive_temperature_logits_coefficient * logit_entropy + cfg.adaptive_temperature_attention_coefficient * attn_entropy -
            cfg.adaptive_temperature_agreement_coefficient * agreement
        )
        top_p = torch.clamp(torch.tensor(cfg.top_p * (1 + cfg.adaptive_top_p_coefficient * attn_varentropy)), 0.1, 1.0)
        top_k = int(
            torch.clamp(
                torch.round(
                    torch.tensor(cfg.top_k) *
                    (1 + cfg.adaptive_top_k_interaction_coefficient * interaction_strength - cfg.adaptive_top_k_agreement_coefficient * agreement)
                ),
                min=1,
                max=100
            ).item()
        )
        min_p = torch.clamp(torch.tensor((cfg.min_p * (1 - cfg.adaptive_min_p_coefficient * logit_varentropy))), 0.01, 0.5)

        samples = []
        for _ in range(cfg.n_adaptive_samples):
            sample = _sample(logits, temperature=temperature, top_p=top_p.item(), top_k=top_k, min_p=min_p.item(), generator=generator)
            samples.append(sample)

        def score_sample(sample):
            # Ensure sample is a 1D tensor of indices
            sample_indices = sample.view(-1).to(torch.long)

            # Create one-hot encoding
            one_hot = F.one_hot(sample_indices, num_classes=logits.shape[-1])

            # Calculate log probability
            log_probs = F.log_softmax(logits[:, -1], dim=-1)
            log_prob = torch.sum(log_probs * one_hot, dim=-1)

            confidence_score = (
                (1 - logit_entropy / cfg.thresholds.entropy.high) * cfg.adaptive_score_logits_entropy_coefficient +
                (1 - attn_entropy / cfg.thresholds.attn_entropy.high) * cfg.adaptive_score_attention_entropy_coefficient +
                (1 - logit_varentropy / cfg.thresholds.varentropy.high) * cfg.adaptive_score_logits_varentropy_coefficient +
                (1 - attn_varentropy / cfg.thresholds.attn_varentropy.high) * cfg.adaptive_score_attention_varentropy_coefficient +
                (agreement / cfg.thresholds.agreement.high) * cfg.adaptive_score_agreement_coefficient +
                (interaction_strength / cfg.thresholds.interaction_strength.high) * cfg.adaptive_score_interaction_strength_coefficient
            )
            return log_prob + confidence_score

        sample_scores = torch.stack([score_sample(sample) for sample in samples])
        best_sample_idx = torch.argmax(sample_scores)
        sampled_token = samples[best_sample_idx]
        return sampled_token, sampler_state
