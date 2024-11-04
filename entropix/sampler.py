import torch
import torch.nn.functional as F
from typing import Tuple

from entropix.stats import calculate_metrics
from entropix.config import SamplerState, SamplerConfig

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def multinomial_sample_one(probs_sort: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    """Samples one token from a multinomial distribution with sorted probabilities."""
    q = torch.rand(probs_sort.shape, generator=generator, device=probs_sort.device)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(torch.int32)

def _sample(logits: torch.Tensor, temperature: float, top_p: float, top_k: int, min_p: float, generator: torch.Generator | None = None) -> torch.Tensor:
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
    gen_tokens: torch.Tensor,
    logits: torch.Tensor,
    attention_scores: torch.Tensor,
    temperature: float,
    clarifying_question_token: int = 2564,
    generator: torch.Generator = torch.Generator(device=device).manual_seed(1337)
) -> Tuple[torch.Tensor, SamplerState]:
    metrics = calculate_metrics(logits, attention_scores)
    ent, vent = metrics.logits_entropy, metrics.logits_varentropy
    attn_ent, attn_vent = metrics.attn_entropy, metrics.attn_varentropy
    agreement = metrics.agreement
    interaction_strength = metrics.interaction_strength

    cfg = SamplerConfig()

    # Low Entropy, Low Varentropy: "flowing with unspoken intent"
    if cfg.states["flowing"] and (
        ent < cfg.low_logits_entropy_threshold and vent < cfg.low_logits_varentropy_threshold and attn_ent < cfg.low_attention_entropy_threshold
        and attn_vent < cfg.low_attention_varentropy_threshold and (not cfg.states["agreement"] or agreement < cfg.low_agreement_threshold) and
        (not cfg.states["interaction_strength"] or interaction_strength < cfg.low_interaction_strength_threshold)
    ):
        sampler_state = SamplerState.FLOWING
        sampled_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True).to(torch.int32)
        return sampled_token, sampler_state

    # High Entropy, Low Varentropy: "treading carefully, asking clarifying questions"
    elif cfg.states["treading"] and (
        ent > cfg.high_logits_entropy_threshold and vent < cfg.low_logits_varentropy_threshold and attn_ent < cfg.low_attention_entropy_threshold
        and attn_vent < cfg.low_attention_varentropy_threshold and (not cfg.states["agreement"] or agreement < cfg.low_agreement_threshold) and
        (not cfg.states["interaction_strength"] or interaction_strength < cfg.low_interaction_strength_threshold)
    ):
        sampler_state = SamplerState.TREADING
        # Insert a clarifying question token if not already present
        if not torch.isin(gen_tokens[:, -1], torch.tensor([clarifying_question_token], device=device, dtype=gen_tokens.dtype)).any():
            sampled_token = torch.tensor([[clarifying_question_token]], dtype=torch.int32, device=device)
            return sampled_token, sampler_state
        else:
            # If we've just asked a question, sample with slightly higher temperature
            temp_adj = cfg.high_entropy_attention_offset + cfg.high_entropy_attention_coefficient * attn_ent
            sampled_token = _sample(
                logits, temperature=min(1.5, cfg.temperature * temp_adj), top_p=cfg.top_p, top_k=cfg.top_k, min_p=cfg.min_p, generator=generator
            )
            return sampled_token, sampler_state

    # Low Entropy, High Varentropy: "exploring forks in the path"
    elif cfg.states["exploring"] and (
        ent < cfg.high_logits_entropy_threshold and vent > cfg.high_logits_varentropy_threshold and attn_ent < cfg.low_attention_entropy_threshold
        and attn_vent > cfg.high_attention_varentropy_threshold and (not cfg.states["agreement"] or agreement < cfg.low_agreement_threshold) and
        (not cfg.states["interaction_strength"] or interaction_strength < cfg.low_interaction_strength_threshold)
    ):
        sampler_state = SamplerState.EXPLORING
        temp_adj = cfg.low_entropy_interaction_strength_offset + cfg.low_entropy_interaction_strength_coefficient * interaction_strength
        top_k_adj = max(5, int(cfg.top_k * (1 + 0.5 * (1 - agreement))))
        sampled_token = _sample(
            logits, temperature=min(1.5, cfg.temperature * temp_adj), top_p=cfg.top_p, top_k=top_k_adj, min_p=cfg.min_p, generator=generator
        )
        return sampled_token, sampler_state

    # High Entropy, High Varentropy: "resampling in the mist"
    elif cfg.states["resampling"] and (
        ent > cfg.medium_logits_entropy_threshold and vent > cfg.high_logits_varentropy_threshold and attn_ent > cfg.high_attention_entropy_threshold
        and attn_vent > cfg.high_attention_varentropy_threshold and (not cfg.states["agreement"] or agreement > cfg.high_agreement_threshold) and
        (not cfg.states["interaction_strength"] or interaction_strength > cfg.high_interaction_strength_threshold)
    ):
        sampler_state = SamplerState.RESAMPLING
        # Use high temperature and adjusted top_p based on attention metrics
        temp_adj = cfg.high_entropy_varentropy_attention_offset + cfg.high_entropy_varentropy_attention_coefficient * attn_vent
        top_p_adj = max(0.5, cfg.top_p - cfg.high_entropy_attention_coefficient * attn_ent)
        sampled_token = _sample(
            logits, temperature=max(2.0, cfg.temperature * temp_adj), top_p=top_p_adj, top_k=cfg.top_k, min_p=cfg.min_p, generator=generator
        )
        return sampled_token, sampler_state

    # All other cases: use adaptive sampling
    else:
        sampler_state = SamplerState.ADAPTIVE
        '''temperature = 0.666
        sampled_token = adaptive_sample(
            logits,
            temperature=temperature,
            epsilon=0.1,
            generator=generator
        )'''
        logits_uncertainty = ent + vent
        attn_uncertainty = attn_ent + attn_vent

        temperature = cfg.temperature * (
            1 + cfg.adaptive_temperature_logits_coefficient * ent + cfg.adaptive_temperature_attention_coefficient * attn_ent -
            cfg.adaptive_temperature_agreement_coefficient * agreement
        )
        top_p = torch.clamp(torch.tensor(cfg.top_p * (1 + cfg.adaptive_top_p_coefficient * attn_vent)), 0.1, 1.0)
        top_k = int(
            torch.clamp(
                torch.round(
                    torch.tensor(cfg.top_k) * (
                        1 + cfg.adaptive_top_k_interaction_coefficient * interaction_strength -
                        cfg.adaptive_top_k_agreement_coefficient * agreement
                    )
                ),
                min=1,
                max=100
            ).item()
        )
        min_p = torch.clamp(torch.tensor((cfg.min_p * (1 - cfg.adaptive_min_p_coefficient * vent))), 0.01, 0.5)

        samples = []
        for _ in range(cfg.n_adaptive_samples):
            sample = _sample(logits, temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p, generator=generator)
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
                (1 - ent / cfg.high_logits_entropy_threshold) * cfg.adaptive_score_logits_entropy_coefficient +
                (1 - attn_ent / cfg.high_attention_entropy_threshold) * cfg.adaptive_score_attention_entropy_coefficient +
                (1 - vent / cfg.high_logits_varentropy_threshold) * cfg.adaptive_score_logits_varentropy_coefficient +
                (1 - attn_vent / cfg.high_attention_varentropy_threshold) * cfg.adaptive_score_attention_varentropy_coefficient +
                (agreement / cfg.high_agreement_threshold) * cfg.adaptive_score_agreement_coefficient +
                (interaction_strength / cfg.high_interaction_strength_threshold) * cfg.adaptive_score_interaction_strength_coefficient
            )
            return log_prob + confidence_score

        sample_scores = torch.stack([score_sample(sample) for sample in samples])
        best_sample_idx = torch.argmax(sample_scores)
        sampled_token = samples[best_sample_idx]
        return sampled_token, sampler_state
