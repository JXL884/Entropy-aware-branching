import os
import math
from pathlib import Path
from typing import Optional, Tuple, NamedTuple

import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from entropix.config import DEFAULT_MASK_VALUE, LayerWeights, ModelParams, XfmrWeights
from entropix.tokenizer import Tokenizer
from entropix.sampler import sample
from entropix.stats import AttnStats
from entropix.kvcache import KVCache

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

################################################################################
#                                   Weights                                    #
################################################################################
def load_weights(ckpt_dir: Path, n_layers: int) -> XfmrWeights:
    w = {}
    layer_weights = []
    with torch.inference_mode():
        for file in ckpt_dir.glob("*.npy"):
            name = '.'.join(str(file).split('/')[-1].split('.')[:-1])
            jax_weight = jnp.load(file=file, mmap_mode='r', allow_pickle=True)
            np_weight = np.array(jax_weight).astype(np.float32)
            weight = torch.from_numpy(np_weight).to(torch.bfloat16).to(device)
            w[name] = weight.to(device)
        for i in range(n_layers):
            layer_weights.append(
                LayerWeights(
                    wq=w[f'layers.{i}.attention.wq.weight'],
                    wk=w[f'layers.{i}.attention.wk.weight'],
                    wv=w[f'layers.{i}.attention.wv.weight'],
                    wo=w[f'layers.{i}.attention.wo.weight'],
                    w1=w[f'layers.{i}.feed_forward.w1.weight'],
                    w2=w[f'layers.{i}.feed_forward.w2.weight'],
                    w3=w[f'layers.{i}.feed_forward.w3.weight'],
                    ffn_norm=w[f'layers.{i}.ffn_norm.weight'],
                    attention_norm=w[f'layers.{i}.attention_norm.weight'],
                )
            )
        xfmr_weights = XfmrWeights(tok_embeddings=w['tok_embeddings.weight'], norm=w['norm.weight'], output=w['output.weight'], layer_weights=layer_weights)
        return xfmr_weights

################################################################################
#                                 Attention                                    #
################################################################################
def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return w * (x * torch.rsqrt(torch.pow(x, 2).mean(-1, keepdim=True) + eps))

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    reshape_xq = xq.float().reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.float().reshape(*xk.shape[:-1], -1, 2)
    xq_ = torch.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    xk_ = torch.complex(reshape_xk[..., 0], reshape_xk[..., 1])
    xq_out = xq_ * freqs_cis.unsqueeze(0).unsqueeze(2)
    xk_out = xk_ * freqs_cis.unsqueeze(0).unsqueeze(2)
    xq_out = torch.stack((xq_out.real, xq_out.imag), dim=-1).reshape(*xq_out.shape[:-1], -1)
    xk_out = torch.stack((xk_out.real, xk_out.imag), dim=-1).reshape(*xk_out.shape[:-1], -1)
    return xq_out.to(dtype), xk_out.to(dtype)

def attention(
    x: torch.Tensor,
    layer_weights: LayerWeights,
    model_params,
    cur_pos: int,
    layer_idx: int,
    freqs_cis: torch.Tensor,
    kvcache: KVCache,
    attn_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, KVCache, torch.Tensor]:
    if x.dim() == 2:  # add batch dimension to 2d input
        bs = 1
        seq_len, dim = x.shape
        x = x.unsqueeze(0)
    else:
        bs, seq_len, dim = x.shape
    n_rep = model_params.n_local_heads // model_params.n_local_kv_heads
    xq = F.linear(x, layer_weights.wq).view(bs, seq_len, model_params.n_local_heads, model_params.head_dim)
    xk = F.linear(x, layer_weights.wk).view(bs, seq_len, model_params.n_local_kv_heads, model_params.head_dim)
    xv = F.linear(x, layer_weights.wv).view(bs, seq_len, model_params.n_local_kv_heads, model_params.head_dim)
    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis, dtype=xq.dtype)
    keys, values, kvcache = kvcache.update(xk, xv, layer_idx, cur_pos, n_rep)
    xq = xq.permute(0, 2, 1, 3)  # (bs, n_heads, seqlen, head_dim)
    keys = keys.permute(0, 2, 3, 1)  # (bs, n_heads, head_dim, cache_len + seqlen)
    values = values.permute(0, 2, 1, 3)  # (bs, n_heads, cache_len + seqlen, head_dim)
    scores = torch.matmul(xq, keys)
    pre_scores = scores / math.sqrt(model_params.head_dim)
    scores = pre_scores.to(torch.float32)  # Always do attention softmax at float32
    if cur_pos == 0: scores = scores + attn_mask
    mask = torch.where(scores != 0.0, scores, DEFAULT_MASK_VALUE)
    padded_logits = torch.where((mask >= DEFAULT_MASK_VALUE * 0.5), scores, DEFAULT_MASK_VALUE)
    scores = F.softmax(padded_logits, dim=-1).to(x.dtype)
    output = torch.matmul(scores.to(values.dtype), values)
    output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
    out = F.linear(output, layer_weights.wo)
    # If input was 2D, remove the batch dimension from the output
    if dim == 2: out = out.squeeze(0)
    return out, kvcache, pre_scores

def feed_forward(x: torch.Tensor, layer_weights: LayerWeights) -> torch.Tensor:
    return F.linear(F.silu(F.linear(x, layer_weights.w1)) * F.linear(x, layer_weights.w3), layer_weights.w2)

################################################################################
#                                 Transformer                                  #
################################################################################

def xfmr(
    xfmr_weights: XfmrWeights,
    model_params: ModelParams,
    tokens: torch.Tensor,
    cur_pos: int,
    freqs_cis: torch.Tensor,
    kvcache: KVCache,
    attn_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, KVCache, torch.Tensor, AttnStats]:
    h = xfmr_weights.tok_embeddings[tokens]
    attn_stats = AttnStats.new(bsz=tokens.shape[0], n_layers=model_params.n_layers, n_heads=model_params.n_local_heads, device=device)
    for i in range(model_params.n_layers):
        norm_x = rms_norm(h, xfmr_weights.layer_weights[i].attention_norm)
        h_attn, kvcache, scores = attention(norm_x, xfmr_weights.layer_weights[i], model_params, cur_pos, i, freqs_cis, kvcache, attn_mask=attn_mask)
        attn_stats = attn_stats.update(scores[:, :, -1, :], i)
        h = h + h_attn
        h = h + feed_forward(rms_norm(h, xfmr_weights.layer_weights[i].ffn_norm), xfmr_weights.layer_weights[i])
    logits = F.linear(rms_norm(h, xfmr_weights.norm), xfmr_weights.output)
    return logits, kvcache, scores, attn_stats

def apply_scaling(freqs: torch.Tensor) -> torch.Tensor:
    SCALE_FACTOR = 8.0
    LOW_FREQ_FACTOR = 1.0
    HIGH_FREQ_FACTOR = 4.0
    OLD_CONTEXT_LEN = 8192  # original llama3 length

    low_freq_wavelen = OLD_CONTEXT_LEN / LOW_FREQ_FACTOR
    high_freq_wavelen = OLD_CONTEXT_LEN / HIGH_FREQ_FACTOR

    def scale_freq(freq: torch.Tensor) -> torch.Tensor:
        wavelen = 2 * torch.pi / freq

        # Calculate smooth factor
        smooth = (OLD_CONTEXT_LEN / wavelen - LOW_FREQ_FACTOR) / (HIGH_FREQ_FACTOR - LOW_FREQ_FACTOR)
        smooth = torch.clamp(smooth, 0.0, 1.0)  # Ensure smooth is between 0 and 1

        # Calculate scaled frequency
        scaled = (1 - smooth) * freq / SCALE_FACTOR + smooth * freq

        # Apply conditional scaling
        scaled = torch.where(
            wavelen < high_freq_wavelen,
            freq,  # No scaling
            torch.where(
                wavelen > low_freq_wavelen,
                freq / SCALE_FACTOR,  # Apply scaling factor
                scaled  # Apply smooth scaling
            )
        )
        return scaled

    scaled_freqs = torch.vmap(scale_freq)(freqs)

    return scaled_freqs

def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0, use_scaled: bool = False, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2, dtype=dtype, device=device)[:(dim // 2)] / dim))
    if use_scaled:
        freqs = apply_scaling(freqs)

    t = torch.arange(end, dtype=dtype, device=device).unsqueeze(1)  # Shape: (end, 1)
    freqs = freqs.unsqueeze(0)  # Shape: (1, dim//2)
    freqs = t * freqs  # Broadcasting to shape: (end, dim//2)
    return torch.exp(1j * freqs)

def build_attn_mask(seqlen: int, start_pos: int) -> torch.Tensor:
    mask = None
    if seqlen > 1:
        mask = torch.full((seqlen, seqlen), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        mask = torch.hstack([torch.zeros((seqlen, start_pos)), mask]).to(torch.float32).to(device)
    else:
        raise ValueError("seqlen <= 1")
    return mask

def generate(
    xfmr_weights: XfmrWeights,
    model_params: ModelParams,
    tokenizer: Tokenizer,
    prompt: str,
    max_tokens: int = 8192,
    stop_tokens: list | torch.Tensor | None = None,
    temperature: float = 1.0,
    stream: bool = True
) -> str:
    """
    Generate text from a prompt using the transformer model.

    Args:
        xfmr_weights: Model weights
        model_params: Model parameters
        tokenizer: Tokenizer instance
        prompt: Input prompt text
        max_tokens: Maximum number of tokens to generate
        stop_tokens: List of token IDs to stop generation
        temperature: Sampling temperature
        stream: Whether to stream output token by token

    Returns:
        Generated text string
    """
    if not stop_tokens: stop_tokens = [128001, 128008, 128009]  # Default stop tokens
    if isinstance(stop_tokens, list): stop_tokens = torch.tensor(stop_tokens, device=device, dtype=torch.int32)

    with torch.inference_mode():
        print("encoding prompt...")
        tokens = torch.tensor([tokenizer.encode(prompt, bos=False, eos=False, allowed_special='all')], dtype=torch.long).to(device)

        # Initialize
        gen_tokens = None
        gen_tokens_list = []
        cur_pos = 0
        bs, seqlen = tokens.shape

        print("initializing model...")
        # Setup attention mask and positional encodings
        attn_mask = build_attn_mask(seqlen, cur_pos)
        freqs_cis = precompute_freqs_cis(model_params.head_dim, model_params.max_seq_len, model_params.rope_theta, model_params.use_scaled_rope)

        # Initialize KV cache
        kvcache = KVCache.new(model_params.n_layers, bs, model_params.max_seq_len, model_params.n_local_kv_heads, model_params.head_dim).to(device)

        # Initial forward pass
        logits, kvcache, _, _ = xfmr(xfmr_weights, model_params, tokens, cur_pos, freqs_cis[:seqlen], kvcache, attn_mask=attn_mask)
        next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True).to(torch.int32)
        gen_tokens = next_token
        gen_tokens_list.append(next_token.item())

        if stream:
            token_text = tokenizer.decode([next_token.item()])  # type: ignore (torch.int32 not recognized as int)
            print(token_text, end='', flush=True)

        cur_pos = seqlen

        # Generation loop
        while cur_pos < max_tokens:
            cur_pos += 1
            logits, kvcache, scores, _ = xfmr(xfmr_weights, model_params, next_token, cur_pos, freqs_cis[cur_pos:cur_pos + 1], kvcache)

            next_token = sample(gen_tokens, logits, scores, temperature)[0]
            gen_tokens = torch.cat((gen_tokens, next_token), dim=1)
            gen_tokens_list.append(next_token.item())

            if stream:
                token_text = tokenizer.decode([next_token.item()])  # type: ignore (torch.int32 not recognized as int)
                print(token_text, end='', flush=True)

            if torch.isin(next_token, stop_tokens).any():
                print("<|stop|>", end='', flush=True)
                break

        output_text = tokenizer.decode(gen_tokens_list)

        return output_text
