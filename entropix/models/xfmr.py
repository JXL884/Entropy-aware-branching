import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from entropix.config import DEFAULT_MASK_VALUE, LayerWeights, ModelParams, XfmrWeights

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_weights(ckpt_dir: Path, n_layers: int, device: torch.device) -> XfmrWeights:
    w = {}
    layer_weights = []
    with torch.inference_mode():
        for file in ckpt_dir.glob("*.npy"):
            name = '.'.join(str(file).split('/')[-1].split('.')[:-1])
            # jax_weight = jnp.load(file=file, mmap_mode='r', allow_pickle=True)
            # np_weight = np.array(jax_weight).astype(np.float32)
            np_weight = np.load(file=file, mmap_mode='r', allow_pickle=True).astype(np.float32)
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

class KVCache(nn.Module):
    def __init__(self, layers: int, bsz: int, max_seq_len: int, kv_heads: int, head_dim: int):
        super(KVCache, self).__init__()
        # Initialize k and v as buffers to ensure they're part of the module state
        self.register_buffer('k', torch.zeros((layers, bsz, max_seq_len, kv_heads, head_dim), dtype=torch.bfloat16, device=device))
        self.register_buffer('v', torch.zeros((layers, bsz, max_seq_len, kv_heads, head_dim), dtype=torch.bfloat16, device=device))

    @classmethod
    def new(cls, layers: int, bsz: int, max_seq_len: int, kv_heads: int, head_dim: int) -> 'KVCache':
        """Creates a new KVCache instance with initialized k and v tensors."""
        return cls(layers, bsz, max_seq_len, kv_heads, head_dim)

    def update(self, xk: torch.Tensor, xv: torch.Tensor, layer_idx: int, cur_pos: int, n_rep: int):
        """
        Updates the cache with new key and value tensors.

        Args:
            xk (torch.Tensor): New key tensor to insert. Shape should align with (bsz, insert_len, kv_heads, head_dim).
            xv (torch.Tensor): New value tensor to insert. Shape should align with (bsz, insert_len, kv_heads, head_dim).
            layer_idx (int): The index of the layer to update.
            cur_pos (int): The current position in the sequence to start inserting.
            n_rep (int): The number of times to repeat the keys and values along the sequence dimension.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - keys: Updated or repeated keys tensor.
                - values: Updated or repeated values tensor.
        """
        # Ensure xk and xv have the correct device and dtype
        xk = xk.to(self.k.dtype)
        xv = xv.to(self.v.dtype)

        # Update the k and v tensors in the specified layer and position
        insert_len = xk.size(1)  # Assuming xk shape is (bsz, insert_len, kv_heads, head_dim)
        self.k[layer_idx, :, cur_pos:cur_pos + insert_len, :, :] = xk
        self.v[layer_idx, :, cur_pos:cur_pos + insert_len, :, :] = xv

        if cur_pos == 0:
            # If inserting at the beginning, repeat the new keys and values
            keys = xk.repeat_interleave(n_rep, dim=2)
            values = xv.repeat_interleave(n_rep, dim=2)
        else:
            # Otherwise, repeat the existing keys and values from the cache
            keys = self.k[layer_idx].repeat_interleave(n_rep, dim=2)
            values = self.v[layer_idx].repeat_interleave(n_rep, dim=2)

        return keys, values, self

    def clear(self):
        """Resets the k and v caches to zeros."""
        self.k.zero_()
        self.v.zero_()

def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return w * (x * torch.rsqrt(torch.pow(x, 2).mean(-1, keepdim=True) + eps))

def apply_rotary_emb( xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    reshape_xq = xq.float().reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.float().reshape(*xk.shape[:-1], -1, 2)
    xq_ = torch.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    xk_ = torch.complex(reshape_xk[..., 0], reshape_xk[..., 1])
    xq_out = xq_ * freqs_cis.unsqueeze(0).unsqueeze(2)
    xk_out = xk_ * freqs_cis.unsqueeze(0).unsqueeze(2)
    xq_out = torch.stack((xq_out.real, xq_out.imag), dim=-1).reshape(*xq_out.shape[:-1], -1)
    xk_out = torch.stack((xk_out.real, xk_out.imag), dim=-1).reshape(*xk_out.shape[:-1], -1)
    return xq_out.to(dtype), xk_out.to(dtype)

def attention(x: torch.Tensor, layer_weights: LayerWeights, model_params, cur_pos: int, layer_idx: int, freqs_cis: torch.Tensor, kvcache: KVCache, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, KVCache, torch.Tensor]:
    if x.dim() == 2: # add batch dimension to 2d input
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
    attn_stats = AttnStats.new(bsz=tokens.shape[0], n_layers=model_params.n_layers, n_heads=model_params.n_local_heads)
    for i in range(model_params.n_layers):
        norm_x = rms_norm(h, xfmr_weights.layer_weights[i].attention_norm)
        h_attn, kvcache, scores = attention(norm_x, xfmr_weights.layer_weights[i], model_params, cur_pos, i, freqs_cis, kvcache, attn_mask=attn_mask)
        attn_stats = attn_stats.update(scores[:, :, -1, :], i)
        h = h + h_attn
        h = h + feed_forward(rms_norm(h, xfmr_weights.layer_weights[i].ffn_norm), xfmr_weights.layer_weights[i])
    logits = F.linear(rms_norm(h, xfmr_weights.norm), xfmr_weights.output)
    return logits, kvcache, scores, attn_stats
