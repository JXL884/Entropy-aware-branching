import json
import logging
import math, random
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Generator, NamedTuple, Optional, Tuple
import copy
import re
import jax.numpy as jnp
import numpy as np
from scipy.special import kv
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich import print as rprint
from openai import OpenAI
import openai
from typing import List
from entropix.config import DEFAULT_MASK_VALUE, SamplerConfig, SamplerState, STATE_COLOR_MAP
from entropix.kvcache import KVCache
from entropix.sampler import sample
from entropix.tokenizer import Tokenizer, Message
from entropix.metrics import AttnMetrics, TokenMetrics, calculate_metrics
from entropix.PRM import process_response
from typing import *

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

################################################################################
#                                    Types                                     #
################################################################################

class LayerWeights(NamedTuple):
    # Attention weights + biases
    wq: torch.Tensor
    bq: Optional[torch.Tensor]
    wk: torch.Tensor
    bk: Optional[torch.Tensor]
    wv: torch.Tensor
    bv: Optional[torch.Tensor]
    wo: torch.Tensor
    bo: Optional[torch.Tensor]

    # Feed-forward weights + biases
    w1: torch.Tensor
    b1: Optional[torch.Tensor]
    w2: torch.Tensor
    b2: Optional[torch.Tensor]
    w3: torch.Tensor
    b3: Optional[torch.Tensor]

    # Layer norms
    ffn_norm: torch.Tensor
    attention_norm: torch.Tensor


class XfmrWeights(NamedTuple):
    tok_embeddings: torch.Tensor
    norm: torch.Tensor
    output: torch.Tensor
    layer_weights: list[LayerWeights]

class ModelParams(NamedTuple):
    name: str
    dim: int
    n_layers: int
    n_local_heads: int
    n_local_kv_heads: int
    head_dim: int
    max_position_embeddings: int
    rope_theta: float
    use_scaled_rope: bool
    hf_id: str | None = None

class Model(NamedTuple):
    weights: XfmrWeights
    params: ModelParams
    tokenizer: Tokenizer

@dataclass
class GenerationData:
    prompt: str
    response: str
    tokens: list[str]
    messages: list[Message]
    branches: list[list[dict]]
    metrics: list[TokenMetrics]
    sampler_cfg: SamplerConfig
    sampler_states: list[SamplerState]
    branch_count: int = 0
    branch_choices: List[int] = field(default_factory=list)
    branch_pairwise_similarities: List[List[float]] = field(default_factory=list)

    def to_dict(self):
        return {
            "prompt": self.prompt,
            "response": self.response,
            "tokens": self.tokens,
            "messages": [m.model_dump() for m in self.messages],
            "branches": self.branches,
            "metrics": [asdict(m) for m in self.metrics],
            "sampler_cfg": self.sampler_cfg.model_dump(),
            "sampler_states": [s.name for s in self.sampler_states],
            "branch_count": self.branch_count,
            "branch_choices": self.branch_choices,
            "branch_pairwise_similarities": self.branch_pairwise_similarities,
        }

    # def save(self, fp: str):
    #     with open(fp, "w") as f:
    #         s = json.dumps(self.to_dict())
    #         f.write(s)

    def save(self, fp: str):
        dir_path = os.path.dirname(fp)  # Extract the directory path
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path) 

        with open(fp, "w") as f:
            s = json.dumps(self.to_dict())
            f.write(s)


    @classmethod
    def load(cls, fp: str):
        with open(fp, 'rb') as f:
            data = json.load(f)
        defaults = {"branches": [], "metrics": [], "messages": [], "tokens": [], "sampler_states": [], "prompt": "", "response": ""}
        for k, default in defaults.items():
            if k not in data:
                logging.warning(f"Missing field '{k}' in loaded data, using default: {default}")
                data[k] = default
        data["metrics"] = [TokenMetrics(**m) for m in data["metrics"]]
        data["messages"] = [Message(**m) for m in data["messages"]]
        data["sampler_cfg"] = SamplerConfig(**data["sampler_cfg"])
        data["sampler_states"] = [SamplerState[name] for name in data["sampler_states"]]
        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        defaults = {"branches": [], "metrics": [], "messages": [], "tokens": [], "sampler_states": [], "prompt": "", "response": "", "branch_count": 0, "branch_choices": [], "branch_pairwise_similarities": []}
        for k, default in defaults.items():
            if k not in data:
                logging.warning(f"Missing field '{k}' in loaded data, using default: {default}")
                data[k] = default
        data["metrics"] = [TokenMetrics(**m) for m in data["metrics"]]
        data["messages"] = [Message(**m) for m in data["messages"]]
        data["sampler_cfg"] = SamplerConfig.from_dict(data["sampler_cfg"])
        data["sampler_states"] = [SamplerState[name] for name in data["sampler_states"]]
        return cls(**data)

################################################################################
#                                   Weights                                    #
################################################################################
def load_weights(ckpt_dir: Path | str, model_cfg: ModelParams) -> XfmrWeights:
    print(f"Loading weights from {ckpt_dir}...")
    if isinstance(ckpt_dir, str): ckpt_dir = Path(ckpt_dir)
    w = {}
    layer_weights = []
    sep = '\\' if os.name == 'nt' else '/'
    with torch.inference_mode():
        for file in ckpt_dir.glob("*.npy"):
            name = '.'.join(str(file).split(sep)[-1].split('.')[:-1])
            jax_weight = jnp.load(file=file, mmap_mode='r', allow_pickle=True)
            np_weight = np.array(jax_weight).astype(np.float32)
            weight = torch.from_numpy(np_weight).to(torch.float32).to(device)
            w[name] = weight.to(device)
        for i in range(model_cfg.n_layers):
            # For convenience, define a helper to fetch a key (and return None if missing)
            def get_w(key):
                return w.get(key, None)

            att_wq = get_w(f'layers.{i}.attention.wq.weight')
            att_bq = get_w(f'layers.{i}.attention.wq.bias')
            att_wk = get_w(f'layers.{i}.attention.wk.weight')
            att_bk = get_w(f'layers.{i}.attention.wk.bias')
            att_wv = get_w(f'layers.{i}.attention.wv.weight')
            att_bv = get_w(f'layers.{i}.attention.wv.bias')
            att_wo = get_w(f'layers.{i}.attention.wo.weight')
            att_bo = get_w(f'layers.{i}.attention.wo.bias')  # Qwen may or may not have this

            ff_w1 = get_w(f'layers.{i}.feed_forward.w1.weight')
            ff_b1 = get_w(f'layers.{i}.feed_forward.w1.bias')
            ff_w2 = get_w(f'layers.{i}.feed_forward.w2.weight')
            ff_b2 = get_w(f'layers.{i}.feed_forward.w2.bias')
            ff_w3 = get_w(f'layers.{i}.feed_forward.w3.weight')
            ff_b3 = get_w(f'layers.{i}.feed_forward.w3.bias')

            ln_ffn = w[f'layers.{i}.ffn_norm.weight']
            ln_att = w[f'layers.{i}.attention_norm.weight']

            # 3) Build the layer’s NamedTuple
            layer_weights.append(
                LayerWeights(
                    wq=att_wq, bq=att_bq,
                    wk=att_wk, bk=att_bk,
                    wv=att_wv, bv=att_bv,
                    wo=att_wo, bo=att_bo,

                    w1=ff_w1, b1=ff_b1,
                    w2=ff_w2, b2=ff_b2,
                    w3=ff_w3, b3=ff_b3,

                    ffn_norm=ln_ffn,
                    attention_norm=ln_att,
                )
            )

        xfmr_weights = XfmrWeights(tok_embeddings=w['tok_embeddings.weight'], norm=w['norm.weight'], output=w['output.weight'], layer_weights=layer_weights)
        return xfmr_weights

################################################################################
#                                 Attention                                    #
################################################################################
def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (w * x).to(input_dtype)

import torch
from typing import Tuple, Optional

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

# def attention(
#     x: torch.Tensor,
#     layer_weights: LayerWeights,
#     model_params,
#     cur_pos: int,
#     layer_idx: int,
#     freqs_cis: torch.Tensor,
#     kvcache: KVCache,
#     attn_mask: Optional[torch.Tensor] = None
# ) -> Tuple[torch.Tensor, KVCache, torch.Tensor]:
#     if x.dim() == 2:  # add batch dimension to 2d input
#         bs = 1
#         seq_len, dim = x.shape
#         x = x.unsqueeze(0)
#     else:
#         bs, seq_len, dim = x.shape
#     n_rep = model_params.n_local_heads // model_params.n_local_kv_heads
#     xq = F.linear(x, layer_weights.wq).view(bs, seq_len, model_params.n_local_heads, model_params.head_dim)
#     xk = F.linear(x, layer_weights.wk).view(bs, seq_len, model_params.n_local_kv_heads, model_params.head_dim)
#     xv = F.linear(x, layer_weights.wv).view(bs, seq_len, model_params.n_local_kv_heads, model_params.head_dim)
#     xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis, dtype=xq.dtype)
#     keys, values, kvcache = kvcache.update(xk, xv, layer_idx, cur_pos, n_rep)
#     xq = xq.permute(0, 2, 1, 3)  # (bs, n_heads, seqlen, head_dim)
#     keys = keys.permute(0, 2, 3, 1)  # (bs, n_heads, head_dim, cache_len + seqlen)
#     values = values.permute(0, 2, 1, 3)  # (bs, n_heads, cache_len + seqlen, head_dim)
#     scores = torch.matmul(xq, keys)
#     pre_scores = scores / math.sqrt(model_params.head_dim)
#     scores = pre_scores.to(torch.float32)  # Always do attention softmax at float32
#     if cur_pos == 0: scores = scores + attn_mask
#     mask = torch.where(scores != 0.0, scores, DEFAULT_MASK_VALUE)
#     padded_logits = torch.where((mask >= DEFAULT_MASK_VALUE * 0.5), scores, DEFAULT_MASK_VALUE)
#     scores = F.softmax(padded_logits, dim=-1).to(x.dtype)
#     output = torch.matmul(scores.to(values.dtype), values)
#     output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
#     out = F.linear(output, layer_weights.wo)
#     # If input was 2D, remove the batch dimension from the output
#     if dim == 2: out = out.squeeze(0)
#     return out, kvcache, pre_scores

def feed_forward(x: torch.Tensor, layer_weights: LayerWeights) -> torch.Tensor:
    dtype = layer_weights.w1.dtype
    x = x.to(dtype)
    # Apply gate_proj and activate it
    gate_proj = F.linear(x, layer_weights.w1)
    activated_gate = F.silu(gate_proj)
    
    # Apply up_proj
    up_proj = F.linear(x, layer_weights.w3)
    
    # Element-wise multiplication
    gated_up = activated_gate * up_proj
    
    # Apply down_proj
    down_proj = F.linear(gated_up, layer_weights.w2)
    
    return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    if query.shape[1] != module.num_heads:
        # i.e., if dimension 1 is seq_len rather than num_heads, transpose them
        query = query.transpose(1, 2)
        key   = key.transpose(1, 2)
        value = value.transpose(1, 2)

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class QwenAttention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.dim
        self.num_heads = config.n_local_heads
        self.num_kv_heads = config.n_local_kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = getattr(config, 'attention_dropout', 0.0)
        self.is_causal = True

        # Linear projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Sliding window (Qwen-specific)
        self.use_sliding_window = (
            getattr(config, 'use_sliding_window', False) and
            layer_idx >= getattr(config, 'max_window_layers', 0)
        )
        self.sliding_window = getattr(config, 'sliding_window', None) if self.use_sliding_window else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        cache_position: Optional[torch.LongTensor] = None,
        cur_pos: Optional[int] = None,
        output_attentions: bool = False,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Any], Optional[torch.Tensor]]:
        
        input_shape = hidden_states.shape[:-1]
        
        # Handle 2D input by adding a batch dimension
        input_was_2d = (hidden_states.dim() == 2)
        if input_was_2d:
            hidden_states = hidden_states.unsqueeze(0)
        hidden_states = hidden_states.to(self.q_proj.weight.dtype)
        bs, seq_len, _ = hidden_states.shape

        # Compute query, key, value projections.
        # Using .view() to reshape into (bs, seqlen, n_heads, head_dim)
        query = self.q_proj(hidden_states).view(bs, seq_len, self.num_heads, self.head_dim)
        key = self.k_proj(hidden_states).view(bs, seq_len, self.num_kv_heads, self.head_dim)
        value = self.v_proj(hidden_states).view(bs, seq_len, self.num_kv_heads, self.head_dim)

        # Apply rotary embeddings.
        query, key = apply_rotary_emb(query, key, freqs_cis=freqs_cis)

        sliding_window = None

        attn_output, attn_weights = eager_attention_forward(
            self,
            query,
            key,
            value,
            attention_mask,
            scaling=self.scaling,
            sliding_window=sliding_window,  # main diff with Llama
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if output_attentions:
            return attn_output, past_key_value, attn_weights
        return attn_output, past_key_value, None

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
) -> Tuple[torch.Tensor, KVCache, torch.Tensor, AttnMetrics]:
    h = xfmr_weights.tok_embeddings[tokens]
    attn_stats = AttnMetrics.new(bsz=tokens.shape[0], n_layers=model_params.n_layers, n_heads=model_params.n_local_heads, device=device)
    for i in range(model_params.n_layers):
        norm_x = rms_norm(h, xfmr_weights.layer_weights[i].attention_norm)
        attn_layer = QwenAttention(config=model_params, layer_idx=i)
        h_attn, kvcache, scores = attn_layer(
            hidden_states=norm_x,
            freqs_cis=freqs_cis,
            attention_mask=attn_mask,
            past_key_value=kvcache,
            cache_position=None,  # Set this appropriately if your cache requires it
            cur_pos=cur_pos,
            output_attentions=True  # Set to False if you do not need to return attention scores
        )
        attn_stats = attn_stats.update(scores[:, :, -1, :], i)
        h = h + h_attn
        # Feed-forward block with Qwen2MLP structure
        h = h + feed_forward(
            rms_norm(h, xfmr_weights.layer_weights[i].ffn_norm),
            xfmr_weights.layer_weights[i],
        )
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

def precompute_freqs_cis(dim: int, end: int, theta: float = 1000000.0, use_scaled: bool = False, dtype: torch.dtype = torch.float32) -> torch.Tensor:
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

################################################################################
#                                 Branches                                     #
################################################################################

@dataclass
class Branch:
    tokens: torch.Tensor | list
    kvcache: KVCache
    cur_pos: int
    tokens_text: list[str] = field(default_factory=list)
    metrics: list[TokenMetrics] = field(default_factory=list)
    sampler_states: list[SamplerState] = field(default_factory=list)

    def to_dict(self):
        return {
            "tokens": [t.item() for t in self.tokens],
            "tokens_text": self.tokens_text,
            "metrics": [asdict(m) for m in self.metrics],
            "sampler_states": [s.name for s in self.sampler_states],
        }

def should_stop_branch(token_text, token_context, metrics):
    BRANCH_STOP_TOKENS = {".", ".\n", "!", "?", ";", ":", "{", "}", "\n\n", ".\n\n", ":\n\n"}

    if token_text in BRANCH_STOP_TOKENS:
        if token_text == ".":
            # Special handling for ".", check if the previous token is a digit
            if token_context and token_context[-1].isdigit():
                return False  # It's part of a number
        return True
    return False

def send_api_message(messages: list[Message]):
    api_key = os.getenv("OPENROUTER_API_KEY")
    assert api_key is not None, "OPENROUTER_API_KEY environment variable not set"
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    completion = client.chat.completions.create(
        # https://openrouter.ai/models
        model="meta-llama/llama-3.3-70b-instruct",
        messages=messages  # type: ignore
    )
    eval = completion.choices[0].message.content
    if eval is None: eval = ""
    return eval

def _generate_branches(
    model,
    next_token,
    kvcache,
    cur_pos,
    logits,
    metrics,
    stop_tokens,
    max_tokens,
    sampler_cfg,
    print_stream,
) -> list[Branch]:
    sampler_state = SamplerState.BRANCHING
    branches = []
    for i, branch_token in enumerate(next_token[0]):
        branch_token = branch_token.unsqueeze(0)
        token_text = model.tokenizer.decode([branch_token.item()])  # type: ignore (torch.int32 not recognized as int)
        prefix = "├─" if i < len(next_token[0]) - 1 else "└─"
        if print_stream: rprint(f"\n[{STATE_COLOR_MAP[sampler_state]}]{prefix} {token_text.replace('\n', '\\n')}[/]", end='')
        branch_pos = cur_pos + 1
        # kvcache = kvcache.cpu()
        branch_kvcache = copy.deepcopy(kvcache)
        branch_gen_logits = [logits]
        branch_gen_metrics = [metrics]
        branch_gen_tokens = [branch_token]
        branch_gen_tokens_text = [token_text]
        branch_sampler_states = [sampler_state]
        if not torch.isin(branch_token, stop_tokens).any():
            while branch_pos < max_tokens:
                # branch_logits, branch_kvcache, branch_scores, _ = xfmr(
                #     model.weights, model.params, branch_token, branch_pos, freqs_cis[branch_pos:branch_pos + 1], branch_kvcache, attn_mask=None
                # )
                if branch_token.dim() == 1:
                    branch_token = branch_token.unsqueeze(0)  # [1] -> [1,1]
                #print("branch_token", branch_token.shape, branch_token)
                outputs = model.weights(
                    input_ids=branch_token,         # shape [1,1] (the new token)
                    past_key_values=branch_kvcache,    # the branch's KV state
                    use_cache=True,
                    output_attentions=True
                )

                # -- 2) Extract model outputs
                branch_logits = outputs.logits               # shape [1,1,vocab_size]
                branch_scores = outputs.attentions[-1]       # final layer attention
                branch_kvcache = outputs.past_key_values 

                branch_gen_logits.append(branch_logits)
                branch_metrics = calculate_metrics(branch_logits, branch_scores)
                branch_gen_metrics.append(branch_metrics)
                branch_token, branch_sampler_state = sample(branch_logits, branch_scores, branch_metrics, sampler_cfg, can_branch=False)
                branch_gen_tokens.append(branch_token)
                branch_token_text = model.tokenizer.decode([branch_token.item()])  # type: ignore (torch.int32 not recognized as int)
                branch_gen_tokens_text.append(branch_token_text)
                branch_sampler_states.append(branch_sampler_state)
                branch_pos += 1
                if print_stream:
                    rprint(f"[{STATE_COLOR_MAP[branch_sampler_state]}]{branch_token_text.replace('\n', '\\n')}[/]", end='')
                if torch.isin(branch_token, stop_tokens).any() or branch_pos >= max_tokens: break

                token_context = branch_gen_tokens_text[:-1]
                stop = should_stop_branch(branch_token_text, token_context, branch_metrics)
                if stop:
                    break
                if branch_pos >= max_tokens:
                    break
        branches.append(
            Branch(
                tokens=branch_gen_tokens,
                kvcache=branch_kvcache,
                cur_pos=branch_pos,
                tokens_text=branch_gen_tokens_text,
                metrics=branch_gen_metrics,
                sampler_states=branch_sampler_states,
            )
        )
    return branches

def eval_branches(branches, messages, response, model, sampler_cfg):
    analysis_prompt_sys = (
        "You are an expert evaluator assessing reasoning chains. "
        "Here're several generated candidate branch completions below. "
        "Please choose the most correct and relevant one for the conversation to continue with:\n\n"
    )
    analysis_prompt = ""
    for m in messages:
        if m.role == "user":
            analysis_prompt += f"{m.role}: {m.content}\n"
    analysis_prompt += "\n"

    analysis_prompt += "Previously generated tokens:\n" + response + "\n\n"
    for i, b in enumerate(branches):
        completion_text = "".join(b.tokens_text)
        analysis_prompt += f"branch {i}:\n{completion_text}\n\n"

    analysis_prompt += "Which candidate branch number is the most relevant and cohere one to continue generatin with? Please think step by step then put your final answer in {branch }. For example: {branch 2}"

    analysis_messages = [Message(role="system", content=analysis_prompt_sys), Message(role="user", content=analysis_prompt)]

    print(analysis_messages)
    if sampler_cfg.self_feedback:
        decision = generate(
                messages=analysis_messages,
                model=model,
                sampler_cfg=sampler_cfg,
                max_tokens=500,
                print_stream=True,
                apply_chat_template=True,
                allow_branching=False,  # Don't allow branching on self-feedback
        )
        decision_response = decision.response.strip()
    else:
        feedbacks = send_api_message(analysis_messages)
        print(feedbacks)
        decision_response = feedbacks.strip()

    # Extract the content inside the {}
    match = re.findall(r'\{(.*?)\}', decision_response)
    if match:
        answer_content = match[-1].strip()
        number_match = re.search(r'\b(\d+)\b', answer_content)
        if number_match:
            chosen_index = int(number_match.group(1))
        else:
            print("Failed to find a number inside the {}. Defaulting to candidate 0.")
            chosen_index = 0
    else:
        print("Failed to find {} in the response. Defaulting to candidate 0.")
        chosen_index = 0

    return chosen_index

def score_branch(branches, messages, response, score_model):
    branch_responses = []
    for i, branch in enumerate(branches):
        completion_text = "".join(branch.tokens_text)
        branch_responses.append(f"branch {i}: {completion_text}")

    samples = branch_responses
    analysis_prompt = ""
    for m in messages:
        if m.role == "user":  
            analysis_prompt += f"{m.role}: {m.content}\n"
    analysis_prompt += "\n"

    analysis_prompt += response

    processed_sample = process_response(analysis_prompt, samples, score_model)
    chosen_index = processed_sample["step_scores"].index(max(processed_sample["step_scores"]))

    return chosen_index

def get_openai_embeddings(
    texts: list[str], 
    model_name: str = "text-embedding-3-large"
) -> list[list[float]]:
    """
    Returns a list of embedding vectors (list of floats) for each text in `texts`.
    Uses OpenAI's text-embedding-3-large model by default. 
    """
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key is not None, "OPENAI_API_KEY environment variable not set"
    client = OpenAI(api_key=api_key)
    embeddings = []
    for text in texts:
        text = text.replace("\n", " ")

        response = client.embeddings.create(
            input=[text], 
            model=model_name
        )

        embedding = response.data[0].embedding
        embeddings.append(embedding)
    return embeddings

def pairwise_cosine_similarity(embeddings: list[list[float]]) -> np.ndarray:
    """
    Given a list of embeddings [num_texts x embedding_dim],
    return the NxN pairwise cosine similarity matrix.
    """
    arr = np.array(embeddings)  # shape: (N, embedding_dim)
    # L2-norm for each row
    norms = np.linalg.norm(arr, axis=1, keepdims=True)  # shape: (N, 1)
    arr_normed = arr / (norms + 1e-12)  # avoid zero division
    # Pairwise dot product
    sim_matrix = arr_normed @ arr_normed.T  # shape: (N, N)
    return sim_matrix


def _generate(
    messages: list[Message] | list[dict[str, str]] | str,  # type: ignore -> allow definition to be overriden after type conversion
    model: Model,
    score_model : Model,
    sampler_cfg: SamplerConfig | None = None,
    max_tokens: int | None = None,
    print_stream: bool = False,
    apply_chat_template: bool = True,
    allow_branching: bool = True,
    feedback_provider: str = "PRM",
    random_select: bool = False,
    calculate_sim: bool = False
) -> Generator[Tuple[Optional[str], Optional[TokenMetrics], Optional[SamplerState], Optional[GenerationData]], None, None]:
    """
    Core function to generate text using the transformer model and stream the response out.

    Args:
        messages: Input messages or a string prompt
        model: Model to use for generation
        sampler_cfg: Sampler configuration
        max_tokens: Optional, defaults None. Maximum number of tokens to generate, or the max sequence length of the model if None.
        print_stream: Optional, default False. Flag to print the generated tokens to the console
        apply_chat_template: Optional, default True. Flag to apply the chat template to the input messages
        allow_branching (bool): Whether branching is allowed.
        feedback_provider (str): The feedback provider, either "llama3.3" or "PRM".

    Yields:
        Tuple of (generated token text, token metrics, sampler state, complete Generation object (at the last token only))
    """
    # If the tokenizer has 'stop_token_ids', use them (e.g. LLaMA).
    if hasattr(model.tokenizer, "stop_token_ids"):
        stop_ids = model.tokenizer.stop_token_ids
    # Otherwise, try using eos_token_id (often set to <|endoftext|>)
    elif (hasattr(model.tokenizer, "eos_token_id") 
          and model.tokenizer.eos_token_id is not None):
        stop_ids = [model.tokenizer.eos_token_id]
    else:
        # Fallback: either define your own or skip the concept of a single "stop token"
        # If you know <|endoftext|> is ID 151645, you can do:
        stop_ids = [151645]  # Qwen's <|endoftext|> ID

    stop_tokens = torch.tensor(stop_ids, device=device, dtype=torch.int32)
    if max_tokens is None or max_tokens > model.params.max_position_embeddings: max_tokens = model.params.max_position_embeddings
    if sampler_cfg is None:
        logging.warning("No sampler config provided, using default config")
        sampler_cfg = SamplerConfig()

    if isinstance(messages, str):
        prompt = messages
        messages = [Message(role="system", content=prompt)]
        logging.warning("entropix.model._generate: prompt passed as a string, cannot save messages to output GenerationData.")
    elif isinstance(messages, list) and isinstance(messages[0], dict):  # convert list[dict] to list[Message] so all messages are validated
        messages = [Message(**m) if not isinstance(m, Message) else m for m in messages]  # type: ignore
    assert isinstance(messages, list) and all(isinstance(m, Message) for m in messages)
    messages: list[Message] = messages  # type: ignore
    if apply_chat_template: prompt = model.tokenizer.apply_chat_template(messages)

    if print_stream:
        print()
        for state, color in STATE_COLOR_MAP.items():
            rprint(f"[{color}]■[/] [dim]{state.value}[/]")
        print()

    with torch.inference_mode():
        # print("prompt:", prompt)
        tokens = torch.tensor([prompt], dtype=torch.long).to(device)
        # print("tokens:", tokens)
        bs, seqlen = tokens.shape
        cur_pos = 0
        # freqs_end = seqlen

        # attn_mask = build_attn_mask(seqlen, cur_pos)
        # freqs_cis = precompute_freqs_cis(model.params.head_dim, model.params.max_position_embeddings, model.params.rope_theta, model.params.use_scaled_rope)
        # kvcache = KVCache.new(model.params.n_layers, bs, model.params.max_position_embeddings, model.params.n_local_kv_heads, model.params.head_dim).to(device)

        next_token = tokens
        gen_tokens = torch.zeros(1, 1, dtype=torch.int32, device=device)
        past_key_values = None
        response = ""
        gen_tokens_text = []
        gen_logits = []
        gen_metrics = []
        gen_branches = []
        sampler_states = []
        branch_count = 0
        branch_choices = []
        all_pairwise_similarities = []

        while cur_pos < max_tokens:
            # attn = attn_mask if cur_pos < seqlen else None
            # logits, kvcache, scores, attn_stats = xfmr(model.weights, model.params, next_token, cur_pos, freqs_cis[cur_pos:freqs_end], kvcache, attn_mask=attn)

            outputs = model.weights(
                input_ids=next_token,  # feed only the last token or the entire sequence
                past_key_values=past_key_values,  # None for first iteration
                use_cache=True,                   # important for incremental generation
                output_attentions=True,          # set True if you want them
                output_hidden_states=False,       # set True if you want them
            )
            # (1) logits
            logits = outputs.logits  # shape [batch_size, seq_len, vocab_size]

            # (2) "kvcache"
            past_key_values = outputs.past_key_values  # Hugging Face's cache

            # (3) "scores"
            # final layer's attention for the last token:
            scores = outputs.attentions[-1]        # shape [bs, n_heads, seq_len, seq_len]
            # scores = final_layer_attn[:, :, -1, :]  # shape [bs, n_heads, 1, seq_len]
            # all layers, keep 'outputs.attentions' as is.
            
            metrics = calculate_metrics(logits, scores)
            next_token, sampler_state = sample(
                logits,
                scores,
                metrics,
                sampler_cfg,
                can_branch=allow_branching and cur_pos >= seqlen,  # NOTE: always disallows branching on the first token
            )

            if sampler_state != SamplerState.BRANCHING:
                gen_logits.append(logits)
                gen_metrics.append(metrics)
                sampler_states.append(sampler_state)
                cur_pos = seqlen if cur_pos < seqlen else cur_pos + 1
                freqs_end = cur_pos + 1
                gen_tokens = torch.cat((gen_tokens, next_token), dim=1)
                token_text = model.tokenizer.decode([next_token.item()])  # type: ignore (torch.int32 not recognized as int)
                gen_tokens_text.append(token_text)
                if torch.isin(next_token, stop_tokens).any(): break
                response += token_text
                if print_stream: rprint(f"[{STATE_COLOR_MAP[sampler_state]}]{token_text}[/]", end='')
                yield token_text, metrics, sampler_state, None
            else:
                branches = _generate_branches(model, next_token, past_key_values, cur_pos, logits, metrics, stop_tokens, max_tokens, sampler_cfg, print_stream)
                gen_branches.append([branch.to_dict() for branch in branches])

                # Increment your branch counter
                branch_count += 1
                
                if random_select:
                    chosen_index = chosen_index = random.randint(0, 4)  # ablation
                else:
                    if feedback_provider == "llama3.3":
                        chosen_index = eval_branches(branches, messages, response, model, sampler_cfg)
                    elif feedback_provider == "PRM":
                        chosen_index = score_branch(branches, messages, response, score_model)
                    else:
                        raise ValueError("Invalid feedback_provider name. Must be 'llama3.3' or 'PRM'.")
                best_branch = branches[chosen_index]

                # Record the choice
                branch_choices.append(chosen_index)


                branch_texts = []
                for b in branches:
                    text = "".join(b.tokens_text)
                    branch_texts.append(text)

                if len(branches) > 1:
                    embeddings = get_openai_embeddings(branch_texts, model_name="text-embedding-3-large")
                    sim_matrix = pairwise_cosine_similarity(embeddings)
                else:
                    # Only 1 branch => trivial 1x1 matrix
                    sim_matrix = np.array([[1.0]])
                all_pairwise_similarities.append(sim_matrix.tolist())    

                for branch in branches:
                    if branch != best_branch:
                        del branch

                next_token = best_branch.tokens[-1]
                kvcache = best_branch.kvcache.to(device)
                cur_pos = best_branch.cur_pos
                freqs_end = cur_pos + 1
                gen_tokens = torch.cat([gen_tokens, torch.tensor(best_branch.tokens, device=device).unsqueeze(0)], dim=1)
                gen_tokens_text.extend(best_branch.tokens_text)
                gen_metrics.extend(best_branch.metrics)
                sampler_states.extend(best_branch.sampler_states)
                branch_response = "".join(best_branch.tokens_text)
                response += branch_response
                if print_stream:
                    rprint(f"\n[{STATE_COLOR_MAP[SamplerState.BRANCHING]}]=>[/]", end='')
                    for state, text in zip(best_branch.sampler_states, best_branch.tokens_text):
                        rprint(f"[{STATE_COLOR_MAP[state]}]{text.replace('\n', '\\n')}[/]", end='')
                if torch.isin(next_token, stop_tokens).any(): break

        if print_stream: print()
        messages.append(Message(role="assistant", content=response))
        gen = GenerationData(
            prompt=prompt,
            response=response,
            tokens=gen_tokens_text,
            messages=messages,
            branches=gen_branches,
            metrics=gen_metrics,
            sampler_cfg=sampler_cfg,
            sampler_states=sampler_states,
            branch_count=branch_count,
            branch_choices=branch_choices,
            branch_pairwise_similarities=all_pairwise_similarities
        )
        yield "", metrics, sampler_state, gen

def stream(
    messages: list[Message] | list[dict[str, str]] | str,
    model: Model,
    sampler_cfg: SamplerConfig | None = None,
    max_tokens: int | None = None,
    print_stream: bool = False,
    apply_chat_template: bool = True,
):
    for token_text, metrics, sampler_state, gen in _generate(
        messages=messages,
        model=model,
        sampler_cfg=sampler_cfg,
        max_tokens=max_tokens,
        print_stream=print_stream,
        apply_chat_template=apply_chat_template,
    ):
        yield token_text, metrics, sampler_state, gen

def generate(
    messages: list[Message] | list[dict[str, str]] | str,
    model: Model,
    score_model: Model,
    sampler_cfg: SamplerConfig | None = None,
    max_tokens: int | None = None,
    print_stream: bool = False,
    apply_chat_template: bool = True,
    allow_branching: bool = True,
    feedback_provider: str = "PRM",
    random_select: bool = False,
    calculate_sim: bool = False
):
    for token_text, metrics, sampler_state, gen in _generate(
        messages=messages,
        model=model,
        score_model = score_model,
        sampler_cfg=sampler_cfg,
        max_tokens=max_tokens,
        print_stream=print_stream,
        apply_chat_template=apply_chat_template,
        allow_branching=allow_branching,
        feedback_provider=feedback_provider,
        random_select=random_select,
        calculate_sim=calculate_sim
    ):
        if gen is not None:
            return gen
    raise RuntimeError("Generation failed to complete")
