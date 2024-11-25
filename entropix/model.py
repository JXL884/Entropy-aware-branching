import json
import logging
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Generator, Literal, NamedTuple, Optional, Tuple
import copy
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F

from entropix.config import DEFAULT_MASK_VALUE, SamplerConfig, SamplerState
from entropix.kvcache import KVCache
from entropix.sampler import sample, branching_sample, adaptive_sample
from entropix.stats import AttnStats, TokenMetrics, calculate_metrics
from entropix.tokenizer import Tokenizer, Message

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

################################################################################
#                                    Types                                     #
################################################################################

class LayerWeights(NamedTuple):
    wq: torch.Tensor
    wk: torch.Tensor
    wv: torch.Tensor
    wo: torch.Tensor
    w1: torch.Tensor
    w2: torch.Tensor
    w3: torch.Tensor
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
    max_seq_len: int
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
    metrics: list[TokenMetrics]
    sampler_cfg: SamplerConfig
    sampler_states: list[SamplerState]

    def to_dict(self):
        return {
            "prompt": self.prompt,
            "response": self.response,
            "tokens": self.tokens,
            "messages": [m.model_dump() for m in self.messages],
            "metrics": [asdict(m) for m in self.metrics],
            "sampler_cfg": asdict(self.sampler_cfg),
            "sampler_states": [s.name for s in self.sampler_states],
        }

    def save(self, fp: str):
        with open(fp, "w") as f:
            s = json.dumps(self.to_dict())
            f.write(s)

    @classmethod
    def load(cls, fp: str):
        with open(fp, 'rb') as f:
            data = json.load(f)
        data["metrics"] = [TokenMetrics(**m) for m in data["metrics"]]
        data["messages"] = [Message(**m) for m in data["messages"]]
        data["sampler_cfg"] = SamplerConfig(**data["sampler_cfg"])
        data["sampler_states"] = [SamplerState[name] for name in data["sampler_states"]]
        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        data["metrics"] = [TokenMetrics(**m) for m in data["metrics"]]
        data["messages"] = [Message(**m) for m in data["messages"]]
        data["sampler_cfg"] = SamplerConfig(**data["sampler_cfg"])
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
    with torch.inference_mode():
        for file in ckpt_dir.glob("*.npy"):
            name = '.'.join(str(file).split('/')[-1].split('.')[:-1])
            jax_weight = jnp.load(file=file, mmap_mode='r', allow_pickle=True)
            np_weight = np.array(jax_weight).astype(np.float32)
            weight = torch.from_numpy(np_weight).to(torch.bfloat16).to(device)
            w[name] = weight.to(device)
        for i in range(model_cfg.n_layers):
            layer_weights.append(
                LayerWeights(
                    wq=w[f'{ckpt_dir}\\layers.{i}.attention.wq.weight'],
                    wk=w[f'{ckpt_dir}\\layers.{i}.attention.wk.weight'],
                    wv=w[f'{ckpt_dir}\\layers.{i}.attention.wv.weight'],
                    wo=w[f'{ckpt_dir}\\layers.{i}.attention.wo.weight'],
                    w1=w[f'{ckpt_dir}\\layers.{i}.feed_forward.w1.weight'],
                    w2=w[f'{ckpt_dir}\\layers.{i}.feed_forward.w2.weight'],
                    w3=w[f'{ckpt_dir}\\layers.{i}.feed_forward.w3.weight'],
                    ffn_norm=w[f'{ckpt_dir}\\layers.{i}.ffn_norm.weight'],
                    attention_norm=w[f'{ckpt_dir}\\layers.{i}.attention_norm.weight'],
                )
            )
        xfmr_weights = XfmrWeights(tok_embeddings=w[f'{ckpt_dir}\\tok_embeddings.weight'], norm=w[f'{ckpt_dir}\\norm.weight'], output=w[f'{ckpt_dir}\\output.weight'], layer_weights=layer_weights)
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
    messages: list[Message] | list[dict[str, str]] | str,  # type: ignore -> allow definition to be overridden after type conversion
    model: Model,
    sampler_cfg: SamplerConfig | None = None,
    max_tokens: int | None = None,
    print_stream: bool = True,
    metrics: bool = True,
    apply_chat_template: bool = True,
) -> GenerationData:
    """
    Generate text using the transformer model with branching.

    Args:
        messages: Input messages or a string prompt
        model: Model to use for generation
        sampler_cfg: Sampler configuration
        max_tokens: Maximum number of tokens to generate
        print_stream: Optional, default False. Flag to print the generated tokens to the console
        metrics: Optional, default True. Flag to calculate and return entropy metrics
        apply_chat_template: Optional, default True. Flag to apply the chat template to the input messages

    Returns:
        GenerationData: A dataclass containing the generated text, tokens, messages, metrics, sampler configuration, and sampler states
    """
    stop_tokens = torch.tensor(model.tokenizer.stop_token_ids, device=device, dtype=torch.int32)
    if max_tokens is None:
        max_tokens = model.params.max_seq_len
    if sampler_cfg is None:
        logging.warning("No sampler config provided, using default config")
        sampler_cfg = SamplerConfig()

    if isinstance(messages, str):
        prompt = messages
        messages = [Message(role="system", content=prompt)]
        logging.warning("entropix.model.generate: prompt passed as a string, cannot save messages to output GenerationData.")
    elif isinstance(messages, list) and isinstance(messages[0], dict):  # convert list[dict] to list[Message] so all messages are validated
        messages = [Message(**m) if not isinstance(m, Message) else m for m in messages]  # type: ignore

    assert isinstance(messages, list) and all(isinstance(m, Message) for m in messages)
    messages: list[Message] = messages  # type: ignore
    if apply_chat_template:
        prompt = model.tokenizer.apply_chat_template(messages)

    breaker = False

    with torch.inference_mode():
        # Initial encoding of the prompt
        initial_tokens = torch.tensor([model.tokenizer.encode(prompt, bos=False, eos=False, allowed_special='all')], dtype=torch.long).to(device)
        bs, seqlen = initial_tokens.shape

        # Initial model state
        attn_mask = build_attn_mask(seqlen, 0)
        freqs_end = seqlen
        freqs_cis = precompute_freqs_cis(model.params.head_dim, model.params.max_seq_len, model.params.rope_theta, model.params.use_scaled_rope)
        initial_kvcache = KVCache.new(model.params.n_layers, bs, model.params.max_seq_len, model.params.n_local_kv_heads, model.params.head_dim).to(device)

        # Initialize branches
        branches = []
        initial_branch = {
            'tokens': initial_tokens,
            'kvcache': initial_kvcache,
            'cur_pos': 0,
            'freqs_end': freqs_end,
            'response': "",
            'gen_tokens_text': [],
            'gen_metrics': [],
            'sampler_states': [],
            'active': True,
            'iterations_left': None,  # Not in initial 5-iteration period
            'can_branch': False,
        }

        branches.append(initial_branch)

        total_iterations = 0

        # Main generation loop
        while total_iterations < max_tokens and any(branch['active'] for branch in branches):
            branches_to_evaluate = []
            new_branches = []

            for branch in branches:
                if not branch['active']:
                    continue  # Skip inactive branches

                tokens = branch['tokens']
                kvcache = branch['kvcache']
                cur_pos = branch['cur_pos']
                freqs_end = branch['freqs_end']
                response = branch['response']

                # Determine if we need to use attention mask
                attn = attn_mask if cur_pos < seqlen else None

                print("cur_pos", cur_pos)
                print("freqs_end", freqs_end)

                # Run the model to get logits and other outputs
                logits, kvcache, scores, attn_stats = xfmr(
                    model.weights,
                    model.params,
                    tokens,   
                    cur_pos,
                    freqs_cis[cur_pos:freqs_end],
                    kvcache,
                    attn_mask=attn,
                )

                # Use branching_sample instead of sample
                next_tokens_list, sampler_state = branching_sample(tokens, logits, scores, sampler_cfg, can_branch=branch['can_branch'])
                
                

                # If branching occurs
                if len(next_tokens_list[0]) > 1 and branch['can_branch']:
                    # Deactivate the current branch
                    #branch['active'] = False

                    print(next_tokens_list)
                    # Create new branches for each token in next_tokens_list
                    for sampled_token in next_tokens_list[0]:
                        print("sampled_token", sampled_token)
                        new_branch = {
                            'tokens': torch.cat([tokens, sampled_token.unsqueeze(0).unsqueeze(0)], dim=1),
                            'kvcache': copy.deepcopy(kvcache),
                            'cur_pos': seqlen if cur_pos < seqlen else cur_pos + 1,
                            'freqs_end': (seqlen + 1) if cur_pos < seqlen else cur_pos + 2,
                            'response': response + model.tokenizer.decode(sampled_token.tolist()),
                            'gen_tokens_text': branch['gen_tokens_text'] + [model.tokenizer.decode(sampled_token.tolist())],
                            'gen_metrics': branch['gen_metrics'] + ([calculate_metrics(logits, scores)]),
                            'sampler_states': branch['sampler_states'] + [sampler_state],
                            'active': True,
                            'iterations_left': 4,  # Already generated one token
                            'can_branch': False,   # Cannot branch during initial 5 iterations
                        }
                        new_branches.append(new_branch)
                else:
                    # Continue the current branch
                    print(next_tokens_list)
                    sampled_token = next_tokens_list[0]
                    print("sampled_token", sampled_token)
                    branch['tokens'] = torch.cat([tokens, sampled_token.unsqueeze(0)], dim=1)
                    branch['kvcache'] = kvcache
                    branch['cur_pos'] = seqlen if cur_pos < seqlen else cur_pos + 1
                    branch['freqs_end'] = (seqlen + 1) if cur_pos < seqlen else cur_pos + 2
                    branch['response'] += model.tokenizer.decode(sampled_token.tolist())
                    branch['gen_tokens_text'].append(model.tokenizer.decode(sampled_token.tolist()))
                    branch['gen_metrics'].append(calculate_metrics(logits, scores))
                    branch['sampler_states'].append(sampler_state)

                    # Decrement iterations_left if in initial 5 iterations
                    if branch['iterations_left'] is not None:
                        branch['iterations_left'] -= 1
                        if branch['iterations_left'] == 0:
                            branches_to_evaluate.append(branch)

                    # Allow branching again after initial 5 iterations
                    if branch['iterations_left'] is None:
                        branch['can_branch'] = True

                    new_branches.append(branch)
                    print("new branches: ", len(new_branches))

                    # Check for stop tokens or max tokens
                    if torch.isin(sampled_token, stop_tokens).any() or branch['cur_pos'] >= max_tokens:
                        breaker = True
                    if breaker:
                        break
            if breaker:
                break
            # Evaluate branches after 5 tokens
            if branches_to_evaluate:
                # Calculate average metrics for each branch
                branch_scores = []
                for branch in branches_to_evaluate:
                    # Example: using the average log probability as the score
                    entropies = [m.logits_entropy for m in branch['gen_metrics'][-5:]]  # Last 5 tokens
                    avg_entropy = sum(entropies) / len(entropies)
                    # add average entropy to branch_scores
                    branch_scores.append(avg_entropy)


                # Select the branch index with the best (smallest) average score
                selected_branch = branch_scores.index(min(branch_scores))

                # Deactivate other branches
                for _, branch in enumerate(branches_to_evaluate):
                    if _ != selected_branch:
                        branch['active'] = False
                    else:
                        # Reset iterations_left and allow branching again
                        branch['iterations_left'] = None
                        branch['can_branch'] = True
                # Ensure the selected branch is added back to new_branches
                #new_branches = branches_to_evaluate[selected_branch]

                # check if there's any activate branches
                if not any(branch['active'] for branch in branches):
                    print("No active branches")

            # Update branches
            branches = [branch for branch in new_branches if branch['active']]

            # Increment total iterations
            total_iterations += 1

            # Debugging print statements
            print(f"Iteration {total_iterations}: Active branches = {len(branches)}")

        # Collect the final responses from branches
        final_branches = [branch for branch in branches if not branch['active']]

        print(final_branches.__len__())
        print(branches)

        # If no branches are inactive (e.g., max_tokens reached), collect active branches
        if len(final_branches) == 0:
            final_branches = branches

        # Print the response if needed
        if print_stream:
            print(final_branches[-1]['response'])

        messages.append(Message(role="assistant", content=final_branches[-1]['response']))
        return GenerationData(
            prompt=prompt,
            response=final_branches[-1]['response'],
            tokens=final_branches[-1]['gen_tokens_text'],
            messages=messages,
            metrics=final_branches[-1]['gen_metrics'],
            sampler_cfg=sampler_cfg,
            sampler_states=final_branches[-1]['sampler_states'],
        )


def generate_ori(
    messages: list[Message] | list[dict[str, str]] | str,  # type: ignore -> allow definition to be overriden after type conversion
    model: Model,
    sampler_cfg: SamplerConfig | None = None,
    max_tokens: int | None = None,
    print_stream: bool = False,
    metrics: bool = True,
    apply_chat_template: bool = True,
) -> GenerationData:
    """
    Generate text using the transformer model.

    Args:
        messages: Input messages or a string prompt
        model: Model to use for generation
        sampler_cfg: Sampler configuration
        max_tokens: Maximum number of tokens to generate
        print_stream: Optional, default False. Flag to print the generated tokens to the console
        metrics: Optional, default True. Flag to calculate and return entropy metrics
        apply_chat_template: Optional, default True. Flag to apply the chat template to the input messages

    Returns:
        GenerationData: A dataclass containing the generated text, tokens, messages, metrics, sampler configuration, and sampler states
    """
    stop_tokens = torch.tensor(model.tokenizer.stop_token_ids, device=device, dtype=torch.int32)
    if max_tokens is None: max_tokens = model.params.max_seq_len
    if sampler_cfg is None:
        logging.warning("No sampler config provided, using default config")
        sampler_cfg = SamplerConfig()

    if isinstance(messages, str):
        prompt = messages
        messages = [Message(role="system", content=prompt)]
        logging.warning("entropix.model.generate: prompt passed as a string, cannot save messages to output GenerationData.")
    elif isinstance(messages, list) and isinstance(messages[0], dict): # convert list[dict] to list[Message] so all messages are validated
        messages = [Message(**m) if not isinstance(m, Message) else m for m in messages]  # type: ignore

    assert isinstance(messages, list) and all(isinstance(m, Message) for m in messages)
    messages: list[Message] = messages  # type: ignore
    if apply_chat_template:
        prompt = model.tokenizer.apply_chat_template(messages)

    with torch.inference_mode():
        tokens = torch.tensor([model.tokenizer.encode(prompt, bos=False, eos=False, allowed_special='all')], dtype=torch.long).to(device)
        bs, seqlen = tokens.shape
        cur_pos = 0

        attn_mask = build_attn_mask(seqlen, cur_pos)
        freqs_cis = precompute_freqs_cis(model.params.head_dim, model.params.max_seq_len, model.params.rope_theta, model.params.use_scaled_rope)
        kvcache = KVCache.new(model.params.n_layers, bs, model.params.max_seq_len, model.params.n_local_kv_heads, model.params.head_dim).to(device)

        next_token = tokens
        freqs_end = seqlen
        gen_tokens = torch.zeros(1, 1, dtype=torch.int32, device=device)

        response = ""
        gen_tokens_text = []
        gen_metrics = []
        sampler_states = []

        while cur_pos < max_tokens:
            attn = attn_mask if cur_pos < seqlen else None
            logits, kvcache, scores, attn_stats = xfmr(model.weights, model.params, next_token, cur_pos, freqs_cis[cur_pos:freqs_end], kvcache, attn_mask=attn)
            next_token, sampler_state = sample(gen_tokens, logits, scores, sampler_cfg)

            if metrics:
                gen_metrics.append(calculate_metrics(logits, scores))
                sampler_states.append(sampler_state)

            cur_pos = seqlen if cur_pos < seqlen else cur_pos + 1
            freqs_end = cur_pos + 1

            # print("next token", next_token)
            # print("next token shape", next_token.shape)
            # print("gen token", gen_tokens)
            # print("gen token shape", gen_tokens.shape)

            gen_tokens = torch.cat((gen_tokens, next_token), dim=1)

            token_text = model.tokenizer.decode([next_token.item()])  # type: ignore (torch.int32 not recognized as int)
            gen_tokens_text.append(token_text)

            # break after adding the stop token and its metrics to output but before adding to response text / printing
            if torch.isin(next_token, stop_tokens).any(): break

            if print_stream: print(token_text, end='', flush=True)
            response += token_text
        if print_stream: print()

        messages.append(Message(role="assistant", content=response))
        return GenerationData(
            prompt=prompt,
            response=response,
            tokens=gen_tokens_text,
            messages=messages,
            metrics=gen_metrics,
            sampler_cfg=sampler_cfg,
            sampler_states=sampler_states,
        )


def stream(
    messages: list[Message] | list[dict[str, str]] | str,  # type: ignore -> allow definition to be overriden after type conversion
    model: Model,
    sampler_cfg: SamplerConfig | None = None,
    max_tokens: int | None = None,
    print_stream: bool = False,
    metrics: bool = True,
    apply_chat_template: bool = True,
) -> Generator[Tuple[Optional[str], Optional[TokenMetrics], Optional[SamplerState], Optional[GenerationData]], None, None]:
    """
    Stream generated text using the transformer model.

    Args:
        messages: Input messages or a string prompt
        model: Model to use for generation
        sampler_cfg: Sampler configuration
        max_tokens: Maximum number of tokens to generate
        print_stream: Optional, default False. Flag to print the generated tokens to the console
        metrics: Optional, default True. Flag to calculate and return entropy metrics
        apply_chat_template: Optional, default True. Flag to apply the chat template to the input messages

    Yields:
        Tuple of (generated token text, token metrics, sampler state, complete Generation object (at the last token only))
    """
    stop_tokens = torch.tensor(model.tokenizer.stop_token_ids, device=device, dtype=torch.int32)
    if max_tokens is None:
        max_tokens = model.params.max_seq_len
    if sampler_cfg is None:
        logging.warning("No sampler config provided, using default config")
        sampler_cfg = SamplerConfig()

    if isinstance(messages, str):
        prompt = messages
        messages = [Message(role="system", content=prompt)]
        logging.warning("entropix.model.generate: prompt passed as a string, cannot save messages to output GenerationData.")
    elif isinstance(messages, list) and isinstance(messages[0], dict): # convert list[dict] to list[Message] so all messages are validated
        messages = [Message(**m) if not isinstance(m, Message) else m for m in messages]  # type: ignore

    assert isinstance(messages, list) and all(isinstance(m, Message) for m in messages)
    messages: list[Message] = messages  # type: ignore
    if apply_chat_template:
        prompt = model.tokenizer.apply_chat_template(messages)

    with torch.inference_mode():
        tokens = torch.tensor([model.tokenizer.encode(prompt, bos=False, eos=False, allowed_special='all')], dtype=torch.long).to(device)
        bs, seqlen = tokens.shape
        cur_pos = 0

        attn_mask = build_attn_mask(seqlen, cur_pos)
        freqs_cis = precompute_freqs_cis(model.params.head_dim, model.params.max_seq_len, model.params.rope_theta, model.params.use_scaled_rope)
        kvcache = KVCache.new(model.params.n_layers, bs, model.params.max_seq_len, model.params.n_local_kv_heads, model.params.head_dim).to(device)

        next_token = tokens
        freqs_end = seqlen
        gen_tokens = torch.zeros(1, 1, dtype=torch.int32, device=device)

        response = ""
        gen_tokens_text = []
        gen_metrics = []
        sampler_states = []

        while cur_pos < max_tokens:
            attn = attn_mask if cur_pos < seqlen else None
            logits, kvcache, scores, attn_stats = xfmr(model.weights, model.params, next_token, cur_pos, freqs_cis[cur_pos:freqs_end], kvcache, attn_mask=attn)
            next_token, sampler_state = branching_sample(gen_tokens, logits, scores, sampler_cfg)

            if metrics:
                token_metrics = calculate_metrics(logits, scores)
                gen_metrics.append(token_metrics)
                sampler_states.append(sampler_state)
            else:
                token_metrics = None

            cur_pos = seqlen if cur_pos < seqlen else cur_pos + 1
            freqs_end = cur_pos + 1

            gen_tokens = torch.cat((gen_tokens, next_token), dim=1)

            token_text = model.tokenizer.decode([next_token.item()])  # type: ignore
            gen_tokens_text.append(token_text)

            # break after adding the stop token and its metrics to output but before adding to response text and yielding
            if torch.isin(next_token, stop_tokens).any(): break

            yield token_text, token_metrics, sampler_state, None

        messages.append(Message(role="assistant", content=response))
        gen = GenerationData(
            prompt=prompt,
            response=response,
            tokens=gen_tokens_text,
            messages=messages,
            metrics=gen_metrics,
            sampler_cfg=sampler_cfg,
            sampler_states=sampler_states,
        )
        yield "", token_metrics, sampler_state, gen
