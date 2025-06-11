import json
import logging
import math, random
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Generator, NamedTuple, Optional, Tuple
import copy
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich import print as rprint
from openai import OpenAI
import openai
from typing import List
from transformers import DynamicCache
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
#                                 Inserting                                    #
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

def should_stop_branch(token_text, token_context):
    BRANCH_STOP_TOKENS = {".", ". ", ".\n", "!", "?", "\n\n", ".\n\n"}
    # BRANCH_STOP_TOKENS = {"\n\n"}

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


def rollback_kv_cache_by_one_token(past_key_values):
    """
    Rolls back the KV cache by one token for each layer.
    past_key_values is a transformers.Cache object, it returns a new Cache object.


    Args:
        past_key_values: The past_key_values from a model.
                         Can be a transformers.Cache object, a tuple of 
                         (key_tensor, value_tensor) pairs, or None.
                         K and V tensors are expected to have the sequence length
                         at dimension -2 (e.g., shape [batch, heads, seq_len, dim]).

    Returns:
        A new past_key_values with the last token's state removed
    """

    # Create a new cache object of the same type
    new_cache = type(past_key_values)()
    
    # Roll back each layer
    for layer_idx in range(len(past_key_values.key_cache)):
        key_tensor = past_key_values.key_cache[layer_idx]
        value_tensor = past_key_values.value_cache[layer_idx]
        
        if key_tensor is not None and value_tensor is not None:
            # Remove the last token (sequence dimension is at -2)
            if key_tensor.size(-2) > 0:  # Check if there are tokens to remove
                rolled_back_key = key_tensor[..., :-1, :]
                rolled_back_value = value_tensor[..., :-1, :]
                
                new_cache.update(rolled_back_key, rolled_back_value, layer_idx)
            else:
                # If no tokens to remove, keep empty tensors
                new_cache.update(key_tensor, value_tensor, layer_idx)
        
    return new_cache

def insert_tokens(
    model,
    next_token: torch.Tensor,
    past_key_values,
    logits: torch.Tensor,
    metrics,
    cur_pos: int,
    seqlen: int,
    gen_tokens: torch.Tensor,
    gen_tokens_text: list[str],
    response: str,
    gen_logits: list[torch.Tensor],
    gen_metrics: list,
    sampler_states: list,
    sampler_cfg,
    allow_branching: bool,
    print_stream: bool,
    include_trigger_token: bool,
    insert_text: str
) -> Generator[Tuple[Optional[str], Optional[TokenMetrics], Optional[SamplerState], Optional[GenerationData]], None, None]:
    stop_ids = [151645]
    stop_tokens = torch.tensor(stop_ids, device=device, dtype=torch.int32)

    new_tokens_ids = []
    new_tokens_text = []
    new_metrics = []
    current_past_kv = past_key_values

    # Correctly encode the text snippet.
    # insert_ids = model.tokenizer.apply_chat_template(insert_text, add_generation_prompt=True, tokenize=True, enable_thinking=True)
    insert_ids = model.tokenizer.encode(insert_text, add_special_tokens=False)

    for rid in insert_ids:
        new_tokens_ids.append(rid)
        forced_token = torch.tensor([[rid]], device=device, dtype=torch.int32)
        
        with torch.inference_mode():
            forced_outputs = model.weights(
                input_ids=forced_token,
                past_key_values=current_past_kv,
                use_cache=True,
                output_attentions=True
            )
        
        # Update the state for the next iteration of *this* loop
        current_past_kv = forced_outputs.past_key_values
        
        # Log the results
        token_text = model.tokenizer.decode([rid])
        new_tokens_text.append(token_text)
        
        forced_logits = forced_outputs.logits
        forced_scores = forced_outputs.attentions[-1]
        forced_metrics = calculate_metrics(forced_logits, forced_scores)
        new_metrics.append(forced_metrics)

        if print_stream:
            rprint(f"[{STATE_COLOR_MAP[SamplerState.PAUSE]}]{token_text}[/]", end='')

    return new_tokens_ids, new_tokens_text, new_metrics, current_past_kv

def get_next_step(
    model: Model,
    original_messages: list[Message],
    current_response: str,
    max_new_tokens: int = 500, # Limit the length of the "next step"
) -> str:
    """
    Asks the model to reflect on its current generation and suggest a next step.

    This function performs a separate, self-contained generation loop.

    Args:
        model: The language model instance.
        original_messages: The initial list of messages that started the generation.
        current_response: The text generated by the assistant so far, up to the PAUSE.
        max_new_tokens: The maximum number of tokens to generate for the next step.
        stop_ids: A list of token IDs that should stop the generation.

    Returns:
        A string containing the model's suggested next step.
    """
    thinking: list[int] = [151667] # Qwen's <think>
    stop_thinking: list[int] = [151645, 151668] # Qwen's </think> and stop token
    thinking_tokens = torch.tensor(thinking, device=device, dtype=torch.int32)
    stop_tokens = torch.tensor(stop_thinking, device=device, dtype=torch.int32)

    # 1. Construct the "meta-prompt" for reflection.
    meta_prompt_messages = [
        Message(role="system", content="You are a collaborative AI expert. You are given a conversation history where the last assistant message is an incomplete, step-by-step solution. "
    "Your task is to reflect on the previous solution and continue the solution by generating the next logical step. Do not repeat the previous step "
    # "1.  **Analyze and Verify:** "
    # "   *   Read the entire conversation to understand the user's goal and the solution's progress. "
    # "   *   Critically evaluate the last step taken by the assistant. Is the formula correct? Is the reasoning sound? "
    # "   *   Identify the exact point where the assistant left off. "
    # "2.  **Plan the Next Step:** "
    # "   *   Based on your analysis, determine the immediate next action required to solve the problem. "
    # "   *   For example, if the last step was defining a formula, the next step is likely plugging in the values. If the last step was a calculation, the next step might be interpreting that result or performing the next calculation in the sequence. "
    ),
        # whatever the user message is, we just need to add the user message to the meta-prompt
        Message(role="user", content=original_messages[-1].content),
        Message(role="assistant", content=current_response),
        Message(role="assistant", content="I need to briefly complete the next step ONLY. DO NOT SOLVE THE PROBLEM. Continue from the pre-existing reasoning process. /think")
    ]

    # 2. Tokenize the meta-prompt.
    meta_prompt_ids = model.tokenizer.apply_chat_template(
        meta_prompt_messages,
        add_generation_prompt=True,
        tokenize=True,
        enable_thinking=True
    )
    next_token = torch.tensor([meta_prompt_ids], device=device, dtype=torch.int32)

    # 3. Perform a new, separate generation to get the next step.
    generated_ids = []
    past_key_values = None
    with torch.inference_mode():
        for i in range(max_new_tokens):
            outputs = model.weights(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True
            )
            past_key_values = outputs.past_key_values

            # For this internal generation, we can just use simple argmax sampling.
            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
            next_token = next_token_id.unsqueeze(0)
            # Check for stop token
            if torch.isin(next_token, stop_tokens).any():
                break

            if not torch.isin(next_token, thinking_tokens).any():
                generated_ids.append(next_token_id.item())


    # 4. Decode and return the generated text.
    next_step_text = model.tokenizer.decode(generated_ids).strip()
    return next_step_text

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
    calculate_sim: bool = False,
    do_insert_bos: bool = False,
    do_insert_eos: bool = False,
    want_insert: bool = True,
    enable_thinking: bool = False,
    insert_text: str | None = None
) -> Generator[Tuple[Optional[str], Optional[TokenMetrics], Optional[SamplerState], Optional[GenerationData]], None, None]:
    
    stop_ids = [151645]  # Qwen's <|endoftext|> ID

    stop_tokens = torch.tensor(stop_ids, device=device, dtype=torch.int32)
    if max_tokens is None or max_tokens > model.params.max_position_embeddings:
        max_tokens = model.params.max_position_embeddings
    if sampler_cfg is None:
        logging.warning("No sampler config provided, using default config")
        sampler_cfg = SamplerConfig()

    # Convert messages to a prompt
    if isinstance(messages, str):
        prompt = messages
        messages = [Message(role="system", content=prompt)]
        logging.warning("entropix.model._generate: prompt passed as a string, cannot save messages to output GenerationData.")
    elif isinstance(messages, list) and isinstance(messages[0], dict):
        messages = [Message(**m) if not isinstance(m, Message) else m for m in messages]  # type: ignore
    assert isinstance(messages, list) and all(isinstance(m, Message) for m in messages)
    messages: list[Message] = messages  # type: ignore
    if apply_chat_template:
        print("The prompt is", messages)
        prompt = model.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, enable_thinking=enable_thinking)
        prompt_for_print = model.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=enable_thinking)
        print(prompt_for_print)

    if print_stream:
        print()
        for state, color in STATE_COLOR_MAP.items():
            rprint(f"[{color}]■[/] [dim]{state.value}[/]")
        print()

    #print("The prompt is", prompt)

    with torch.inference_mode():
        tokens = torch.tensor([prompt], dtype=torch.int32).to(device)
        bs, seqlen = tokens.shape

        next_token = tokens
        gen_tokens = torch.zeros(1, 1, dtype=torch.int32, device=device)
        last_pause_step = -9999
        cur_seen_tokens = 0
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
        track_pause = False
        track_end = False

        while cur_seen_tokens < max_tokens:
            outputs = model.weights(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True,
                output_hidden_states=False,
            )

            logits = outputs.logits
            past_key_values = outputs.past_key_values
            cur_seen_tokens = past_key_values.seen_tokens
            scores = outputs.attentions[-1]

            metrics = calculate_metrics(logits, scores)
            num_tokens_so_far = gen_tokens.shape[1]
            next_token, sampler_state = sample(
                logits,
                scores,  
                metrics,
                sampler_cfg,
                can_branch=allow_branching and past_key_values.seen_tokens >= seqlen,
                current_step=past_key_values.seen_tokens,  # new parameter to track the current step
                last_pause_step=last_pause_step
            )
            token_text = model.tokenizer.decode([next_token.item()])
            if sampler_state == SamplerState.PAUSE:
                track_pause = True
                #print("could pause in the future")
                if not should_stop_branch(token_text, gen_tokens_text):
                    sampler_state = SamplerState.ARGMAX

            if track_pause and should_stop_branch(token_text, gen_tokens_text):
                #print("pausing now")
                # we are in a pause state
                if not want_insert:
                    sampler_state = SamplerState.ARGMAX
                    track_pause = False
                    #print("not inserting, continuing")
                    continue
                else:
                    sampler_state = SamplerState.PAUSE
                    last_pause_step = past_key_values.seen_tokens
                    track_pause = False

            # ──────────────────────────────────────────────────────────────────
            # CASE 1: SamplerState.ARGMAX (normal decoding)
            # ──────────────────────────────────────────────────────────────────
            if sampler_state == SamplerState.ARGMAX:
                if past_key_values.seen_tokens == seqlen and do_insert_bos:    
                    # 2. Roll back the KV cache to the state *before* the trigger token.
                    rolled_back_kv = rollback_kv_cache_by_one_token(past_key_values)

                    # 3. Call our clean insertion function.
                    inserted_ids, inserted_text, inserted_metrics, new_past_kv = insert_tokens(
                    model, next_token, past_key_values, logits, metrics,
                    past_key_values.seen_tokens, seqlen, gen_tokens, gen_tokens_text,
                    response, gen_logits, gen_metrics, sampler_states,
                    sampler_cfg, allow_branching, print_stream,
                    include_trigger_token=False,
                    insert_text=insert_text
                    )

                    response += "".join(inserted_text)
                    gen_tokens_text.extend(inserted_text)
                    gen_metrics.extend(inserted_metrics)

                    new_ids_tensor = torch.tensor([inserted_ids], dtype=torch.int32, device=device)
                    gen_tokens = torch.cat((gen_tokens, new_ids_tensor), dim=1)
                    sampler_states.extend([SamplerState.PAUSE] * len(inserted_ids))
                    past_key_values = new_past_kv

                    if inserted_ids:
                        last_inserted_id = inserted_ids[-1]
                        next_token = torch.tensor([[last_inserted_id]], device=device, dtype=torch.int32)

                if torch.isin(next_token, stop_tokens).any() and not track_end and do_insert_eos:
                    track_end = True
                    
                    # phrase = "Final Answer: **A. [124.5; 135.5]**"
                    # phrase_ids = model.tokenizer.encode(phrase, add_special_tokens=False)
                    # 2. Roll back the KV cache to the state *before* the trigger token.
                    # for i in phrase_ids:
                    #     rolled_back_kv = rollback_kv_cache_by_one_token(past_key_values)
                    #     past_key_values = rolled_back_kv

                    past_key_values = rollback_kv_cache_by_one_token(past_key_values)

                    # 3. Call our clean insertion function.
                    inserted_ids, inserted_text, inserted_metrics, new_past_kv = insert_tokens(
                    model, next_token, past_key_values, logits, metrics,
                    past_key_values.seen_tokens, seqlen, gen_tokens, gen_tokens_text,
                    response, gen_logits, gen_metrics, sampler_states,
                    sampler_cfg, allow_branching, print_stream,
                    include_trigger_token=False,
                    insert_text=insert_text
                    )

                    response += "".join(inserted_text)
                    gen_tokens_text.extend(inserted_text)
                    gen_metrics.extend(inserted_metrics)

                    new_ids_tensor = torch.tensor([inserted_ids], dtype=torch.int32, device=device)
                    gen_tokens = torch.cat((gen_tokens, new_ids_tensor), dim=1)
                    sampler_states.extend([SamplerState.PAUSE] * len(inserted_ids))
                    past_key_values = new_past_kv

                    if inserted_ids:
                        last_inserted_id = inserted_ids[-1]
                        next_token = torch.tensor([[last_inserted_id]], device=device, dtype=torch.int32)
                else:
                    # Normal token processing
                    gen_logits.append(logits)
                    gen_metrics.append(metrics)
                    sampler_states.append(sampler_state)

                    gen_tokens = torch.cat((gen_tokens, next_token), dim=1)
                    token_text = model.tokenizer.decode([next_token.item()])
                    gen_tokens_text.append(token_text)
                    response += token_text

                    if print_stream:
                        rprint(f"[{STATE_COLOR_MAP[sampler_state]}]{token_text}[/]", end='')

                    if torch.isin(next_token, stop_tokens).any():
                        yield token_text, metrics, sampler_state, None
                        break

                    yield token_text, metrics, sampler_state, None

        
            # ──────────────────────────────────────────────────────────────────
            # CASE 2: SamplerState.PAUSE (we want to forcibly insert " oh wait")
            # ──────────────────────────────────────────────────────────────────
            elif sampler_state == SamplerState.PAUSE:
                # 1. A PAUSE has been triggered by `next_token`.
                #    DO NOT add this trigger token to our response history yet.
                #    Log its metrics, as it was a valid generation step.
                gen_logits.append(logits)
                gen_metrics.append(metrics)
                sampler_states.append(sampler_state)

                # sampler_cfg.thresholds.logit_entropy.high = sampler_cfg.thresholds.logit_entropy.high + 0.5
                # sampler_cfg.thresholds.logit_varentropy.high = sampler_cfg.thresholds.logit_varentropy.high + 0.5

                if print_stream:
                    # Visually show the user the trigger token was caught, but don't save it.
                    rprint(f"({token_text})", end='')

                # 2. Roll back the KV cache to the state *before* the trigger token.
                rolled_back_kv = rollback_kv_cache_by_one_token(past_key_values)

                # 3. get all the previous tokens and put together to feed into the model in another generation
                #    To ask for the next step

                # print user messages
                #print("The user messages are", messages[-1].content)

                next_step_text = get_next_step(
                    model=model,
                    original_messages=messages,
                    current_response=response,
                )

                # 4. Call our clean insertion function.
                inserted_ids, inserted_text, inserted_metrics, new_past_kv = insert_tokens(
                model, next_token, past_key_values, logits, metrics,
                past_key_values.seen_tokens, seqlen, gen_tokens, gen_tokens_text,
                response, gen_logits, gen_metrics, sampler_states,
                sampler_cfg, allow_branching, print_stream,
                include_trigger_token=False,
                insert_text=next_step_text
                )

                response += "".join(inserted_text)
                gen_tokens_text.extend(inserted_text)
                gen_metrics.extend(inserted_metrics)

                new_ids_tensor = torch.tensor([inserted_ids], dtype=torch.int32, device=device)
                gen_tokens = torch.cat((gen_tokens, new_ids_tensor), dim=1)
                sampler_states.extend([SamplerState.PAUSE] * len(inserted_ids))
                past_key_values = new_past_kv

                if inserted_ids:
                    last_inserted_id = inserted_ids[-1]
                    next_token = torch.tensor([[last_inserted_id]], device=device, dtype=torch.int32)

        # Build final GenerationData if you want
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
    calculate_sim: bool = False,
    do_insert_bos: bool = False,
    do_insert_eos: bool = False,
    want_insert: bool = True,
    enable_thinking: bool = False,
    insert_text: str = " oh wait"
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
        calculate_sim=calculate_sim,
        do_insert_bos=do_insert_bos,
        do_insert_eos=do_insert_eos,
        want_insert=want_insert,
        enable_thinking=enable_thinking,
        insert_text=insert_text
    ):
        if gen is not None:
            return gen
    raise RuntimeError("Generation failed to complete")
