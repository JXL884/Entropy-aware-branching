from __future__ import annotations

import json
import logging
import os
from enum import Enum
from dataclasses import asdict, dataclass, field
from typing import Any, Generator, List, Literal, NamedTuple, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from openai import OpenAI
from rich import print as rprint
from transformers import DynamicCache

from entropix.config import (
    STATE_COLOR_MAP,
    SamplerConfig,
    SamplerState,
    DynamicThresholdManager,
)
from entropix.kvcache import KVCache
from entropix.metrics import TokenMetrics, calculate_metrics
from entropix.sampler import sample
from entropix.tokenizer import Message, Tokenizer

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {device}")


class GenerationMode(Enum):
    """High-level generation mode used by FlowController."""

    NORMAL = "normal"                           # Regular adaptive sampling
    TRIGGERED = "triggered"                     # Uncertainty detected, waiting for stop token
    INSERTING = "inserting"                     # Currently injecting reflection text
    COOLDOWN = "cooldown"                       # Cooldown period to avoid immediate re-trigger
    THINKING_COMPLETE = "thinking_complete"     # Thinking phase completed, no more insertions


@dataclass
class FlowController:
    """Encapsulates pause / insertion / cooldown logic."""

    cfg: SamplerConfig
    mode: GenerationMode = GenerationMode.NORMAL
    last_pause_step: int = -9999                # step index of last completed pause
    thinking_complete: bool = False             # flag to track if thinking phase is complete

    def request_pause(self, step: int):
        """Sampler signalled uncertainty – attempt to enter TRIGGERED."""
        if not (step - self.last_pause_step) < self.cfg.cooldown_length:
            self.mode = GenerationMode.TRIGGERED

    def on_token_sampled(self, token_text: str, context: list[str], step: int, 
                        stop_thinking_token_ids: list[int], next_token_id: int):
        """Called **after** we sample a token but **before** insertion. Decides whether the stop token criteria are fulfilled."""
        # Check for thinking completion first (only if not already complete)
        if not self.thinking_complete and next_token_id in stop_thinking_token_ids:
            self.thinking_complete = True
            self.mode = GenerationMode.THINKING_COMPLETE
            return
            
        # Existing logic for pause detection (only if thinking is not complete)
        if (self.mode is GenerationMode.TRIGGERED and 
            not self.thinking_complete and 
            should_stop_branch(token_text, context)):
            self.mode = GenerationMode.INSERTING

    def insertion_complete(self, step: int):
        """Call after reflection insertion is done."""
        self.mode = GenerationMode.COOLDOWN
        self.last_pause_step = step


################################################################################
#                              Helper Functions                                 #
################################################################################


def get_model_tokens(tokenizer) -> dict:
    """Dynamically get model-specific tokens from tokenizer."""
    stop_token_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id else []

    thinking_token_ids = []
    stop_thinking_token_ids = []

    if "<think>" in tokenizer.get_vocab():
        thinking_token_ids = [tokenizer.encode("<think>", add_special_tokens=False)[0]]

    if "</think>" in tokenizer.get_vocab():
        stop_thinking_token_ids = [
            tokenizer.encode("</think>", add_special_tokens=False)[0]
        ]
        if tokenizer.eos_token_id:
            stop_thinking_token_ids.append(tokenizer.eos_token_id)

    return {
        "stop_token_ids": stop_token_ids,
        "thinking_token_ids": thinking_token_ids,
        "stop_thinking_token_ids": stop_thinking_token_ids,
    }


def should_insert_at_start(current_state: dict, insert_at_start: bool) -> bool:
    """Check if we should insert text at the start of generation."""
    return (
        insert_at_start
        and current_state["past_key_values"].seen_tokens == current_state["seqlen"]
    )


def should_insert_at_end(
    current_state: dict, insert_at_end: bool, stop_token_ids: list
) -> bool:
    """Check if we should insert text at the end of generation."""
    return (
        insert_at_end
        and torch.isin(
            current_state["next_token"], torch.tensor(stop_token_ids, device=device)
        ).any()
        and not current_state.get("track_end", False)
    )


def insert_text_at_position(
    model: Model,
    insertion_type: Literal["start", "end", "pause"],
    current_state: dict,
    sampler_cfg: SamplerConfig,
    insertion_text: str | None = None,
) -> dict:
    """
    Unified text insertion function that handles BOS, EOS, and PAUSE insertions.

    Args:
        insertion_type: "start" (BOS), "end" (EOS), or "pause" (reflection)
        current_state: Current generation state
        sampler_cfg: Sampler configuration
        insertion_text: Text to insert (None for auto-generation in pause mode)

    Returns:
        Updated state with inserted tokens
    """
    # 1. Determine insertion text
    if insertion_text is None:
        if insertion_type == "pause":
            # Auto-generate reflection text
            insertion_text = get_next_step(
                model=model,
                original_messages=current_state["messages"],
                current_response=current_state["response"],
                max_new_tokens=500,  # Default limit
            )
        else:
            insertion_text = "Wait"  # Default fallback text

    # 2. Roll back KV cache to before the trigger token (not needed for the middle approach)
    # rolled_back_kv = rollback_kv_cache_by_one_token(current_state["past_key_values"])

    # 3. Encode and insert the text
    insert_ids = model.tokenizer.encode(insertion_text, add_special_tokens=False)

    new_tokens_ids = []
    new_tokens_text = []
    new_metrics = []
    current_past_kv = current_state["past_key_values"]

    # 4. Process each token in the insertion text
    for rid in insert_ids:
        new_tokens_ids.append(rid)
        forced_token = torch.tensor([[rid]], device=device, dtype=torch.int32)

        with torch.inference_mode():
            forced_outputs = model.weights(
                input_ids=forced_token,
                past_key_values=current_past_kv,
                use_cache=True,
                output_attentions=True,
            )

        # Update state for next iteration
        current_past_kv = forced_outputs.past_key_values

        # Log results
        token_text = model.tokenizer.decode([rid])
        new_tokens_text.append(token_text)

        forced_logits = forced_outputs.logits
        forced_scores = forced_outputs.attentions[-1]
        forced_metrics = calculate_metrics(forced_logits, forced_scores)
        new_metrics.append(forced_metrics)

        if current_state["print_stream"]:
            rprint(f"[{STATE_COLOR_MAP[SamplerState.PAUSE]}]{token_text}[/]", end="")

    # 5. Update current state
    current_state["response"] += "".join(new_tokens_text)
    current_state["gen_tokens_text"].extend(new_tokens_text)
    current_state["gen_metrics"].extend(new_metrics)
    current_state["past_key_values"] = current_past_kv

    # 6. Update token tensors
    if new_tokens_ids:
        new_ids_tensor = torch.tensor(
            [new_tokens_ids], dtype=torch.int32, device=device
        )
        # Append the inserted tokens and their states
        current_state["gen_tokens"] = torch.cat(
            (current_state["gen_tokens"], new_ids_tensor), dim=1
        )
        current_state["sampler_states"].extend(
            [SamplerState.PAUSE] * len(new_tokens_ids)
        )

        # The next token for the main loop is the *last token of the insertion*.
        # The loop will use this token as input to predict what comes next.
        last_inserted_id = new_tokens_ids[-1]
        current_state["next_token"] = torch.tensor(
            [[last_inserted_id]], device=device, dtype=torch.int32
        )
    return current_state


def process_normal_token(
    current_state: dict, sampler_state: SamplerState, stream_output: bool
) -> dict:
    """Process a normal token (non-insertion)."""
    # Log the token
    current_state["gen_logits"].append(current_state["logits"])
    current_state["gen_metrics"].append(current_state["metrics"])
    current_state["sampler_states"].append(sampler_state)

    # Add to generation
    current_state["gen_tokens"] = torch.cat(
        (current_state["gen_tokens"], current_state["next_token"]), dim=1
    )
    current_state["gen_tokens_text"].append(current_state["token_text"])
    current_state["response"] += current_state["token_text"]

    if stream_output:
        rprint(
            f"[{STATE_COLOR_MAP[sampler_state]}]{current_state['token_text']}[/]",
            end="",
        )

    return current_state


def initialize_generation_state(
    messages: list[Message] | list[dict[str, str]] | str,
    model: Model,
    format_messages: bool,
    enable_thinking: bool,
    max_tokens: int | None,
) -> dict:
    """Initialize the generation state."""
    # Convert messages to standard format
    if isinstance(messages, str):
        prompt = messages
        messages = [Message(role="system", content=prompt)]
        logging.warning(
            "entropix.model._generate: prompt passed as a string, cannot save messages to output GenerationData."
        )
    elif isinstance(messages, list) and isinstance(messages[0], dict):
        messages = [Message(**m) if not isinstance(m, Message) else m for m in messages]

    assert isinstance(messages, list) and all(isinstance(m, Message) for m in messages)

    # Apply chat template if requested
    if format_messages:
        print("The prompt is", messages)
        prompt = model.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=enable_thinking,
        )
        prompt_for_print = model.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=enable_thinking,
        )
        print(prompt_for_print)
    else:
        prompt = messages[-1].content if messages else ""

    # Set max tokens
    if max_tokens is None or max_tokens > model.params.max_position_embeddings:
        max_tokens = model.params.max_position_embeddings

    # Initialize tensors
    tokens = torch.tensor([prompt], dtype=torch.int32).to(device)
    bs, seqlen = tokens.shape

    return {
        "messages": messages,
        "prompt": prompt,
        "seqlen": seqlen,
        "max_tokens": max_tokens,
        "next_token": tokens,
        "gen_tokens": torch.zeros(1, 1, dtype=torch.int32, device=device),
        "cur_seen_tokens": 0,
        "past_key_values": None,
        "response": "",
        "gen_tokens_text": [],
        "gen_logits": [],
        "gen_metrics": [],
        "gen_branches": [],
        "sampler_states": [],
        "threshold_history": [],  
        "branch_count": 0,
        "branch_choices": [],
        "all_pairwise_similarities": [],
        "track_end": False,
        "print_stream": False,  # Will be set by caller
    }


def build_generation_data(
    current_state: dict, messages: list[Message]
) -> GenerationData:
    """Build the final GenerationData object."""
    messages.append(Message(role="assistant", content=current_state["response"]))
    return GenerationData(
        prompt=current_state["prompt"],
        response=current_state["response"],
        tokens=current_state["gen_tokens_text"],
        messages=messages,
        branches=current_state["gen_branches"],
        metrics=current_state["gen_metrics"],
        sampler_cfg=current_state.get("sampler_cfg"),
        sampler_states=current_state["sampler_states"],
        branch_count=current_state["branch_count"],
        branch_choices=current_state["branch_choices"],
        branch_pairwise_similarities=current_state["all_pairwise_similarities"],
        threshold_history=current_state["threshold_history"],
    )


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
    threshold_history: List[dict] = field(
        default_factory=list
    )  # Track actual thresholds used

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
            "threshold_history": self.threshold_history,
        }

    def save(self, fp: str):
        dir_path = os.path.dirname(fp)  # Extract the directory path
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)

        with open(fp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4, ensure_ascii=False, sort_keys=True)

    @classmethod
    def load(cls, fp: str):
        with open(fp, "rb") as f:
            data = json.load(f)
        defaults = {
            "branches": [],
            "metrics": [],
            "messages": [],
            "tokens": [],
            "sampler_states": [],
            "prompt": "",
            "response": "",
            "sampler_cfg": SamplerConfig().model_dump(),
            "threshold_history": [],
        }
        for k, default in defaults.items():
            if k not in data:
                logging.warning(
                    f"Missing field '{k}' in loaded data, using default: {default}"
                )
                data[k] = default
        data["metrics"] = [TokenMetrics(**m) for m in data["metrics"]]
        data["messages"] = [Message(**m) for m in data["messages"]]
        data["sampler_cfg"] = SamplerConfig.from_dict(data["sampler_cfg"])
        data["sampler_states"] = [SamplerState[name] for name in data["sampler_states"]]
        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        defaults = {
            "branches": [],
            "metrics": [],
            "messages": [],
            "tokens": [],
            "sampler_states": [],
            "prompt": "",
            "response": "",
            "branch_count": 0,
            "branch_choices": [],
            "branch_pairwise_similarities": [],
            "sampler_cfg": SamplerConfig().model_dump(),
            "threshold_history": [],
        }
        for k, default in defaults.items():
            if k not in data:
                logging.warning(
                    f"Missing field '{k}' in loaded data, using default: {default}"
                )
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
    BRANCH_STOP_TOKENS = [
        "\n\n",
        ",\n\n",
        ".\n\n",
        "]\n\n",
        ")\n\n",
        "],\n\n",
        "].\n\n",
        "].\n\n",
        ").\n\n",
        ".)\n\n",
        "?\n\n",
        "!\n\n",
    ]

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
        messages=messages,  # type: ignore
    )
    eval = completion.choices[0].message.content
    if eval is None:
        eval = ""
    return eval


def get_openai_embeddings(
    texts: list[str], model_name: str = "text-embedding-3-large"
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

        response = client.embeddings.create(input=[text], model=model_name)

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


def get_next_step(
    model: Model,
    original_messages: list[Message],
    current_response: str,
    max_new_tokens: int = 500,  # Limit the length of the "next step"
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
    thinking: list[int] = [151667]  # Qwen's <think>
    stop_thinking: list[int] = [151645, 151668]  # Qwen's </think> and stop token
    thinking_tokens = torch.tensor(thinking, device=device, dtype=torch.int32)
    stop_tokens = torch.tensor(stop_thinking, device=device, dtype=torch.int32)

    # 1. Construct the "meta-prompt" for reflection.
    meta_prompt_messages = [
        Message(
            role="system",
            content="You are a collaborative AI expert. You are given a conversation history where the last assistant message is an incomplete, step-by-step solution. "
            "Your task is to reflect on the previous solution and continue the solution by generating the next logical step. Do not repeat the previous step ",
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
        Message(
            role="assistant",
            content="I need to briefly complete the next step ONLY. DO NOT SOLVE THE PROBLEM. Continue from the pre-existing reasoning process. /think",
        ),
    ]

    # 2. Tokenize the meta-prompt.
    meta_prompt_ids = model.tokenizer.apply_chat_template(
        meta_prompt_messages,
        add_generation_prompt=True,
        tokenize=True,
        enable_thinking=True,
    )
    next_token = torch.tensor([meta_prompt_ids], device=device, dtype=torch.int32)

    # 3. Perform a new, separate generation to get the next step.
    generated_ids = []
    past_key_values = None
    with torch.inference_mode():
        for i in range(max_new_tokens):
            outputs = model.weights(
                input_ids=next_token, past_key_values=past_key_values, use_cache=True
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
    messages: list[Message] | list[dict[str, str]] | str,
    model: Model,
    sampler_cfg: SamplerConfig | None = None,
    max_tokens: int | None = None,
    stream_output: bool = False,
    format_messages: bool = True,
    enable_thinking: bool = False,
    enable_uncertainty_detection: bool = True,
    enable_insertion: bool = True,
    insertion_text: str = "\n\nWait",
    insert_at_start: bool = False,
    insert_at_end: bool = False,
) -> Generator[
    Tuple[
        Optional[str],
        Optional[TokenMetrics],
        Optional[SamplerState],
        Optional[GenerationData],
    ],
    None,
    None,
]:
    # 1. Setup and validation
    if sampler_cfg is None:
        logging.warning("No sampler config provided, using default config")
        sampler_cfg = SamplerConfig()

    # Initialize dynamic threshold manager if dynamic thresholding is enabled
    threshold_manager = None
    if sampler_cfg.thresholds.dynamic.strategy != "static":
        threshold_manager = DynamicThresholdManager(
            sampler_cfg.thresholds, sampler_cfg.thresholds.dynamic
        )

    # 2. Initialize state
    current_state = initialize_generation_state(
        messages, model, format_messages, enable_thinking, max_tokens
    )
    current_state["print_stream"] = stream_output
    current_state["sampler_cfg"] = sampler_cfg

    flow = FlowController(sampler_cfg)

    # 3. Get model-specific tokens
    model_tokens = get_model_tokens(model.tokenizer)

    # 4. Show state legend if streaming
    if stream_output:
        print()
        for state, color in STATE_COLOR_MAP.items():
            rprint(f"[{color}]■[/] [dim]{state.value}[/]")
        print()

    # 5. Main generation loop
    with torch.inference_mode():
        while current_state["cur_seen_tokens"] < current_state["max_tokens"]:
            # Get model outputs
            outputs = model.weights(
                input_ids=current_state["next_token"],
                past_key_values=current_state["past_key_values"],
                use_cache=True,
                output_attentions=True,
                output_hidden_states=False,
            )

            # Update state with outputs
            current_state.update(
                {
                    "logits": outputs.logits,
                    "past_key_values": outputs.past_key_values,
                    "cur_seen_tokens": outputs.past_key_values.seen_tokens,
                    "scores": outputs.attentions[-1],
                }
            )

            # Calculate metrics and sample
            metrics = calculate_metrics(
                current_state["logits"], current_state["scores"]
            )

            # Get current thresholds (static or dynamic) for tracking
            current_thresholds = None
            if threshold_manager is not None:
                current_thresholds = threshold_manager.get_thresholds(
                    {
                        "logit_entropy": metrics.logit_entropy,
                        "logit_varentropy": metrics.logit_varentropy,
                        "attn_entropy": metrics.attn_entropy,
                        "attn_varentropy": metrics.attn_varentropy,
                        "agreement": metrics.agreement,
                        "interaction_strength": metrics.interaction_strength,
                    }
                )
            else:
                current_thresholds = sampler_cfg.thresholds

            # Store threshold values for this step
            threshold_step = {
                "logit_entropy": {
                    "low": current_thresholds.logit_entropy.low,
                    "medium": current_thresholds.logit_entropy.medium,
                    "high": current_thresholds.logit_entropy.high,
                },
                "logit_varentropy": {
                    "low": current_thresholds.logit_varentropy.low,
                    "medium": current_thresholds.logit_varentropy.medium,
                    "high": current_thresholds.logit_varentropy.high,
                },
                "attn_entropy": {
                    "low": current_thresholds.attn_entropy.low,
                    "medium": current_thresholds.attn_entropy.medium,
                    "high": current_thresholds.attn_entropy.high,
                },
                "attn_varentropy": {
                    "low": current_thresholds.attn_varentropy.low,
                    "medium": current_thresholds.attn_varentropy.medium,
                    "high": current_thresholds.attn_varentropy.high,
                },
                "agreement": {
                    "low": current_thresholds.agreement.low,
                    "medium": current_thresholds.agreement.medium,
                    "high": current_thresholds.agreement.high,
                },
                "interaction_strength": {
                    "low": current_thresholds.interaction_strength.low,
                    "medium": current_thresholds.interaction_strength.medium,
                    "high": current_thresholds.interaction_strength.high,
                },
            }
            current_state["threshold_history"].append(threshold_step)

            next_token, sampler_state = sample(
                current_state["logits"],
                current_state["scores"],
                metrics,
                sampler_cfg,
                threshold_manager=threshold_manager,
                can_branch=enable_uncertainty_detection
                and current_state["cur_seen_tokens"] >= current_state["seqlen"],
                current_step=current_state["cur_seen_tokens"],
                last_pause_step=flow.last_pause_step,
            )

            current_state.update(
                {
                    "next_token": next_token,
                    "metrics": metrics,
                    "token_text": model.tokenizer.decode([next_token.item()]),
                }
            )

            # 1) Signal potential pause from sampler
            if sampler_state is SamplerState.PAUSE:
                flow.request_pause(current_state["cur_seen_tokens"])

            # 2) Update flow state with the freshly sampled token
            flow.on_token_sampled(
                current_state["token_text"],
                current_state["gen_tokens_text"],
                current_state["cur_seen_tokens"],
                model_tokens["stop_thinking_token_ids"],
                next_token.item(),
            )

            # 3) Determine how this token should be coloured in stream output
            visible_state = (
                SamplerState.ADAPTIVE
                if flow.mode in (GenerationMode.TRIGGERED, GenerationMode.INSERTING, GenerationMode.THINKING_COMPLETE)
                else sampler_state
            )

            # 4) Log / stream the token normally
            current_state = process_normal_token(
                current_state, visible_state, stream_output
            )

            # 5) Optional BOS / EOS insertions (only if thinking is not complete)
            if (not flow.thinking_complete and 
                should_insert_at_start(current_state, insert_at_start)):
                current_state = insert_text_at_position(
                    model, "start", current_state, sampler_cfg, insertion_text
                )
            elif (not flow.thinking_complete and 
                  should_insert_at_end(current_state, insert_at_end, model_tokens["stop_token_ids"])):
                current_state["track_end"] = True
                current_state = insert_text_at_position(
                    model, "end", current_state, sampler_cfg, insertion_text
                )

            # 6) If flow decided it's time to insert reflection text (only if thinking is not complete)
            if (flow.mode is GenerationMode.INSERTING and 
                enable_insertion and 
                not flow.thinking_complete):
                current_state = insert_text_at_position(
                    model, "pause", current_state, sampler_cfg, insertion_text
                )
                flow.insertion_complete(current_state["cur_seen_tokens"])

            # Check for stop conditions
            if torch.isin(
                current_state["next_token"],
                torch.tensor(model_tokens["stop_token_ids"], device=device),
            ).any():
                yield (
                    current_state["token_text"],
                    current_state["metrics"],
                    sampler_state,
                    None,
                )
                break

            yield (
                current_state["token_text"],
                current_state["metrics"],
                sampler_state,
                None,
            )

        # Return final generation data
        yield (
            "",
            current_state["metrics"],
            sampler_state,
            build_generation_data(current_state, current_state["messages"]),
        )


def stream(
    messages: list[Message] | list[dict[str, str]] | str,
    model: Model,
    sampler_cfg: SamplerConfig | None = None,
    max_tokens: int | None = None,
    stream_output: bool = False,
    format_messages: bool = True,
    enable_thinking: bool = False,
    enable_uncertainty_detection: bool = True,
    enable_insertion: bool = True,
    insertion_text: str = " oh wait",
    insert_at_start: bool = False,
    insert_at_end: bool = False,
):
    for token_text, metrics, sampler_state, gen in _generate(
        messages=messages,
        model=model,
        sampler_cfg=sampler_cfg,
        max_tokens=max_tokens,
        stream_output=stream_output,
        format_messages=format_messages,
        enable_thinking=enable_thinking,
        enable_uncertainty_detection=enable_uncertainty_detection,
        enable_insertion=enable_insertion,
        insertion_text=insertion_text,
        insert_at_start=insert_at_start,
        insert_at_end=insert_at_end,
    ):
        yield token_text, metrics, sampler_state, gen


def generate(
    messages: list[Message] | list[dict[str, str]] | str,
    model: Model,
    sampler_cfg: SamplerConfig | None = None,
    max_tokens: int | None = None,
    stream_output: bool = False,
    format_messages: bool = True,
    enable_thinking: bool = False,
    enable_uncertainty_detection: bool = True,
    enable_insertion: bool = True,
    insertion_text: str = " oh wait",
    insert_at_start: bool = False,
    insert_at_end: bool = False,
) -> GenerationData:
    for token_text, metrics, sampler_state, gen in _generate(
        messages=messages,
        model=model,
        sampler_cfg=sampler_cfg,
        max_tokens=max_tokens,
        stream_output=stream_output,
        format_messages=format_messages,
        enable_thinking=enable_thinking,
        enable_uncertainty_detection=enable_uncertainty_detection,
        enable_insertion=enable_insertion,
        insertion_text=insertion_text,
        insert_at_start=insert_at_start,
        insert_at_end=insert_at_end,
    ):
        if gen is not None:
            return gen
    raise RuntimeError("Generation failed to complete")
