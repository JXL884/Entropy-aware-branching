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

def should_stop_branch(token_text, token_context):
    BRANCH_STOP_TOKENS = {".", ". ", ".\n", "!", "?", ":", "{", "}", "\n\n", ".\n\n", ":\n\n"}

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
):
    if include_trigger_token:
        # Append the triggering token (e.g., stop token if included)
        gen_logits.append(logits)
        gen_metrics.append(metrics)
        sampler_states.append(SamplerState.ARGMAX)
        cur_pos = seqlen if cur_pos < seqlen else cur_pos + 1
        gen_tokens = torch.cat([gen_tokens, next_token], dim=1)
        token_text = model.tokenizer.decode([next_token.item()])
        gen_tokens_text.append(token_text)
        response += token_text
        if print_stream:
            rprint(f"[{STATE_COLOR_MAP[SamplerState.ARGMAX]}]{token_text}[/]", end='')
        yield token_text, metrics, SamplerState.ARGMAX, None

    # 2) Insert whatever
    insert_ids = model.tokenizer.encode(insert_text, add_special_tokens=False)
    for rid in insert_ids:
        forced_token = torch.tensor([[rid]], device=device, dtype=torch.int32)
        gen_tokens = torch.cat([gen_tokens, forced_token], dim=1)

        token_text = model.tokenizer.decode([rid])
        gen_tokens_text.append(token_text)
        response += token_text

        if print_stream:
            rprint(f"[{STATE_COLOR_MAP[SamplerState.PAUSE]}]{token_text}[/]", end='')

        with torch.inference_mode():
            forced_outputs = model.weights(
                input_ids=forced_token,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True
            )

        past_key_values = forced_outputs.past_key_values
        forced_logits = forced_outputs.logits
        forced_scores = forced_outputs.attentions[-1]
        forced_metrics = calculate_metrics(forced_logits, forced_scores)
        gen_logits.append(forced_logits)
        gen_metrics.append(forced_metrics)
        sampler_states.append(SamplerState.PAUSE)

        cur_pos += 1

        yield token_text, forced_metrics, SamplerState.PAUSE, None

    # # 3) Sample a new next_token
    # next_token, sampler_state = sample(
    #     forced_logits,
    #     forced_scores,
    #     forced_metrics,
    #     sampler_cfg,
    #     can_branch=allow_branching and cur_pos >= seqlen,
    #     current_step=cur_pos
    # )
    # token_text = model.tokenizer.decode([next_token.item()])
    # # 4) Yield the last inserted token
    # yield token_text, forced_metrics, SamplerState.PAUSE, None

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
    do_insert: bool = False,
    insert_text: str | None = None
) -> Generator[Tuple[Optional[str], Optional[TokenMetrics], Optional[SamplerState], Optional[GenerationData]], None, None]:

    # (A) Initialize the "oh wait" cooldown
    cooldown_length = 20          # minimum number of tokens between "oh wait" insertions
    last_oh_wait_step = -9999     # track when we last inserted "oh wait"

    # # If the tokenizer has 'stop_token_ids', use them
    # if hasattr(model.tokenizer, "stop_token_ids"):
    #     stop_ids = model.tokenizer.stop_token_ids
    # elif (hasattr(model.tokenizer, "eos_token_id") 
    #       and model.tokenizer.eos_token_id is not None):
    #     stop_ids = [model.tokenizer.eos_token_id]
    # else:
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
        prompt = model.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        print(prompt)

    if print_stream:
        print()
        for state, color in STATE_COLOR_MAP.items():
            rprint(f"[{color}]■[/] [dim]{state.value}[/]")
        print()

    #print("The prompt is", prompt)

    with torch.inference_mode():
        tokens = torch.tensor([prompt], dtype=torch.long).to(device)
        bs, seqlen = tokens.shape
        cur_pos = seqlen

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
        track_pause = False
        track_end = False

        while cur_pos < max_tokens:
            #print("cur_pos", cur_pos)
            outputs = model.weights(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True,
                output_hidden_states=False,
            )

            logits = outputs.logits
            past_key_values = outputs.past_key_values
            scores = outputs.attentions[-1]

            metrics = calculate_metrics(logits, scores)
            num_tokens_so_far = gen_tokens.shape[1]
            next_token, sampler_state = sample(
                logits,
                scores,  
                metrics,
                sampler_cfg,
                can_branch=allow_branching and cur_pos >= seqlen,
                current_step=num_tokens_so_far  # new parameter to track the current step
            )
            token_text = model.tokenizer.decode([next_token.item()])
            if sampler_state == SamplerState.PAUSE:
                track_pause = True
                #print("could pause in the future")
                if not should_stop_branch(token_text, gen_tokens_text):
                    sampler_state = SamplerState.ARGMAX

            if track_pause and should_stop_branch(token_text, gen_tokens_text):
                #print("pausing now")
                # uncomment this to not insert anything
                # sampler_state = SamplerState.ARGMAX
                sampler_state = SamplerState.PAUSE
                track_pause = False

            # ──────────────────────────────────────────────────────────────────
            # CASE 1: SamplerState.ARGMAX (normal decoding)
            # ──────────────────────────────────────────────────────────────────
            if sampler_state == SamplerState.ARGMAX:
                if cur_pos == seqlen and do_insert:    
                    insert_count = 0
                    for token_text, metrics, state, _ in insert_tokens(
                        model,
                        next_token, past_key_values, logits, metrics,
                        cur_pos, seqlen, gen_tokens, gen_tokens_text,
                        response, gen_logits, gen_metrics, sampler_states,
                        sampler_cfg, allow_branching, print_stream,
                        include_trigger_token=False,
                        insert_text=insert_text
                    ):
                        yield token_text, metrics, state, None
                        insert_count += 1
                        last_yielded = token_text
                    # After insertion, decode the last token to set next_token
                    if last_yielded:
                        next_token = torch.tensor([[model.tokenizer.encode(last_yielded)[-1]]], device=device, dtype=torch.int32)
                    #cur_pos += insert_count
                    #print("inserted", insert_count, "tokens")

                if torch.isin(next_token, stop_tokens).any() and not track_end:
                    track_end = True
                    if print_stream:
                        rprint(f"[{STATE_COLOR_MAP[sampler_state]}]{token_text}[/]", end='')
                    insert_count = 0
                    for token_text, metrics, state, _ in insert_tokens(
                        model,
                        next_token, past_key_values, logits, metrics,
                        cur_pos, seqlen, gen_tokens, gen_tokens_text,
                        response, gen_logits, gen_metrics, sampler_states,
                        sampler_cfg, allow_branching, print_stream,
                        include_trigger_token=False,
                        insert_text=insert_text
                    ):
                        yield token_text, metrics, state, None
                        insert_count += 1
                        last_yielded = token_text
                    if last_yielded:
                        next_token = torch.tensor([[model.tokenizer.encode(last_yielded)[-1]]], device=device, dtype=torch.int32)
                    #cur_pos += insert_count
                else:
                    gen_logits.append(logits)
                    gen_metrics.append(metrics)
                    sampler_states.append(sampler_state)

                    # Move cur_pos forward (first step => from 0 to seqlen, else increment)
                    cur_pos = seqlen if cur_pos < seqlen else cur_pos + 1

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
            # CASE 2: SamplerState.BRANCHING
            # ──────────────────────────────────────────────────────────────────
            elif sampler_state == SamplerState.BRANCHING:
                branches = _generate_branches(
                    model, next_token, past_key_values, cur_pos,
                    logits, metrics, stop_tokens, max_tokens,
                    sampler_cfg, print_stream
                )
                gen_branches.append([branch.to_dict() for branch in branches])
                branch_count += 1

                if random_select:
                    chosen_index = random.randint(0, 4)
                else:
                    if feedback_provider == "llama3.3":
                        chosen_index = eval_branches(branches, messages, response, model, sampler_cfg)
                    elif feedback_provider == "PRM":
                        chosen_index = score_branch(branches, messages, response, score_model)
                    else:
                        raise ValueError("Invalid feedback_provider name. Must be 'llama3.3' or 'PRM'.")

                best_branch = branches[chosen_index]
                branch_choices.append(chosen_index)

                branch_texts = ["".join(b.tokens_text) for b in branches]
                if len(branches) > 1:
                    embeddings = get_openai_embeddings(branch_texts, model_name="text-embedding-3-large")
                    sim_matrix = pairwise_cosine_similarity(embeddings)
                else:
                    sim_matrix = np.array([[1.0]])
                all_pairwise_similarities.append(sim_matrix.tolist())

                # discard unchosen branches
                for branch in branches:
                    if branch != best_branch:
                        del branch

                next_token = best_branch.tokens[-1]
                kvcache = best_branch.kvcache.to(device)
                cur_pos = best_branch.cur_pos

                gen_tokens = torch.cat(
                    [gen_tokens, torch.tensor(best_branch.tokens, device=device).unsqueeze(0)],
                    dim=1
                )
                gen_tokens_text.extend(best_branch.tokens_text)
                gen_metrics.extend(best_branch.metrics)
                sampler_states.extend(best_branch.sampler_states)
                branch_response = "".join(best_branch.tokens_text)
                response += branch_response

                if print_stream:
                    rprint(f"\n[{STATE_COLOR_MAP[SamplerState.BRANCHING]}]=>[/]", end='')
                    for state, text in zip(best_branch.sampler_states, best_branch.tokens_text):
                        rprint(f"[{STATE_COLOR_MAP[state]}]{text.replace('\n', '\\n')}[/]", end='')

                if torch.isin(next_token, stop_tokens).any():
                    break

                yield token_text, metrics, sampler_state, None

            # ──────────────────────────────────────────────────────────────────
            # CASE 3: SamplerState.PAUSE (we want to forcibly insert " oh wait")
            # ──────────────────────────────────────────────────────────────────
            elif sampler_state == SamplerState.PAUSE:
                insert_count = 0
                for token_text, metrics, state, _ in insert_tokens(
                    model,
                    next_token, past_key_values, logits, metrics,
                    cur_pos, seqlen, gen_tokens, gen_tokens_text,
                    response, gen_logits, gen_metrics, sampler_states,
                    sampler_cfg, allow_branching, print_stream,
                    include_trigger_token=True,
                    insert_text=insert_text
                ):
                    yield token_text, metrics, state, None
                    insert_count += 1
                    last_yielded = token_text
                # After insertion, decode the last token to set next_token
                if last_yielded:
                    next_token = torch.tensor([[model.tokenizer.encode(last_yielded)[-1]]], device=device, dtype=torch.int32)
                #cur_pos += insert_count

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
    do_insert: bool = False,
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
        do_insert=do_insert,
        insert_text=insert_text
    ):
        if gen is not None:
            return gen
    raise RuntimeError("Generation failed to complete")
