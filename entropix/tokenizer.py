import json
import os
from pathlib import Path
from typing import (
    AbstractSet,
    Any,
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
)
from pydantic import BaseModel, model_validator
from typing_extensions import Self
from transformers import PreTrainedTokenizerFast

# The following constants remain unchanged
TIKTOKEN_MAX_ENCODE_CHARS = 400_000
MAX_NO_WHITESPACES_CHARS = 25_000


class Message(BaseModel):
    class ToolCallFunction(BaseModel):
        name: str
        arguments: str

    class ToolCall(BaseModel):
        id: str
        type: str  # only "function" is currently supported in openai api
        function: "Message.ToolCallFunction"

    content: str | list[str]
    role: Literal["system", "user", "assistant", "tool"]
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None

    @model_validator(mode='after')
    def validate_role_restricted_params(self) -> Self:
        if self.role != "assistant" and self.tool_calls is not None:
            raise ValueError("Only assistant messages can have tool_calls")
        elif self.role == "tool" and self.tool_call_id is None:
            raise ValueError("Tool messages must have a tool_call_id")
        return self

class Tokenizer:
    """
    Tokenizing and encoding/decoding text using a tokenizer.json file.
    """

    special_tokens: Dict[str, int]
    num_reserved_special_tokens = 17

    def __init__(self, tokenizer_path: str | Path, tokenizer_cfg_path: str | None = None):
        """
        Initializes the Tokenizer with a tokenizer.json file.

        Args:
            tokenizer_path (str): The path to the tokenizer.json file.
        """
        if isinstance(tokenizer_path, Path): tokenizer_path = str(tokenizer_path)
        assert os.path.isfile(tokenizer_path), tokenizer_path

        self.model = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

        if tokenizer_cfg_path is None: tokenizer_cfg_path = f"{tokenizer_path[:-5]}_config.json"

        # Load special tokens from tokenizer config file and set up tokenizer
        with open(tokenizer_cfg_path) as f:
            cfg = json.load(f)

        self.special_tokens = {token['content']: int(token_id) for token_id, token in cfg['added_tokens_decoder'].items()}

        self.n_words = self.model.vocab_size
        self.bos_token = cfg["bos_token"]
        self.bos_id = self.special_tokens[self.bos_token]
        self.eos_token = cfg["eos_token"]
        self.eos_id = self.special_tokens[self.eos_token]

        # TODO: probably need a better way to infer/set eot and eom since neither are in llama/smollm tokenizer configs and they are different (indicitive of other tokenizers?)
        self.eot_token = cfg["eot_token"] if "eot_token" in cfg else "<|eot_id|>" if "<|eot_id|>" in self.special_tokens else self.bos_token
        self.eot_id = self.special_tokens[self.eot_token]
        self.eom_token = cfg["eom_token"] if "eom_token" in cfg else "<|eom_id|>" if "<|eom_id|>" in self.special_tokens else self.eos_token
        self.eom_id = self.special_tokens[self.eom_token]

        self.stop_tokens = [self.eot_token, self.eom_token]
        self.stop_token_ids = [self.eot_id, self.eom_id]

        # TODO: same with these other special tokens not (always) defined in the configs
        self.pad_token = cfg["pad_token"
                            ] if "pad_token" in cfg else "<|finetune_right_pad_id|>" if "<|finetune_right_pad_id|>" in self.special_tokens else self.eos_token
        self.pad_id = self.special_tokens[self.pad_token]
        self.python_tag_token = cfg[
            "python_tag"
        ] if "python_tag" in cfg else "<|python_tag|>" if "<|python_tag|>" in self.special_tokens else "<jupyter_code>" if "<jupyter_code>" in self.special_tokens else ""
        self.python_tag_id = self.special_tokens[self.python_tag_token] if self.python_tag_token else None

        self.chat_template = cfg["chat_template"] if "chat_template" in cfg else None

        # FIX: and these seem to never be defined?
        self.start_header_token = cfg["start_header_token"
                                     ] if "start_header_token" in cfg else "<|start_header_id|>" if "<|start_header_id|>" in self.special_tokens else ""
        self.start_header_id = self.special_tokens[self.start_header_token] if self.start_header_token else None
        self.end_header_token = cfg["end_header_token"
                                   ] if "end_header_token" in cfg else "<|end_header_id|>" if "<|end_header_id|>" in self.special_tokens else ""
        self.end_header_id = self.special_tokens[self.end_header_token] if self.end_header_token else None

    def encode(
        self,
        s: str,
        *,
        bos: bool,
        eos: bool,
        allowed_special: Optional[Union[Literal['all'], AbstractSet[str]]] = None,
        disallowed_special: Union[Literal['all'], Collection[str]] = (),
    ) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.
            allowed_special ("all"|set[str]): allowed special tokens in string
            disallowed_special ("all"|set[str]): special tokens that raise an error when in string

        Returns:
            list[int]: A list of token IDs.
        """
        if allowed_special is None:
            allowed_special = set()
        assert isinstance(s, str)

        substrs = (
            substr for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
            for substr in self._split_whitespaces_or_nonwhitespaces(s[i:i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS)
        )
        t: List[int] = []
        for substr in substrs:
            t.extend(self.model.encode(substr, add_special_tokens=False))
        if bos:
            t.insert(0, self.bos_id)
        if eos:
            t.append(self.eos_id)
        return t

    def decode(self, t: Sequence[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (List[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        return self.model.decode(t)

    def apply_chat_template(self, messages: list[Message] | str, tools: list[dict[str, Any]] | None = None) -> str:
        if isinstance(messages, str): messages = [Message(role="user", content=messages)]
        if self.chat_template:
            from jinja2 import Template
            template = Template(self.chat_template)
            return template.render(messages=messages, add_generation_prompt=True, custom_tools=tools)
        else:
            out = f"{self.bos_token}"
            for message in messages:
                out += f"{self.start_header_token}{message.role}{self.end_header_token}\n{message.content}{self.eot_token}"
            out += f"{self.start_header_token}assistant{self.end_header_token}\n"
            return out

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(s: str, max_consecutive_slice_len: int) -> Iterator[str]:
        """
        Splits the string `s` so that each substring contains no more than `max_consecutive_slice_len`
        consecutive whitespaces or consecutive non-whitespaces.
        """
        current_slice_len = 0
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False
        slice_start = 0

        for i in range(len(s)):
            is_now_space = s[i].isspace()

            if current_slice_is_space ^ is_now_space:
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                current_slice_len += 1
                if current_slice_len > max_consecutive_slice_len:
                    yield s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1
        yield s[slice_start:]
