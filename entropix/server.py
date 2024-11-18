from contextlib import asynccontextmanager
import uvicorn
import tyro
import os
import time
import json
from functools import lru_cache
import logging
from pathlib import Path
from typing import Generator
import uuid

import torch
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, field_validator
import asyncio
import uvloop
from entropix.config import SamplerConfig
from entropix.models import LLAMA_1B, SMOLLM_360M
from entropix.model import generate, load_weights, Model, stream
from entropix.tokenizer import Tokenizer, Message

model_map = {
    "llama-1b": LLAMA_1B,
    "smollm": SMOLLM_360M,
}

class ServerArgs(BaseModel):
    model: str = "llama-1b"  # default model name
    host: str = "127.0.0.1"
    port: int = 1337
    log_level: str = "info"
    weights: str | Path | None = None
    tokenizer: str | Path | None = None

    def model_post_init(self, __context) -> None:
        if self.weights is None: self.weights = Path(f"weights/{self.model}")
        elif isinstance(self.weights, str): self.weights = Path(self.weights)
        if self.tokenizer is None: self.tokenizer = Path(f"weights/tokenizers/{self.model}.json")
        elif isinstance(self.tokenizer, str): self.tokenizer = Path(self.tokenizer)


class ChatRequest(BaseModel):
    # https://platform.openai.com/docs/api-reference/chat
    # omitted some openai specific stuff (store, metadata, etc)

    messages: list[Message]
    save_path: str | None = None
    model: str | None = None  # currently just preloading and serving a model through server args
    frequency_penalty: float = 0.0
    logit_bias: dict[int, float] | None = None
    logprobs: bool = False
    top_logprobs: int | None = None
    max_completion_tokens: int | None = None
    n: int = 1
    presence_penalty: float = 0.0
    response_format: dict | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    stream_options: dict | None = None
    temperature: float = 1.0
    top_p: float = 1.0
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None

    # NOTE: may want to change limits, just using OpenAI values for now

    @field_validator('frequency_penalty', 'presence_penalty')
    def validate_penalty(cls, v):
        if not -2.0 <= v <= 2.0: raise ValueError("penalties must be between -2.0 and 2.0")
        return v

    @field_validator('temperature', 'top_p')
    def validate_sampling_params(cls, v):
        if not 0 <= v <= 2: raise ValueError("temperature and top_p must be between 0 and 2")
        return v

    @field_validator('n')
    def validate_n(cls, v):
        if v < 1: raise ValueError("n must be >=1")
        return v

    @field_validator('top_logprobs')
    def validate_top_logprobs(cls, v):
        if v is not None and not 0 <= v <= 20:  # TODO: make this full vocab length (or just don't validate?)
            raise ValueError("top_logprobs must be between 0 and 20")
        return v

class ModelManager:
    def __init__(self, model_name: str):
        self.weights_path = "weights"  # NOTE: hardcoded
        self.tokenizer_path = "weights/tokenizers"  # NOTE: hardcoded
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model(model_name)

    def _load_model(self, model_name: str):
        if self.model is not None:
            del self.model
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        model_params = model_map[model_name]
        tokenizer = Tokenizer(f"{self.tokenizer_path}/{model_params.name}.json")
        weights = load_weights(f"{self.weights_path}/{model_params.name}", model_params)
        self.model = Model(weights, model_params, tokenizer)

    def get_model(self, requested_model: str | None = None):
        if (requested_model is not None and self.model is not None and requested_model != self.model.params.name):
            self._load_model(requested_model)
        assert self.model is not None
        return self.model

_model_manager: ModelManager | None = None

@lru_cache
def get_model_manager(model: str = "llama-1b") -> ModelManager:
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager(model)
    return _model_manager

@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: RUF029
    global _model_manager
    logging.info("Loading default model...")
    _model_manager = ModelManager("llama-1b")
    yield
    if _model_manager is not None:
        del _model_manager.model
        if torch.cuda.is_available(): torch.cuda.empty_cache()

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
logging.basicConfig(level=logging.INFO)

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health() -> Response:
    return Response(status_code=200)

def generate_response(messages: list[Message], model: Model, sampler_cfg: SamplerConfig, save_path: str | None, max_tokens: int | None) -> JSONResponse:
    uid = str(uuid.uuid4())
    created_at = int(time.time())
    gen = generate(messages, model, sampler_cfg, max_tokens=max_tokens)
    # NOTE: only one choice
    choices = [{"index": 0, "message": {"role": "assistant", "content": gen.response}, "finish_reason": "stop"}]
    if save_path:
        # with open(f"{save_path}/{created_at}.json", "w") as f:
        #     json.dump(gen.to_dict(), f)
        gen.save(f"{save_path}/{created_at}.json")
        logging.info(f"Saved response to {save_path}/{created_at}.json")
    return JSONResponse(
        content=dict(
            id=uid,
            object="chat.completion",
            created=created_at,
            model=model.params.name,
            choices=choices,
            usage=dict(
                prompt_tokens=None,  # TODO, don't want to tokenize twice
                completion_tokens=len(gen.tokens),
                total_tokens=None,
            )
        )
    )

def stream_response(messages: list[Message], model: Model, sampler_cfg: SamplerConfig, save_path: str | None, max_tokens: int | None) -> Generator[str, None, None]:
    uid = str(uuid.uuid4())
    created_at = int(time.time())

    for token, token_metrics, sampler_state, gen in stream(messages, model, sampler_cfg, max_tokens=max_tokens):
        choices = [
            dict(
                index=0,
                delta={"role": "assistant", "content": token},
                logprobs=None,  # TODO
                finish_reason=None if token else "stop",
            )
        ]
        data = dict(
            id=uid,
            object="chat.completion.chunk",
            created=created_at,
            model=model.params.name,
            choices=choices,
        )
        yield f"data: {json.dumps(data)}\n\n"


    if save_path and gen is not None:
        # with open(f"{save_path}/{created_at}.json", "w") as f:
        #     json.dump(gen.to_dict(), f)
        gen.save(f"{save_path}/{created_at}.json")
        logging.info(f"Saved streaming response to {save_path}/{created_at}.json")

@app.post("/v1/chat/completions")
async def openai_chat_completions(request: ChatRequest, model_manager: ModelManager = Depends(get_model_manager)):
    model = model_manager.get_model(request.model)
    sampler_cfg = SamplerConfig(temperature=request.temperature, top_p=request.top_p)
    save_path = os.path.expanduser(request.save_path) if request.save_path else None
    if request.stream:
        return StreamingResponse(stream_response(request.messages, model, sampler_cfg, save_path, request.max_completion_tokens))
    else:
        return generate_response(request.messages, model, sampler_cfg, save_path, request.max_completion_tokens)


def main(server_args: ServerArgs = tyro.cli(ServerArgs)):
    uvicorn.run(
        app,
        host=server_args.host,
        port=server_args.port,
        log_level=server_args.log_level,
        timeout_keep_alive=5,
        loop="uvloop",
    )

if __name__ == "__main__":
    server_args = tyro.cli(ServerArgs)
    uvicorn.run(
        "server:app",
        host=server_args.host,
        port=server_args.port,
        log_level=server_args.log_level,
        timeout_keep_alive=5,
        loop="uvloop",
        reload=True,
    )
