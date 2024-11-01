import os
from pathlib import Path
from unittest.mock import patch

import jax.numpy as jnp
import ml_dtypes
import torch
from transformers import AutoModelForCausalLM
from transformers.dynamic_module_utils import get_imports

from entropix.config import ModelConfig

_smollm_params = {
    "dim": 960,
    "n_layers": 32,
    "n_heads": 15,
    "n_kv_heads": 5,
    "vocab_size": 49152,
    "norm_eps": 1e-05,
    "rope_theta": 10000.0,
    "use_scaled_rope": False,  # Inferred from "rope_scaling": null
    "max_seq_len": 2048,  # Inferred from "max_position_embeddings"
}

SMOLLM_360M = ModelConfig(
    name="smollm",
    hf_id="HuggingFaceTB/SmolLM2-360M-Instruct",
    n_layers=_smollm_params["n_layers"],
    n_local_heads=_smollm_params["n_heads"],
    n_local_kv_heads=_smollm_params["n_kv_heads"],
    head_dim=_smollm_params["dim"] // _smollm_params["n_heads"],
    max_seq_len=_smollm_params["max_seq_len"],
    rope_theta=_smollm_params["rope_theta"],
    use_scaled_rope=_smollm_params["use_scaled_rope"]
)

def translate_key(in_key: str):
    out_key = in_key.replace('.weight', '')
    if out_key.startswith('model.'):
        out_key = out_key.replace('model.', '')
        if out_key.endswith('input_layernorm'):
            out_key = out_key.replace('input_layernorm', 'attention_norm')
        elif out_key.endswith('mlp.down_proj'):
            out_key = out_key.replace('mlp.down_proj', 'feed_forward.w2')
        elif out_key.endswith('mlp.gate_proj'):
            out_key = out_key.replace('mlp.gate_proj', 'feed_forward.w1')
        elif out_key.endswith('mlp.up_proj'):
            out_key = out_key.replace('mlp.up_proj', 'feed_forward.w3')
        elif out_key.endswith('post_attention_layernorm'):
            out_key = out_key.replace('post_attention_layernorm', 'ffn_norm')
        elif out_key.endswith('self_attn.k_proj'):
            out_key = out_key.replace('self_attn.k_proj', 'attention.wk')
        elif out_key.endswith('self_attn.o_proj'):
            out_key = out_key.replace('self_attn.o_proj', 'attention.wo')
        elif out_key.endswith('self_attn.q_proj'):
            out_key = out_key.replace('self_attn.q_proj', 'attention.wq')
        elif out_key.endswith('self_attn.v_proj'):
            out_key = out_key.replace('self_attn.v_proj', 'attention.wv')
        elif out_key.endswith('down_proj'):
            out_key = out_key.replace('down_proj', 'w2')
        elif out_key.endswith('gate_proj'):
            out_key = out_key.replace('gate_proj', 'w1')
        elif out_key.endswith('up_proj'):
            out_key = out_key.replace('up_proj', 'w3')
        elif out_key == 'embed_tokens':
            out_key = 'tok_embeddings'
        elif out_key == 'norm':
            out_key = 'norm'
        else:
            print(f"Don't know how to handle {in_key=}")
    elif out_key == 'lm_head':
        out_key = 'output'
    else:
        print(f"Don't know how to handle {in_key=}")
    return f'{out_key}.weight'

def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    """Work around for https://huggingface.co/microsoft/phi-1_5/discussions/72."""
    if not str(filename).endswith("/modeling_deepseek.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports

def reverse_permute(tensor: torch.Tensor, n_heads: int = 15, dim1: int = 960, dim2: int = 960) -> torch.Tensor:
    return tensor.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

def download_weights(model_cfg: ModelConfig, out_dir: Path | str | None = None):
    if out_dir is None: out_dir = Path(f"weights/{model_cfg.name}")
    elif isinstance(out_dir, str): out_dir = Path(out_dir)

    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        hf_model = AutoModelForCausalLM.from_pretrained(model_cfg.hf_id, torch_dtype=torch.bfloat16, offload_folder="/tmp/offload", device_map='cpu')
        with torch.no_grad():
            state_dict = hf_model.state_dict()
            for hf_name, param in state_dict.items():
                name = translate_key(hf_name)
                param = param.cpu()
                if name.endswith('wq.weight'):
                    param = reverse_permute(param, n_heads=15, dim1=960, dim2=960)  # 362M
                elif name.endswith('wk.weight'):
                    param = reverse_permute(param, n_heads=5, dim1=320, dim2=960)  # 362M
                else:
                    pass
                bf16_np_out = param.cpu().view(dtype=torch.uint16).numpy().view(ml_dtypes.bfloat16)
                bf16_out = jnp.asarray(bf16_np_out, dtype=jnp.bfloat16).reshape(*param.shape)
                print(f'Writing {hf_name} as {name} to {out_dir}/{name}.npy')
                jnp.save(f'{out_dir}/{name}.npy', bf16_out)
    del hf_model
    del state_dict
