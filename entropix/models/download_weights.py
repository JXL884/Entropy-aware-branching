import torch
import ml_dtypes
import jax.numpy as jnp
from pathlib import Path

from transformers import AutoModelForCausalLM
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports

from entropix.model import ModelParams

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

# def reverse_permute(tensor: torch.Tensor, n_heads: int = 32, dim1:int = 4096, dim2: int = 4096) -> torch.Tensor:
def reverse_permute(tensor: torch.Tensor, n_heads: int, dim1: int, dim2: int) -> torch.Tensor:
    return tensor.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

def fixed_get_imports(filename: str | Path) -> list[str]:
    """Work around for https://huggingface.co/microsoft/phi-1_5/discussions/72."""
    if not str(filename).endswith("/modeling_deepseek.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports

def download_weights(
    model_cfg: ModelParams,
    out_dir: Path | str | None = None,
    download_tokenizer: bool = True,
    tokenizer_out_path: Path | str | None = None,
    tokenizer_url: str | None = None,
    tokenizer_cfg_out_path: Path | str | None = None,
    tokenizer_cfg_url: str | None = None
):
    skip_download = False
    if out_dir is None: out_dir = Path(f"weights/{model_cfg.name}")
    elif isinstance(out_dir, str): out_dir = Path(out_dir)
    if not out_dir.exists(): out_dir.mkdir(parents=True, exist_ok=True)
    elif input(f"{out_dir} already exists, re-download? [y/N] ").lower() != "y": skip_download = True


    token = None
    t_paths_candidates = [Path.home() / '.hf_token', Path.home() / '.cache' / 'huggingface' / 'token']
    for t_path in t_paths_candidates:
        if t_path.exists():
            token = t_path.read_text().strip()
            break

    if not skip_download:
        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_cfg.hf_id, torch_dtype=torch.bfloat16, offload_folder="/tmp/offload", token=token, device_map="cpu"
            )
            with torch.no_grad():
                state_dict = hf_model.state_dict()
                for hf_name, param in state_dict.items():
                    # print(f' {hf_name}: {param.shape=}')
                    name = translate_key(hf_name)
                    param.cpu()
                    if name.endswith('wq.weight'):
                        param = reverse_permute(param, n_heads=model_cfg.n_local_heads, dim1=model_cfg.dim, dim2=model_cfg.dim)
                    elif name.endswith('wk.weight'):
                        dim1 = model_cfg.head_dim * model_cfg.n_local_kv_heads
                        dim2 = model_cfg.dim
                        param = reverse_permute(param, n_heads=model_cfg.n_local_kv_heads, dim1=dim1, dim2=dim2)
                    else:
                        pass
                    bf16_np_out = param.cpu().view(dtype=torch.uint16).numpy().view(ml_dtypes.bfloat16)
                    bf16_out = jnp.asarray(bf16_np_out, dtype=jnp.bfloat16).reshape(*param.shape)
                    print(f'Writing {hf_name} as {name} to {out_dir}/{name}.npy')
                    jnp.save(f'{out_dir}/{name}.npy', bf16_out)

    if download_tokenizer:
        import requests

        if tokenizer_out_path is None: tokenizer_out_path = Path(f"weights/tokenizers/{model_cfg.name}.json")
        elif isinstance(tokenizer_out_path, str): tokenizer_out_path = Path(tokenizer_out_path)
        if tokenizer_cfg_out_path is None: tokenizer_cfg_out_path = Path(f"weights/tokenizers/{model_cfg.name}_config.json")
        elif isinstance(tokenizer_cfg_out_path, str): tokenizer_cfg_out_path = Path(tokenizer_cfg_out_path)

        if not tokenizer_out_path.parent.exists(): tokenizer_out_path.parent.mkdir(parents=True, exist_ok=True)
        if tokenizer_out_path.exists() and input(f"{tokenizer_out_path} already exists, re-download? [y/N] ").lower() != "y": return

        if not tokenizer_url: tokenizer_url = f"https://huggingface.co/{model_cfg.hf_id}/resolve/main/tokenizer.json"
        if not tokenizer_cfg_url: tokenizer_cfg_url = f"https://huggingface.co/{model_cfg.hf_id}/resolve/main/tokenizer_config.json"

        print(f"Downloading tokenizer from {tokenizer_url} to {tokenizer_out_path}...")

        headers = {"Authorization": f"Bearer {token}"}

        response = requests.get(tokenizer_url, headers=headers)
        if response.status_code == 200:
            with open(tokenizer_out_path, 'wb') as f:
                f.write(response.content)
            print(f"Tokenizer downloaded to {tokenizer_out_path}")
        else:
            print(f"Failed to download tokenizer. Status code: {response.status_code}")

        response = requests.get(tokenizer_cfg_url, headers=headers)
        if response.status_code == 200:
            with open(tokenizer_cfg_out_path, 'wb') as f:
                f.write(response.content)
            print(f"Tokenizer config downloaded to {tokenizer_cfg_out_path}")
        else:
            print(f"Failed to download tokenizer config. Status code: {response.status_code}")
