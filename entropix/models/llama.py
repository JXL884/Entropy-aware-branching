from entropix.model import ModelParams

_llama_params = {
    "dim": 2048,
    "n_layers": 16,
    "n_heads": 32,
    "n_kv_heads": 8,
    "vocab_size": 128256,
    "ffn_dim_multiplier": 1.5,
    "multiple_of": 256,
    "norm_eps": 1e-05,
    "rope_theta": 500000.0,
    "use_scaled_rope": True,
    "max_seq_len": 4096,
}

LLAMA_1B = ModelParams(
    name="llama-1b",
    hf_id="meta-llama/Llama-3.2-1B-Instruct",
    dim=_llama_params["dim"],
    n_layers=_llama_params["n_layers"],
    n_local_heads=_llama_params["n_heads"],
    n_local_kv_heads=_llama_params["n_kv_heads"],
    head_dim=_llama_params["dim"] // _llama_params["n_heads"],
    max_seq_len=_llama_params["max_seq_len"],
    rope_theta=_llama_params["rope_theta"],
    use_scaled_rope=_llama_params["use_scaled_rope"]
)
