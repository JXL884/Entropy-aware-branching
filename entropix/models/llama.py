from entropix.model import ModelParams

_1b_params = {
    "head_dim": 64,
    "hidden_size": 2048,
    "num_attention_heads": 32,
    "num_hidden_layers": 16,
    "num_key_value_heads": 8,
    "max_seq_len": 8192,
    "use_scaled_rope": True,
    "rope_theta": 500000.0,
}
LLAMA_1B = ModelParams(
    name="llama-1b",
    hf_id="meta-llama/Llama-3.2-1B-Instruct",
    dim=_1b_params["hidden_size"],
    n_layers=_1b_params["num_hidden_layers"],
    n_local_heads=_1b_params["num_attention_heads"],
    n_local_kv_heads=_1b_params["num_key_value_heads"],
    head_dim=_1b_params["head_dim"],
    max_seq_len=_1b_params["max_seq_len"],
    rope_theta=_1b_params["rope_theta"],
    use_scaled_rope=_1b_params["use_scaled_rope"]
)

_3b_params = {
    "head_dim": 128,
    "hidden_size": 3072,
    "num_attention_heads": 24,
    "num_hidden_layers": 28,
    "num_key_value_heads": 8,
    "max_seq_len": 8192,
    "use_scaled_rope": True,
    "rope_theta": 500000.0,
}
LLAMA_3B = ModelParams(
    name="llama-3b",
    hf_id="meta-llama/Llama-3.2-3B-Instruct",
    dim=_3b_params["hidden_size"],
    n_layers=_3b_params["num_hidden_layers"],
    n_local_heads=_3b_params["num_attention_heads"],
    n_local_kv_heads=_3b_params["num_key_value_heads"],
    head_dim=_3b_params["head_dim"],
    max_seq_len=_3b_params["max_seq_len"],
    rope_theta=_3b_params["rope_theta"],
    use_scaled_rope=_3b_params["use_scaled_rope"]
)


_8b_params = {
    "head_dim": 128,
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
    "max_seq_len": 8192,
    "use_scaled_rope": True,
    "rope_theta": 500000.0,
}
LLAMA_8B = ModelParams(
    name="llama-8b",
    hf_id="meta-llama/Llama-3.1-8B-Instruct",
    dim=_8b_params["hidden_size"],
    n_layers=_8b_params["num_hidden_layers"],
    n_local_heads=_8b_params["num_attention_heads"],
    n_local_kv_heads=_8b_params["num_key_value_heads"],
    head_dim=_8b_params["head_dim"],
    max_seq_len=_8b_params["max_seq_len"],
    rope_theta=_8b_params["rope_theta"],
    use_scaled_rope=_8b_params["use_scaled_rope"]
)
