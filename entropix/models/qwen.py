from entropix.model import ModelParams

_1b_params = {
    "head_dim": 128,
    "hidden_size": 1536,
    "num_attention_heads": 12,
    "num_hidden_layers": 28,
    "num_key_value_heads": 2,
    "max_seq_len": 8192,
    "use_scaled_rope": False,
    "rope_theta": 1000000.0,
}


Qwen_1B = ModelParams(
    name="Qwen-1b",
    hf_id="Qwen/Qwen2.5-1.5B-Instruct",
    dim=_1b_params["hidden_size"],
    n_layers=_1b_params["num_hidden_layers"],
    n_local_heads=_1b_params["num_attention_heads"],
    n_local_kv_heads=_1b_params["num_key_value_heads"],
    head_dim=_1b_params["head_dim"],
    max_seq_len=_1b_params["max_seq_len"],
    rope_theta=_1b_params["rope_theta"],
    use_scaled_rope=_1b_params["use_scaled_rope"]
)
