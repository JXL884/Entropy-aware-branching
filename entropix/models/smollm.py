from entropix.config import ModelConfig

_smollm_360m_params = {
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
    name="SmolLM2-360m",
    hf_id="HuggingFaceTB/SmolLM2-360M-Instruct",
    dim=_smollm_360m_params["dim"],
    n_layers=_smollm_360m_params["n_layers"],
    n_local_heads=_smollm_360m_params["n_heads"],
    n_local_kv_heads=_smollm_360m_params["n_kv_heads"],
    head_dim=_smollm_360m_params["dim"] // _smollm_360m_params["n_heads"],
    max_seq_len=_smollm_360m_params["max_seq_len"],
    rope_theta=_smollm_360m_params["rope_theta"],
    use_scaled_rope=_smollm_360m_params["use_scaled_rope"],
)

