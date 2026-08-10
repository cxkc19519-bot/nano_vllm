import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    model_type: str = "qwen3"
    layer_types: list | None = None
    num_full_attention_layers: int = -1

    # KV Cache Compression Configurations
    compress_threshold: int = 4096
    compress_sink_tokens: int = 64
    compress_recent_window: int = 128
    compress_recent_queries: int = 32
    compress_top_k: int = 1024

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)

        # Detect model type and extract text config for Qwen3.5
        raw_type = getattr(self.hf_config, "model_type", "qwen3")
        if raw_type in ("qwen3_5", "qwen3_5_text", "qwen3_next"):
            self.model_type = "qwen3_5"
            # Qwen3.5 multimodal wraps text config; extract it
            text_cfg = getattr(self.hf_config, "text_config", self.hf_config)
            self.hf_config = text_cfg
        else:
            self.model_type = "qwen3"

        # Extract layer_types for hybrid models
        self.layer_types = getattr(self.hf_config, "layer_types", None)
        if self.layer_types is not None:
            self.num_full_attention_layers = sum(1 for t in self.layer_types if t == "full_attention")
        else:
            self.num_full_attention_layers = getattr(self.hf_config, "num_hidden_layers", 0)

        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
