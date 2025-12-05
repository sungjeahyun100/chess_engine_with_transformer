from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 512
    tie_weights: bool = True


@dataclass
class TrainConfig:
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 2
    warmup_steps: int = 200
    grad_clip: float = 1.0
    save_dir: str = "checkpoints"
    mixed_precision: bool = True
