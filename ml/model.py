from __future__ import annotations
import math
from typing import Optional
import torch
import torch.nn as nn
from .config import ModelConfig


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        T = x.size(1)
        return x + self.pe[:, :T] #type: ignore


class TransformerDecoderLM(nn.Module):
    """A decoder-only transformer for next-token prediction."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = PositionalEncoding(cfg.d_model, cfg.max_seq_len)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            batch_first=True,
            activation='gelu',
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=cfg.n_layers)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, T]
        B, T = x.shape
        tok = self.tok_emb(x)
        h = self.pos(tok)
        # causal mask for decoder
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        memory = torch.zeros(B, 1, self.cfg.d_model, device=x.device)  # dummy memory
        out = self.decoder(tgt=h, memory=memory, tgt_mask=causal_mask, tgt_key_padding_mask=(attn_mask == 0) if attn_mask is not None else None)
        logits = self.lm_head(out)
        return logits
