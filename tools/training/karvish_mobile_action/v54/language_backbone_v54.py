"""
KARVISH V54 — COMPACT LANGUAGE BACKBONE
Integrates all-MiniLM-L6-v2 (22.7M parameters) for zero-shot semantic understanding.

Key Specifications:
- Hidden Dimension: 384
- Transformer Layers: 6
- Attention Heads: 12
- Feedforward Dimension: 1536
- Parameter Budget: ~22.7M parameters
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniLMTextEncoder(nn.Module):
    """
    Compact MiniLM-L6 language backbone.
    Produces contextual 384-dimensional token and pooled sentence representations.
    Loads official pre-trained sentence-transformers/all-MiniLM-L6-v2 weights.
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_dim: int = 384,
        num_layers: int = 6,
        num_heads: int = 12,
        dim_feedforward: int = 1536,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        pretrained: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.pretrained = pretrained

        if pretrained:
            try:
                from transformers import AutoModel
                self.hf_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", attn_implementation="eager")
                self.pooler = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Tanh(),
                )
                return
            except Exception as e:
                print(f"[MiniLMTextEncoder] Pretrained load failed ({e}), falling back to scratch init.")
                self.pretrained = False

        # Fallback scratch init
        self.token_embeddings = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_seq_len, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pooler = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.pretrained and hasattr(self, "hf_model"):
            hf_out = self.hf_model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = hf_out.last_hidden_state  # [B, L, hidden_dim]
            if attention_mask is not None:
                mask_exp = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
                sum_embeddings = (last_hidden * mask_exp).sum(dim=1)
                sum_mask = mask_exp.sum(dim=1).clamp(min=1e-9)
                pooled = sum_embeddings / sum_mask
            else:
                pooled = last_hidden[:, 0]
            pooled = self.pooler(pooled)
            return {"last_hidden_state": last_hidden, "pooled_output": pooled}

        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        embeddings = self.token_embeddings(input_ids) + self.position_embeddings(pos)
        embeddings = self.dropout(self.layer_norm(embeddings))

        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)

        last_hidden = self.transformer(embeddings, src_key_padding_mask=src_key_padding_mask)

        if attention_mask is not None:
            mask_exp = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
            sum_embeddings = (last_hidden * mask_exp).sum(dim=1)
            sum_mask = mask_exp.sum(dim=1).clamp(min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            pooled = last_hidden[:, 0]

        pooled = self.pooler(pooled)
        return {"last_hidden_state": last_hidden, "pooled_output": pooled}
