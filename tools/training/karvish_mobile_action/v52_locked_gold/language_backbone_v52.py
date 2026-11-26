"""
KARVISH V52 — PHASE 3: PRE-TRAINED COMPACT LANGUAGE BACKBONE
Integrates all-MiniLM-L6-v2 (22.7M parameters) for zero-shot semantic understanding.

Key Specifications:
- Hidden Dimension: 384
- Transformer Layers: 6
- Attention Heads: 12
- Feedforward Dimension: 1536
- Parameter Budget: ~22.7M parameters (~22.7 MB in INT8, ~45.4 MB in FP16)
- Solves: Vocabulary gaps on unseen apps, natural language synonym mapping (Explore == Discover).
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
                self.hf_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
                self.pooler = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Tanh(),
                )
                return
            except Exception as e:
                print(f"[MiniLMTextEncoder] Pretrained load failed ({e}), falling back to scratch init.")
                self.pretrained = False

        # Subword Token & Position Embeddings (scratch fallback)
        self.word_embeddings = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_seq_len, hidden_dim)
        self.token_type_embeddings = nn.Embedding(2, hidden_dim)

        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

        # 6-layer Transformer Encoder
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

        # Sentence Pooling Projection
        self.pooler = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        input_ids: torch.Tensor,                # [B, L]
        attention_mask: Optional[torch.Tensor] = None,  # [B, L] (1 for valid, 0 for pad)
        token_type_ids: Optional[torch.Tensor] = None,  # [B, L]
    ) -> Dict[str, torch.Tensor]:
        if getattr(self, "pretrained", False) and hasattr(self, "hf_model"):
            out = self.hf_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            hidden_states = out.last_hidden_state  # [B, L, 384]
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
                sum_embeddings = (hidden_states * mask_expanded).sum(dim=1)
                sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
                pooled_output = sum_embeddings / sum_mask
            else:
                pooled_output = hidden_states.mean(dim=1)
            pooled_output = self.pooler(pooled_output)
            return {
                "last_hidden_state": hidden_states,
                "pooled_output": pooled_output,
            }
        B, L = input_ids.shape
        device = input_ids.device

        pos_ids = torch.arange(L, dtype=torch.long, device=device).unsqueeze(0).expand(B, L)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_emb = self.word_embeddings(input_ids)
        pos_emb = self.position_embeddings(pos_ids)
        type_emb = self.token_type_embeddings(token_type_ids)

        embeddings = self.layer_norm(words_emb + pos_emb + type_emb)
        embeddings = self.dropout(embeddings)

        # Key padding mask: True indicates positions to be ignored in PyTorch Transformer
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None

        # [B, L, hidden_dim]
        hidden_states = self.transformer(embeddings, src_key_padding_mask=key_padding_mask)

        # Mean pooling over valid tokens for sentence-level representation
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
            sum_embeddings = (hidden_states * mask_expanded).sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        else:
            pooled_output = hidden_states.mean(dim=1)

        pooled_output = self.pooler(pooled_output)

        return {
            "last_hidden_state": hidden_states,     # Token-level representations [B, L, 384]
            "pooled_output": pooled_output,         # Sentence vector [B, 384]
        }
