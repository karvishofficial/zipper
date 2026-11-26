"""
KARVISH V54 — EPISODIC CANDIDATE MEMORY BANK
Solves Multi-Screen Scroll Amnesia across long-horizon e-commerce / catalog tasks.
Retains seen item representations across multi-step scroll episodes.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EpisodicCandidateMemoryBank(nn.Module):
    """
    Episodic memory bank holding the top-K observed candidates from prior scroll turns.
    Enables cross-screen comparative reasoning (e.g. comparing price on screen 2 vs screen 1).
    """

    def __init__(self, hidden_dim: int = 384, memory_size: int = 16, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        self.num_heads = num_heads

        # Cross-Attention: Viewport Candidates query Episodic Memory
        self.memory_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.mem_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        candidate_embeddings: torch.Tensor,                  # [B, N, D]
        memory_bank: Optional[torch.Tensor] = None,         # [B, K, D]
        memory_mask: Optional[torch.Tensor] = None,         # [B, K] (1 for valid, 0 for pad)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Queries episodic memory. Returns memory-augmented candidate embeddings.
        """
        B, N, D = candidate_embeddings.shape
        if memory_bank is None or memory_bank.shape[1] == 0:
            return candidate_embeddings, torch.zeros(B, 0, D, device=candidate_embeddings.device)

        key_padding_mask = (memory_mask == 0) if memory_mask is not None else None

        mem_ctx, _ = self.memory_attn(
            query=candidate_embeddings,
            key=memory_bank,
            value=memory_bank,
            key_padding_mask=key_padding_mask,
        )

        gate = self.mem_gate(torch.cat([candidate_embeddings, mem_ctx], dim=-1))
        enhanced_candidates = self.norm(candidate_embeddings + gate * mem_ctx)

        return enhanced_candidates, memory_bank

    def update_memory(
        self,
        current_candidates: torch.Tensor,                   # [B, N, D]
        current_scores: torch.Tensor,                       # [B, N]
        existing_memory: Optional[torch.Tensor] = None,     # [B, K_old, D]
        existing_mask: Optional[torch.Tensor] = None,       # [B, K_old]
        top_k_to_add: int = 4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pushes top-ranked items from the current screen into the episodic memory bank.
        """
        B, N, D = current_candidates.shape
        device = current_candidates.device

        k = min(top_k_to_add, N)
        top_scores, top_indices = torch.topk(current_scores, k=k, dim=1)
        top_idx_exp = top_indices.unsqueeze(-1).expand(B, k, D)
        new_items = torch.gather(current_candidates, 1, top_idx_exp)  # [B, k, D]
        new_mask = torch.ones(B, k, device=device)

        if existing_memory is None or existing_memory.shape[1] == 0:
            return new_items, new_mask

        all_mem = torch.cat([existing_memory, new_items], dim=1)
        all_mask = torch.cat([existing_mask, new_mask], dim=1)

        if all_mem.shape[1] > self.memory_size:
            all_mem = all_mem[:, -self.memory_size :, :]
            all_mask = all_mask[:, -self.memory_size :]

        return all_mem, all_mask
