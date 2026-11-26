"""
KARVISH V54 — MULTIPLICATIVE SEMANTIC ENTITY GATING HEAD
Eliminates False-Positive Distractor Selections (Ginger, Soap, Sponsored items, Feed cards).

Mathematical Foundation:
Given target goal intent vector u_target in R^D and candidate representations v_i in R^{N x D}:
1. Normalized Cosine Alignment:
   cos_sim(u, v_i) = (u . v_i) / (||u|| * ||v_i|| + eps)
2. Temperature-Scaled Sigmoid Gate:
   gate_i = sigmoid((cos_sim(u, v_i) - theta) / tau) in (0, 1)
   where tau is a learnable scaling temperature and theta is a learnable relevance threshold.

Guarantees:
- If a candidate card does NOT match the requested entity (e.g. Ginger / Soap when goal is Tomato),
  gate_i -> 0.0.
- Unrelated items are blocked from receiving comparative price boosts or falsely outranking the target entity.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EntityGatingHead(nn.Module):
    """
    Differentiable semantic entity gating module.
    Prevents unrelated cross-sell candidates from winning comparative numerical ranking.
    """

    def __init__(self, hidden_dim: int = 384, init_tau: float = 0.2, init_theta: float = 0.25):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Target Entity Query Projector
        self.entity_query_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Candidate Key Projector for Entity Matching
        self.cand_key_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Learnable Temperature and Threshold parameters
        self.log_tau = nn.Parameter(torch.tensor(math.log(init_tau), dtype=torch.float32))
        self.theta = nn.Parameter(torch.tensor(init_theta, dtype=torch.float32))

    def forward(
        self,
        goal_intent: torch.Tensor,                  # [B, hidden_dim]
        candidate_embeddings: torch.Tensor,         # [B, N, hidden_dim]
        candidate_pad_mask: Optional[torch.Tensor] = None,  # [B, N]
    ) -> Dict[str, torch.Tensor]:
        B, N, D = candidate_embeddings.shape

        # 1. Project to entity alignment space
        q_entity = self.entity_query_proj(goal_intent)             # [B, D]
        q_norm = F.normalize(q_entity, p=2, dim=-1)                # [B, D]

        k_cand = self.cand_key_proj(candidate_embeddings)          # [B, N, D]
        k_norm = F.normalize(k_cand, p=2, dim=-1)                  # [B, N, D]

        # 2. Compute Cosine Similarity
        # [B, 1, D] x [B, D, N] -> [B, 1, N] -> [B, N]
        cos_sim = torch.bmm(q_norm.unsqueeze(1), k_norm.transpose(1, 2)).squeeze(1)

        # 3. Apply Temperature and Threshold
        tau = torch.exp(self.log_tau).clamp(min=0.05, max=1.0)
        gate = torch.sigmoid((cos_sim - self.theta) / tau)         # [B, N] in (0, 1)

        if candidate_pad_mask is not None:
            gate = gate * candidate_pad_mask.float()
            cos_sim = cos_sim.masked_fill(candidate_pad_mask == 0, -1.0)

        return {
            "entity_gate": gate,       # [B, N] Multiplicative gate in (0, 1)
            "entity_sim": cos_sim,     # [B, N] Raw cosine similarities in [-1, 1]
            "tau": tau,
            "theta": self.theta,
        }
