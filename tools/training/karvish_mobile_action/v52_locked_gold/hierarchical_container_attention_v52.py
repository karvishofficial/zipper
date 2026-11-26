"""
KARVISH V52 — PHASE 5: HIERARCHICAL CONTAINER ATTENTION & SCENE GRAPH
Solves compound multi-attribute card matching without flattening or string mashing.

Key Innovations:
1. Container-Child Cross-Attention: Card retains separate vectors for Title, Badge,
   and Description snippet, allowing independent matching against complex user goals.
2. Relational Scene Graph: 1-layer Multi-Head Self-Attention over all screen candidates
   enabling nodes to attend to competing sibling nodes.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalContainerAttention(nn.Module):
    """
    Cross-attends user goal tokens into candidate card children (Title, Badge, Snippet).
    Outputs a consolidated composite card embedding that reflects multi-attribute alignment.
    """

    def __init__(self, hidden_dim: int = 384, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Card-Child Cross-Attention: Query = Goal, Key/Value = Child Nodes
        self.child_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Fusion Gate: Combines Base Container Feature with Attended Child Signals
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 1-Layer Inter-Candidate Relational Screen Attention (Global Scene Graph)
        self.screen_self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.screen_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        candidate_embeddings: torch.Tensor,         # [B, N, hidden_dim]
        goal_embeddings: torch.Tensor,              # [B, L, hidden_dim]
        candidate_pad_mask: Optional[torch.Tensor] = None,  # [B, N] (0 for pad, 1 for valid)
        child_embeddings: Optional[torch.Tensor] = None,    # [B, N, max_children, hidden_dim]
        child_pad_mask: Optional[torch.Tensor] = None,      # [B, N, max_children]
    ) -> torch.Tensor:
        B, N, H = candidate_embeddings.shape

        # 1. Container-Child Cross-Attention (if hierarchical children exist)
        if child_embeddings is not None and child_embeddings.shape[2] > 0:
            max_c = child_embeddings.shape[2]
            # Reshape to batch across all candidates: [B * N, max_children, H]
            flat_children = child_embeddings.view(B * N, max_c, H)
            # Expand goal queries across all candidates: [B * N, L, H]
            expanded_goal = goal_embeddings.unsqueeze(1).expand(B, N, -1, -1).contiguous().view(B * N, -1, H)

            child_key_pad = None
            if child_pad_mask is not None:
                child_key_pad = (child_pad_mask.view(B * N, max_c) == 0)

            # Query = goal, Key/Value = children
            attended_children, _ = self.child_cross_attn(
                query=expanded_goal,
                key=flat_children,
                value=flat_children,
                key_padding_mask=child_key_pad,
            )
            # Pool attended child signals: [B * N, H] -> [B, N, H]
            pooled_children = attended_children.mean(dim=1).view(B, N, H)

            # Fuse with base candidate embedding
            fused = self.fusion_gate(torch.cat([candidate_embeddings, pooled_children], dim=-1))
            candidate_embeddings = candidate_embeddings + fused

        # 2. Inter-Candidate Relational Scene Graph (Self-Attention over screen)
        key_pad = (candidate_pad_mask == 0) if candidate_pad_mask is not None else None

        screen_context, _ = self.screen_self_attn(
            query=candidate_embeddings,
            key=candidate_embeddings,
            value=candidate_embeddings,
            key_padding_mask=key_pad,
        )

        output_embeddings = self.screen_norm(candidate_embeddings + screen_context)
        return output_embeddings
