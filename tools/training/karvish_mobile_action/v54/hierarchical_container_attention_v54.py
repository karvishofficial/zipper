"""
KARVISH V54 — HIERARCHICAL CONTAINER ATTENTION & CARD BINDING
Binds Parent Product Cards to Actionable Child Buttons (e.g. "ADD" button context binding).

Key Innovations:
1. Parent-to-Child Signal Propagation:
   Actionable children (e.g. "ADD" buttons, quantity steppers) absorb their parent card's
   title, brand, and price embeddings, eliminating button identity blindness.
2. Container-Child Cross-Attention:
   Card retains separate vectors for Title, Badge, and Price, allowing independent matching.
3. Relational Scene Graph:
   Inter-candidate self-attention across all screen elements.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalContainerAttention(nn.Module):
    """
    Hierarchical Container and Scene Graph Attention Module.
    Propagates parent card context to child buttons and models inter-card relations.
    """

    def __init__(self, hidden_dim: int = 384, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Parent-to-Child Context Projection
        self.parent_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Container-Child Cross-Attention: Query = Goal, Key/Value = Child Nodes
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
        candidate_embeddings: torch.Tensor,                 # [B, N, hidden_dim]
        goal_embeddings: torch.Tensor,                      # [B, L, hidden_dim]
        candidate_pad_mask: Optional[torch.Tensor] = None,  # [B, N] (0 for pad, 1 for valid)
        parent_indices: Optional[torch.Tensor] = None,      # [B, N] index of parent card (-1 if none)
        child_embeddings: Optional[torch.Tensor] = None,    # [B, N, max_children, hidden_dim]
        child_pad_mask: Optional[torch.Tensor] = None,      # [B, N, max_children]
    ) -> torch.Tensor:
        B, N, H = candidate_embeddings.shape

        # 1. Parent-to-Child Context Propagation
        # If candidate i has a parent container, inject parent representation into child
        if parent_indices is not None:
            # Clamp indices for safe gathering: replace -1 with 0
            safe_parent_idx = parent_indices.clamp(min=0).unsqueeze(-1).expand(B, N, H)
            parent_emb = candidate_embeddings.gather(1, safe_parent_idx)  # [B, N, H]
            parent_features = self.parent_proj(parent_emb)

            has_parent = (parent_indices >= 0).unsqueeze(-1).to(candidate_embeddings.dtype)  # [B, N, 1]
            candidate_embeddings = candidate_embeddings + parent_features * has_parent

        # 2. Container-Child Cross-Attention (if hierarchical children exist)
        if child_embeddings is not None and child_embeddings.shape[2] > 0:
            max_c = child_embeddings.shape[2]
            flat_children = child_embeddings.view(B * N, max_c, H)
            expanded_goal = goal_embeddings.unsqueeze(1).expand(B, N, -1, -1).contiguous().view(B * N, -1, H)

            child_key_pad = None
            if child_pad_mask is not None:
                child_key_pad = (child_pad_mask.view(B * N, max_c) == 0)

            attended_children, _ = self.child_cross_attn(
                query=expanded_goal,
                key=flat_children,
                value=flat_children,
                key_padding_mask=child_key_pad,
            )
            pooled_children = attended_children.mean(dim=1).view(B, N, H)
            fused = self.fusion_gate(torch.cat([candidate_embeddings, pooled_children], dim=-1))
            candidate_embeddings = candidate_embeddings + fused

        # 3. Inter-Candidate Relational Scene Graph (Self-Attention over screen)
        # Export-friendly dynamic multi-head attention that never bakes a constant N into ONNX
        qkv = F.linear(
            candidate_embeddings,
            self.screen_self_attn.in_proj_weight,
            self.screen_self_attn.in_proj_bias
        )
        q, k, v = qkv.chunk(3, dim=-1)
        num_heads = self.screen_self_attn.num_heads
        head_dim = H // num_heads

        q = q.reshape(B, -1, num_heads, head_dim).transpose(1, 2)
        k = k.reshape(B, -1, num_heads, head_dim).transpose(1, 2)
        v = v.reshape(B, -1, num_heads, head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        if candidate_pad_mask is not None:
            # candidate_pad_mask: [B, N] -> [B, 1, 1, N]
            scores = scores.masked_fill((candidate_pad_mask == 0).unsqueeze(1).unsqueeze(2), -1e4)
        attn_weights = F.softmax(scores, dim=-1)
        screen_context = torch.matmul(attn_weights, v)
        screen_context = screen_context.transpose(1, 2).reshape(B, -1, H)
        screen_context = self.screen_self_attn.out_proj(screen_context)

        output_embeddings = self.screen_norm(candidate_embeddings + screen_context)
        return output_embeddings
