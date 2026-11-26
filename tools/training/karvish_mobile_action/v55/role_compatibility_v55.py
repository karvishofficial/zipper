"""
KARVISH UNIVERSAL V55 — CONTEXTUAL ROLE COMPATIBILITY ENGINE
Solves the Senior's Core Architectural Mandate:
No element role is inherently good or bad; candidate relevance is a learned function:
    Relevance = f(Candidate Role, Current Subgoal, Context, Bounds)

Features:
- 32 Universal Semantic Roles (num_roles=32, role_dim=64).
- 12 Universal Subgoals (num_subgoals=12, subgoal_dim=64).
- Continuous 2D Spatial Geometry (bounds_dim=10, spatial_dim=64).
- Active Goal Context (hidden_dim=384).
- Dynamic Contextual Compatibility Projection into candidate ranking.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from schema_v55 import NUM_UNIVERSAL_ROLES, NUM_UNIVERSAL_SUBGOALS


class ContextualRoleCompatibilityModule(nn.Module):
    """
    Computes neural compatibility score between candidate semantic roles and the active task subgoal:
        compatibility_score = MLP([e_role || e_subgoal || e_spatial || u_active])
    """

    def __init__(
        self,
        hidden_dim: int = 384,
        role_dim: int = 64,
        subgoal_dim: int = 64,
        spatial_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.role_dim = role_dim
        self.subgoal_dim = subgoal_dim
        self.spatial_dim = spatial_dim

        # 1. Semantic Role Embeddings (32 roles)
        self.role_embed = nn.Embedding(NUM_UNIVERSAL_ROLES, role_dim)

        # 2. Subgoal Embeddings (12 universal subgoals)
        self.subgoal_embed = nn.Embedding(NUM_UNIVERSAL_SUBGOALS, subgoal_dim)

        # 3. Spatial Geometry MLP for bounds [B, N, 10] -> [B, N, 64]
        self.spatial_mlp = nn.Sequential(
            nn.Linear(10, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, spatial_dim),
            nn.LayerNorm(spatial_dim),
        )

        # 4. Goal Context Projector [B, hidden_dim] -> [B, 64]
        self.context_proj = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )

        # 5. Non-linear Contextual Compatibility Scorer
        # Input: role_dim(64) + subgoal_dim(64) + spatial_dim(64) + context_dim(64) = 256
        self.compatibility_mlp = nn.Sequential(
            nn.Linear(role_dim + subgoal_dim + spatial_dim + 64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        # Residual zero-init for stable adapter transfer (ControlNet/LoRA best practice)
        nn.init.zeros_(self.compatibility_mlp[-1].weight)
        nn.init.zeros_(self.compatibility_mlp[-1].bias)

        # 6. Candidate Role Representation Projection (compatible with candidate fusion)
        self.role_repr_proj = nn.Linear(role_dim + spatial_dim, role_dim)

    def forward(
        self,
        candidate_roles: torch.Tensor,              # [B, N] long
        candidate_bounds: torch.Tensor,             # [B, N, 10] float
        active_subgoal_idx: torch.Tensor,           # [B] long or [B, num_subgoals] distribution
        u_active: torch.Tensor,                     # [B, hidden_dim]
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            compatibility_bias: [B, N] additive relevance score for ranking
            role_embeddings: [B, N, role_dim] rich role representations
        """
        B, N = candidate_roles.shape

        # 1. Embed Roles: [B, N, 64]
        # Clamp candidate roles safely to vocabulary
        roles_clamped = candidate_roles.clamp(0, NUM_UNIVERSAL_ROLES - 1)
        e_role = self.role_embed(roles_clamped)

        # 2. Embed Spatial Geometry: [B, N, 64]
        e_spatial = self.spatial_mlp(candidate_bounds)

        # 3. Embed Active Subgoal: [B, 64]
        if active_subgoal_idx.dim() == 1:
            subgoals_clamped = active_subgoal_idx.clamp(0, NUM_UNIVERSAL_SUBGOALS - 1)
            e_subgoal = self.subgoal_embed(subgoals_clamped)
        else:
            # If soft distribution over subgoals: weighted sum
            e_subgoal = torch.matmul(active_subgoal_idx, self.subgoal_embed.weight)

        e_subgoal_exp = e_subgoal.unsqueeze(1).expand(B, N, self.subgoal_dim)

        # 4. Project Goal Context: [B, 64] -> expand to [B, N, 64]
        e_ctx = self.context_proj(u_active)
        e_ctx_exp = e_ctx.unsqueeze(1).expand(B, N, 64)

        # 5. Compute Joint Contextual Compatibility
        joint_input = torch.cat([e_role, e_subgoal_exp, e_spatial, e_ctx_exp], dim=-1)  # [B, N, 256]
        compatibility_bias = self.compatibility_mlp(joint_input).squeeze(-1)            # [B, N]

        # 6. Projected role representation
        role_repr = self.role_repr_proj(torch.cat([e_role, e_spatial], dim=-1))          # [B, N, 64]

        return {
            "compatibility_bias": compatibility_bias,  # [B, N]
            "role_embeddings": role_repr,              # [B, N, 64]
            "spatial_features": e_spatial,             # [B, N, 64]
        }
