"""
KARVISH UNIVERSAL V55 — ACTION-CONDITIONED LEARNED SPAN HEAD
Fulfills Phase D & Universal E-Commerce Mandates:
When typing into a SEARCH_INPUT or MESSAGE_INPUT, the pointer head extracts the
precise entity argument (e.g. 'tomato' or 'SpaceX'), while routing superlative
sorting modifiers ('cheapest') to the numerical constraint ranking head.

Conditioning Signals:
  1. Goal sequence tokens: [B, L, hidden_dim]
  2. Target candidate embedding: [B, hidden_dim]
  3. Intended action representation: [B, 64]
  4. Candidate role embedding: [B, 64]
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionConditionedSpanHead(nn.Module):
    """
    Action- and role-conditioned dual-pointer network for extracting exact argument
    spans from user goal sequences.
    """

    def __init__(
        self,
        hidden_dim: int = 384,
        action_dim: int = 64,
        role_dim: int = 64,
        num_actions: int = 8,
        num_roles: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Action and role embeddings
        self.action_embed = nn.Embedding(num_actions, action_dim)
        self.role_embed = nn.Embedding(num_roles, role_dim)

        # Candidate and context conditioning fusion
        self.cond_fusion = nn.Sequential(
            nn.Linear(hidden_dim + action_dim + role_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Start Token Pointer Network: [B, L, hidden_dim * 2] -> [B, L, 1]
        self.start_pointer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # End Token Pointer Network: conditioned on start token representation
        self.end_pointer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Direction Classifier for Scroll Actions: [B, hidden_dim] -> [B, 4]
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 4),  # DOWN, UP, LEFT, RIGHT
        )

        # Argument Abstention Head: 0 = No argument needed, 1 = Argument required
        self.abstention_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 2),
        )

        # Constraint Type Head: NONE (0), CHEAPEST (1), HIGHEST_RATED (2)
        self.constraint_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 3),
        )

    def forward(
        self,
        goal_tokens: torch.Tensor,                  # [B, L, hidden_dim]
        target_candidate_embedding: torch.Tensor,   # [B, hidden_dim]
        target_action_idx: torch.Tensor,            # [B] long
        target_role_idx: torch.Tensor,              # [B] long
        goal_mask: Optional[torch.Tensor] = None,   # [B, L]
    ) -> Dict[str, torch.Tensor]:
        B, L, H = goal_tokens.shape

        e_act = self.action_embed(target_action_idx.clamp(0, 7))  # [B, 64]
        e_role = self.role_embed(target_role_idx.clamp(0, 31))    # [B, 64]

        # Conditioned context vector
        cond_in = torch.cat([target_candidate_embedding, e_act, e_role], dim=-1)  # [B, H + 128]
        u_cond = self.cond_fusion(cond_in)                                       # [B, H]
        u_cond_exp = u_cond.unsqueeze(1).expand(B, L, H)                         # [B, L, H]

        # 1. Joint Start Pointer
        joint_features = torch.cat([goal_tokens, u_cond_exp], dim=-1)            # [B, L, H * 2]
        start_logits = self.start_pointer(joint_features).squeeze(-1)            # [B, L]

        # 2. Joint End Pointer
        end_logits = self.end_pointer(joint_features).squeeze(-1)                # [B, L]

        if goal_mask is not None:
            start_logits = start_logits.masked_fill(goal_mask == 0, -1e4)
            end_logits = end_logits.masked_fill(goal_mask == 0, -1e4)

        # 3. Direction Logits for Scroll
        direction_logits = self.direction_head(u_cond)                           # [B, 4]

        # 4. Abstention Logits
        abstention_logits = self.abstention_head(u_cond)                         # [B, 2]

        # 5. Goal Constraint Classification (CHEAPEST, HIGHEST_RATED, NONE)
        constraint_logits = self.constraint_head(u_cond)                         # [B, 3]

        return {
            "start_logits": start_logits,
            "end_logits": end_logits,
            "direction_logits": direction_logits,
            "abstention_logits": abstention_logits,
            "constraint_logits": constraint_logits,
            "u_cond": u_cond,
        }
