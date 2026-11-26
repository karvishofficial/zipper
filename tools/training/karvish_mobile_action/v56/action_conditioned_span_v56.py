"""
KARVISH UNIVERSAL V56 — DUAL-SLOT ACTION-CONDITIONED LEARNED SPAN HEAD
Solves the Senior's Core Mandate:
Extracts exact arguments across multi-turn workflows:
- Slot 0 (RECIPIENT_OR_QUERY): e.g. contact names, search keywords, topics
- Slot 1 (MESSAGE_PAYLOAD): e.g. text messages, replies, message bodies
- Slot 2 (NUMERIC_CONSTRAINT): e.g. price limits, quantity constraints

Conditioning Signals:
  1. Goal sequence tokens: [B, L, hidden_dim]
  2. Target candidate embedding: [B, hidden_dim]
  3. Intended action representation: [B, 64]
  4. Candidate role embedding: [B, 64]
  5. Semantic slot embedding: [B, 32]
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionConditionedSpanHeadV56(nn.Module):
    """
    Action-, role-, and slot-conditioned dual-pointer network for extracting exact argument
    spans from user goal sequences.
    """

    def __init__(
        self,
        hidden_dim: int = 384,
        action_dim: int = 64,
        role_dim: int = 64,
        slot_dim: int = 32,
        num_actions: int = 8,
        num_roles: int = 32,
        num_slots: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.slot_dim = slot_dim

        # Action, role, and slot embeddings
        self.action_embed = nn.Embedding(num_actions, action_dim)
        self.role_embed = nn.Embedding(num_roles, role_dim)
        self.slot_embed = nn.Embedding(num_slots, slot_dim)

        # Candidate and context conditioning fusion
        self.cond_fusion = nn.Sequential(
            nn.Linear(hidden_dim + action_dim + role_dim + slot_dim, hidden_dim),
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
        slot_idx: Optional[torch.Tensor] = None,    # [B] long (0=Recipient/Query, 1=Payload)
    ) -> Dict[str, torch.Tensor]:
        B, L, H = goal_tokens.shape

        # 1. Embed Intended Action & Candidate Role
        act_clamped = target_action_idx.clamp(0, 7)
        role_clamped = target_role_idx.clamp(0, 31)
        e_act = self.action_embed(act_clamped)      # [B, 64]
        e_role = self.role_embed(role_clamped)     # [B, 64]

        # 2. Embed Argument Slot: default to 1 (PAYLOAD) if role is MESSAGE_INPUT (4), else 0 (RECIPIENT)
        if slot_idx is None:
            slot_idx = torch.where(
                role_clamped == 4,
                torch.ones_like(role_clamped),
                torch.zeros_like(role_clamped)
            )
        slot_clamped = slot_idx.clamp(0, 3)
        e_slot = self.slot_embed(slot_clamped)     # [B, 32]

        # 3. Fuse Candidate Embedding with Action, Role, and Slot conditioning
        cand_in = torch.cat([target_candidate_embedding, e_act, e_role, e_slot], dim=-1)
        cand_ctx = self.cond_fusion(cand_in)        # [B, hidden_dim]
        cand_ctx_exp = cand_ctx.unsqueeze(1).expand(B, L, H)  # [B, L, hidden_dim]

        # 4. Predict Start Token Logits
        start_in = torch.cat([goal_tokens, cand_ctx_exp], dim=-1) # [B, L, hidden_dim * 2]
        span_start_logits = self.start_pointer(start_in).squeeze(-1) # [B, L]

        if goal_mask is not None:
            span_start_logits = span_start_logits.masked_fill(goal_mask == 0, -1e4)

        # 5. Predict End Token Logits conditioned on soft/argmax start token
        start_probs = F.softmax(span_start_logits, dim=-1).unsqueeze(-1)  # [B, L, 1]
        soft_start_rep = (goal_tokens * start_probs).sum(dim=1, keepdim=True).expand(B, L, H)

        end_in = torch.cat([goal_tokens + soft_start_rep, cand_ctx_exp], dim=-1)
        span_end_logits = self.end_pointer(end_in).squeeze(-1)            # [B, L]

        if goal_mask is not None:
            span_end_logits = span_end_logits.masked_fill(goal_mask == 0, -1e4)

        # 6. Direction & Auxiliary Predictions
        direction_logits = self.direction_head(cand_ctx)                   # [B, 4]
        abstention_logits = self.abstention_head(cand_ctx)                 # [B, 2]
        constraint_logits = self.constraint_head(cand_ctx)                 # [B, 3]

        return {
            "span_start_logits": span_start_logits,   # [B, L]
            "span_end_logits": span_end_logits,       # [B, L]
            "direction_logits": direction_logits,     # [B, 4]
            "abstention_logits": abstention_logits,   # [B, 2]
            "constraint_logits": constraint_logits,   # [B, 3]
        }
