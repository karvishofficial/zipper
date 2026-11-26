"""
KARVISH V54 — LEARNED SPAN / COPY-POINTER HEAD
Fulfills Phase D Constitutional Mandate (AGENTS.md Section 7.5).
Completely eliminates Python regex (extract_entity_from_goal) in favor of learned span extraction.

Key Outputs:
1. Span Start Logits: [B, L] distribution over goal tokens.
2. Span End Logits: [B, L] distribution over goal tokens.
3. Scroll Direction Logits: [B, 4] (DOWN, UP, LEFT, RIGHT).
4. Abstention / Condition Logits: [B, 2] (Argument required vs not required).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedSpanPointerHead(nn.Module):
    """
    Learned dual-pointer head that points to exact argument words in the user goal.
    Conditioned on active target candidate and goal sequence.
    """

    def __init__(self, hidden_dim: int = 384, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Target Candidate Conditioning MLP
        self.cand_cond = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Start Token Pointer Network
        self.start_pointer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # End Token Pointer Network (Conditioned on start projection)
        self.end_pointer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Direction Classifier for Scroll actions
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 4),  # DOWN, UP, LEFT, RIGHT
        )

        # Abstention Head (Is argument text required for this action?)
        self.abstention_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 2),  # 0 = No argument, 1 = Argument present
        )

    def forward(
        self,
        goal_tokens: torch.Tensor,                  # [B, L, hidden_dim]
        target_candidate_embedding: torch.Tensor,   # [B, hidden_dim]
        goal_mask: Optional[torch.Tensor] = None,   # [B, L]
    ) -> Dict[str, torch.Tensor]:
        B, L, H = goal_tokens.shape

        # Condition candidate embedding
        c_cond = self.cand_cond(target_candidate_embedding)  # [B, H]
        c_exp = c_cond.unsqueeze(1).expand(B, L, H)          # [B, L, H]

        # 1. Start Pointer Logits
        joint_start = torch.cat([goal_tokens, c_exp], dim=-1) # [B, L, H * 2]
        start_logits = self.start_pointer(joint_start).squeeze(-1) # [B, L]

        # 2. End Pointer Logits
        end_logits = self.end_pointer(joint_start).squeeze(-1)   # [B, L]

        if goal_mask is not None:
            start_logits = start_logits.masked_fill(goal_mask == 0, -1e4)
            end_logits = end_logits.masked_fill(goal_mask == 0, -1e4)

        # 3. Scroll Direction & Abstention
        direction_logits = self.direction_head(c_cond)      # [B, 4]
        abstention_logits = self.abstention_head(c_cond)    # [B, 2]

        return {
            "start_logits": start_logits,
            "end_logits": end_logits,
            "direction_logits": direction_logits,
            "abstention_logits": abstention_logits,
        }
