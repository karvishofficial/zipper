"""
KARVISH V52 — PHASE 6: LEARNED SPAN / COPY-POINTER HEAD
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

        # Abstention Head: Predicts if argument extraction is required
        self.abstention_head = nn.Linear(hidden_dim, 2)

    def forward(
        self,
        goal_tokens: torch.Tensor,                  # [B, L, hidden_dim]
        target_candidate_embedding: torch.Tensor,   # [B, hidden_dim]
        goal_mask: Optional[torch.Tensor] = None,   # [B, L] (1 for valid token, 0 for pad)
    ) -> Dict[str, torch.Tensor]:
        B, L, H = goal_tokens.shape

        # Condition candidate embedding: [B, 1, H] -> expand to [B, L, H]
        c_cond = self.cand_cond(target_candidate_embedding).unsqueeze(1).expand(B, L, H)
        fused_features = torch.cat([goal_tokens, c_cond], dim=-1)  # [B, L, H * 2]

        start_logits = self.start_pointer(fused_features).squeeze(-1)  # [B, L]
        end_logits = self.end_pointer(fused_features).squeeze(-1)      # [B, L]

        # Mask padding tokens so they cannot be selected as span boundaries (FP16 safe)
        if goal_mask is not None:
            pad_mask = (goal_mask == 0)
            start_logits = start_logits.masked_fill(pad_mask, -1e4)
            end_logits = end_logits.masked_fill(pad_mask, -1e4)

        # Global goal representation (first token / mean)
        global_goal = goal_tokens.mean(dim=1)
        direction_logits = self.direction_head(global_goal)
        abstention_logits = self.abstention_head(global_goal)

        return {
            "start_logits": start_logits,             # [B, L]
            "end_logits": end_logits,                 # [B, L]
            "direction_logits": direction_logits,     # [B, 4]
            "abstention_logits": abstention_logits,   # [B, 2]
        }

    def decode_span(
        self,
        goal_text: str,
        tokenizer: Any,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
    ) -> str:
        """Decodes raw pointer logits into physical text argument without regex."""
        s_idx = int(start_logits.argmax().item())
        e_idx = int(end_logits.argmax().item())

        if e_idx < s_idx:
            e_idx = s_idx

        # Tokenize and decode span slice
        tokens = tokenizer.encode(goal_text)
        if s_idx < len(tokens) and e_idx < len(tokens):
            extracted = tokenizer.decode(tokens[s_idx : e_idx + 1])
            return extracted.strip()
        return ""
