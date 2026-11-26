"""
KARVISH V54 — DYNAMIC GOAL-COVERAGE ATTENTION & RECURRENT TASK STATE
Autonomous multi-step sequence tracking & milestone completion detection.

Mathematical Formulations:
1. Advance Gate: gamma_t = sigmoid(W_adv h_t^task) in [0, 1]
2. Coverage Update: c_t = clamp(c_{t-1} + gamma_t * alpha_t, 0.0, 1.0)
3. Dynamic Active Goal Context: u_active = sum(alpha_t * h_goal) + W_fuse h_t^task
4. Task State GRU: h_t^task = GRUCell([Delta_t || v_screen], h_{t-1}^task)
5. Milestone Termination: termination_prob = sigmoid(W_term h_t^task) in [0, 1]
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

UNIVERSAL_SUBGOALS = [
    "LOCATE_INPUT",
    "ENTER_QUERY",
    "SUBMIT_SEARCH",
    "SELECT_RESULT",
    "NAVIGATE_TAB",
    "CONFIRM_ACTION",
    "DISMISS_MODAL",
    "SCROLL_FEED",
    "RECOVER_ERROR",
    "COMPLETE_TASK",
]


class DynamicGoalCoverageModule(nn.Module):
    """
    Tracks multi-turn task completion across goal tokens.
    Maintains coverage accumulator c_t and produces dynamic active context vector.
    """

    def __init__(self, hidden_dim: int = 384, task_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.task_dim = task_dim

        # Attention projections for goal coverage
        self.query_proj = nn.Linear(task_dim, task_dim)
        self.key_proj = nn.Linear(hidden_dim, task_dim)

        # Context fusion projection
        self.fuse_proj = nn.Linear(task_dim, hidden_dim)

        # Advance gating scalar head
        self.advance_gate_proj = nn.Linear(task_dim, 1)

    def forward(
        self,
        goal_tokens: torch.Tensor,                  # [B, L, hidden_dim]
        task_state: torch.Tensor,                   # [B, task_dim]
        c_prev: Optional[torch.Tensor] = None,      # [B, L]
        goal_mask: Optional[torch.Tensor] = None,   # [B, L]
    ) -> Dict[str, torch.Tensor]:
        B, L, _ = goal_tokens.shape

        if c_prev is None:
            c_prev = torch.zeros(B, L, device=goal_tokens.device, dtype=goal_tokens.dtype)

        # 1. Query = task state, Key = goal tokens
        q = self.query_proj(task_state).unsqueeze(1)            # [B, 1, task_dim]
        k = self.key_proj(goal_tokens)                          # [B, L, task_dim]

        # 2. Raw attention scores modulated by prior unconsumed coverage (1 - c_prev)
        uncovered_weight = (1.0 - c_prev).clamp(min=0.01)
        raw_scores = torch.bmm(q, k.transpose(1, 2)).squeeze(1) / math.sqrt(self.task_dim)  # [B, L]

        if goal_mask is not None:
            raw_scores = raw_scores.masked_fill(goal_mask == 0, -1e4)

        # Weight by uncovered ratio
        weighted_scores = raw_scores + torch.log(uncovered_weight)
        alpha_t = F.softmax(weighted_scores, dim=-1).to(goal_tokens.dtype)            # [B, L]

        # 3. Advance Gate: How much new goal content to advance this step
        gamma_t = torch.sigmoid(self.advance_gate_proj(task_state)).squeeze(-1)  # [B]

        # 4. Update Coverage: c_t = clamp(c_{t-1} + gamma_t * alpha_t, 0.0, 1.0)
        c_t = torch.clamp(c_prev + gamma_t.unsqueeze(1) * alpha_t, 0.0, 1.0)

        # 5. Dynamic Active Context Vector: Attend to current focus tokens + fuse task state
        attended_goal = torch.bmm(alpha_t.unsqueeze(1), goal_tokens).squeeze(1)  # [B, hidden_dim]
        u_active = attended_goal + self.fuse_proj(task_state)                     # [B, hidden_dim]

        return {
            "u_active": u_active,       # [B, hidden_dim] Dynamic fused context
            "alpha_t": alpha_t,         # [B, L] Current attention distribution
            "c_t": c_t,                 # [B, L] Accumulated coverage
            "gamma_t": gamma_t,         # [B] Advance probability
        }


class RecurrentTaskStateEngine(nn.Module):
    """
    Maintains causal recurrent memory of environmental task progress.
    Updates h_t from observable UI delta and screen affordance summary.
    """

    def __init__(self, hidden_dim: int = 384, task_dim: int = 128, delta_dim: int = 10):
        super().__init__()
        self.task_dim = task_dim

        # Observable UI Delta Projection
        self.delta_proj = nn.Sequential(
            nn.Linear(delta_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, task_dim),
        )

        # Screen Affordance Summary Projection
        self.screen_proj = nn.Sequential(
            nn.Linear(hidden_dim, task_dim),
            nn.LayerNorm(task_dim),
        )

        # Initial Task State from Goal
        self.init_proj = nn.Sequential(
            nn.Linear(hidden_dim, task_dim),
            nn.Tanh(),
        )

        # Recurrent GRU Cell
        self.task_gru = nn.GRUCell(input_size=task_dim * 2, hidden_size=task_dim)

        # Autonomous Progression Heads
        self.subgoal_head = nn.Linear(task_dim, len(UNIVERSAL_SUBGOALS))
        self.progress_head = nn.Sequential(nn.Linear(task_dim, 1), nn.Sigmoid())
        self.termination_head = nn.Sequential(nn.Linear(task_dim, 1), nn.Sigmoid())
        self.milestone_head = nn.Sequential(nn.Linear(task_dim, 1), nn.Sigmoid())

    def init_state(self, global_goal: torch.Tensor) -> torch.Tensor:
        return self.init_proj(global_goal)

    def forward(
        self,
        h_prev: torch.Tensor,                       # [B, task_dim]
        screen_summary: torch.Tensor,               # [B, hidden_dim]
        observable_delta: torch.Tensor,             # [B, delta_dim]
    ) -> Dict[str, torch.Tensor]:
        delta_feat = self.delta_proj(observable_delta)
        screen_feat = self.screen_proj(screen_summary)

        gru_input = torch.cat([delta_feat, screen_feat], dim=-1)  # [B, task_dim * 2]
        h_t = self.task_gru(gru_input, h_prev)                    # [B, task_dim]

        subgoal_logits = self.subgoal_head(h_t)
        progress_scalar = self.progress_head(h_t).squeeze(-1)
        termination_prob = self.termination_head(h_t).squeeze(-1)
        milestone_prob = self.milestone_head(h_t).squeeze(-1)

        return {
            "h_t": h_t,
            "subgoal_logits": subgoal_logits,
            "progress_scalar": progress_scalar,
            "termination_prob": termination_prob,
            "milestone_prob": milestone_prob,
        }
