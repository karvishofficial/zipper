"""
KARVISH MODEL V52 — THE TEACHER-DISTILLED UNIVERSAL ACTION POLICY
Authoritative PyTorch Architecture for Phase 1 to Phase 10.

Governing Laws:
- AGENTS.md Law 34: Learned assets < 50,000,000 bytes (< 30 MB in INT8).
- AGENTS.md Law 35: Warm latency < 300 ms (< 35 ms on ARM64).
- AGENTS.md Law 5: 100% Neural task intelligence (Zero -5.0, Zero +3.5, Zero regex).

Modules Integrated:
1. MiniLMTextEncoder (Phase 3 Language Backbone, 22.7M params)
2. SpatialGeometryEncoder (Phase 4 Continuous 2D Layout & Screen Zones)
3. HierarchicalContainerAttention (Phase 5 Composite Card Cross-Attention)
4. LearnedSpanPointerHead (Phase 6 Learned Goal Argument Extraction)
5. DynamicGoalCoverageModule & RecurrentTaskStateEngine (Phase 7 Multi-Step State)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from language_backbone_v52 import MiniLMTextEncoder
from spatial_geometry_v52 import SpatialGeometryEncoder
from hierarchical_container_attention_v52 import HierarchicalContainerAttention
from learned_span_head_v52 import LearnedSpanPointerHead
from goal_coverage_and_state_v52 import (
    DynamicGoalCoverageModule,
    RecurrentTaskStateEngine,
    UNIVERSAL_SUBGOALS,
)
from schema_v52 import CANONICAL_ACTIONS


class KarvishModelV52(nn.Module):
    """
    Karvish V52 Universal Action Intelligence Model.
    Compact 28.5M Parameter Multi-Modal Mobile Agent.
    """

    def __init__(
        self,
        hidden_dim: int = 384,
        task_dim: int = 128,
        num_canonical_actions: int = len(CANONICAL_ACTIONS),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.task_dim = task_dim
        self.num_actions = num_canonical_actions

        # 1. Phase 3 Language Backbone (MiniLM-L6, ~22.7M params)
        self.text_encoder = MiniLMTextEncoder(
            hidden_dim=hidden_dim,
            num_layers=6,
            num_heads=12,
            dim_feedforward=1536,
            dropout=dropout,
        )

        # 2. Phase 4 Continuous 2D Spatial Geometry Encoder
        self.spatial_encoder = SpatialGeometryEncoder(
            input_dim=10,
            spatial_dim=64,
            hidden_dim=hidden_dim,
        )

        # Role & Flag Embeddings
        self.role_embed = nn.Embedding(32, 64)
        self.flags_mlp = nn.Sequential(
            nn.Linear(16, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        self.cand_fuse = nn.Sequential(
            nn.Linear(hidden_dim + 64 + 64, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # 3. Phase 5 Hierarchical Container Cross-Attention & Screen Scene Graph
        self.container_attention = HierarchicalContainerAttention(
            hidden_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
        )

        # 4. Phase 7 Dynamic Goal Coverage & Recurrent Task State Engine
        self.goal_coverage = DynamicGoalCoverageModule(
            hidden_dim=hidden_dim,
            task_dim=task_dim,
        )
        self.task_state_engine = RecurrentTaskStateEngine(
            hidden_dim=hidden_dim,
            task_dim=task_dim,
            delta_dim=10,
        )

        # 5. Phase 6 Learned Span / Pointer Head for text argument extraction
        self.span_pointer_head = LearnedSpanPointerHead(
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # 6. Candidate Ranking Head (Factored Interaction Scoring)
        self.rank_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # 7. Action Type Classifier Head (8 Canonical Actions)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_actions),
        )

    def forward(
        self,
        goal_ids: torch.Tensor,                             # [B, L]
        goal_mask: torch.Tensor,                            # [B, L]
        candidate_label_ids: torch.Tensor,                  # [B, N, max_lbl_len]
        candidate_label_mask: torch.Tensor,                 # [B, N, max_lbl_len]
        candidate_bounds: torch.Tensor,                     # [B, N, 10]
        candidate_roles: torch.Tensor,                      # [B, N]
        candidate_flags: torch.Tensor,                      # [B, N, 16]
        candidate_pad_mask: Optional[torch.Tensor] = None,  # [B, N]
        h_prev: Optional[torch.Tensor] = None,              # [B, task_dim]
        c_prev: Optional[torch.Tensor] = None,              # [B, L]
        observable_delta: Optional[torch.Tensor] = None,    # [B, 10]
        child_embeddings: Optional[torch.Tensor] = None,    # [B, N, max_children, hidden_dim]
        child_pad_mask: Optional[torch.Tensor] = None,      # [B, N, max_children]
    ) -> Dict[str, torch.Tensor]:
        B, L = goal_ids.shape
        N = candidate_bounds.shape[1]
        device = goal_ids.device

        # Step 1: Encode Goal Tokens via Pre-trained Language Backbone
        goal_out = self.text_encoder(goal_ids, attention_mask=goal_mask)
        goal_tokens = goal_out["last_hidden_state"]         # [B, L, hidden_dim]
        goal_pooled = goal_out["pooled_output"]             # [B, hidden_dim]

        # Step 2: Initialize or update Recurrent Task State
        if h_prev is None:
            h_prev = self.task_state_engine.init_state(goal_pooled)
        if observable_delta is None:
            observable_delta = torch.zeros(B, 10, device=device)

        # Step 3: Compute Dynamic Goal Coverage & Active Goal Vector
        cov_out = self.goal_coverage(goal_tokens, h_prev, c_prev=c_prev, goal_mask=goal_mask)
        u_active = cov_out["u_active"]                      # [B, hidden_dim]
        c_t = cov_out["c_t"]                                # [B, L]
        gamma_t = cov_out["gamma_t"]                        # [B]

        # Step 4: Encode Candidate Labels
        max_lbl_len = candidate_label_ids.shape[-1]
        flat_lbl_ids = candidate_label_ids.view(B * N, max_lbl_len)
        flat_lbl_mask = candidate_label_mask.view(B * N, max_lbl_len)
        lbl_out = self.text_encoder(flat_lbl_ids, attention_mask=flat_lbl_mask)
        e_text = lbl_out["pooled_output"].view(B, N, self.hidden_dim)  # [B, N, hidden_dim]

        # Step 5: Encode Continuous 2D Spatial Geometry & Screen Zones
        e_spatial = self.spatial_encoder(candidate_bounds)  # [B, N, hidden_dim]

        # Step 6: Encode Roles & Flags
        e_role = self.role_embed(candidate_roles)           # [B, N, 64]
        e_flags = self.flags_mlp(candidate_flags)           # [B, N, 64]

        # Fuse candidate base representations
        v_base = self.cand_fuse(torch.cat([e_text + e_spatial, e_role, e_flags], dim=-1))  # [B, N, hidden_dim]

        # Step 7: Hierarchical Container Cross-Attention & Screen Scene Graph
        v_candidates = self.container_attention(
            candidate_embeddings=v_base,
            goal_embeddings=goal_tokens,
            candidate_pad_mask=candidate_pad_mask,
            child_embeddings=child_embeddings,
            child_pad_mask=child_pad_mask,
        )  # [B, N, hidden_dim]

        # Step 8: Candidate Ranking (Factored 4-Way Interaction)
        u_exp = u_active.unsqueeze(1).expand(B, N, self.hidden_dim)
        diff = torch.abs(u_exp - v_candidates)
        prod = u_exp * v_candidates
        rank_features = torch.cat([u_exp, v_candidates, diff, prod], dim=-1)  # [B, N, hidden_dim * 4]
        ranking_scores = self.rank_head(rank_features).squeeze(-1)            # [B, N]

        if candidate_pad_mask is not None:
            ranking_scores = ranking_scores.masked_fill(candidate_pad_mask == 0, -1e4)

        # Step 9: Action Type Classification (Conditioned on top ranked candidate)
        top_cand_idx = ranking_scores.argmax(dim=-1).unsqueeze(-1).unsqueeze(-1).expand(B, 1, self.hidden_dim)
        top_cand_v = v_candidates.gather(1, top_cand_idx).squeeze(1)         # [B, hidden_dim]
        action_features = torch.cat([u_active, top_cand_v], dim=-1)          # [B, hidden_dim * 2]
        action_logits = self.action_head(action_features)                     # [B, num_actions]

        # Step 10: Phase D Learned Span / Pointer Argument Extraction
        span_out = self.span_pointer_head(
            goal_tokens=goal_tokens,
            target_candidate_embedding=top_cand_v,
            goal_mask=goal_mask,
        )

        # Step 11: Update Task State from Screen Observation
        screen_summary = v_candidates.mean(dim=1)                             # [B, hidden_dim]
        state_out = self.task_state_engine(h_prev, screen_summary, observable_delta)

        return {
            "ranking_scores": ranking_scores,               # [B, N] Candidate logits
            "action_logits": action_logits,                 # [B, 8] Action logits
            "span_start_logits": span_out["start_logits"],  # [B, L]
            "span_end_logits": span_out["end_logits"],      # [B, L]
            "direction_logits": span_out["direction_logits"], # [B, 4]
            "h_t": state_out["h_t"],                        # [B, task_dim]
            "c_t": c_t,                                     # [B, L]
            "gamma_t": gamma_t,                             # [B]
            "subgoal_logits": state_out["subgoal_logits"],  # [B, 10]
            "progress_scalar": state_out["progress_scalar"],# [B]
            "termination_prob": state_out["termination_prob"], # [B]
        }
