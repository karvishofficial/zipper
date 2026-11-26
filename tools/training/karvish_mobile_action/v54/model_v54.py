"""
KARVISH MODEL V54 — MASTER NEURAL ARCHITECTURE
Compact Universal Mobile Action Intelligence (~27.5M Parameters).

Key Architectural Pillars:
1. Pre-trained MiniLM-L6-v2 Language Backbone (384-dim text representations).
2. Continuous 2D Spatial Geometry & Screen Zone LayoutLM-style Engine.
3. Hierarchical Container Context Binding (Parent Card Title & Price propagation into Child Buttons).
4. Multiplicative Semantic Entity Gating Head (Eliminates Ginger/Soap/Distractor False Positives).
5. Continuous Scalar Numerical Grounding Head (Monotonic Magnitude Ordering & Role Disambiguation).
6. Dynamic Goal-Coverage Attention & Recurrent Task Milestone State Engine.
7. Phase D Learned Span Pointer Head for Autonomous Argument Extraction.
8. Episodic Candidate Memory Bank for Multi-Turn Scroll Invariance.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from schema_v54 import CANONICAL_ACTIONS, ACTION_TO_IDX
from language_backbone_v54 import MiniLMTextEncoder
from spatial_geometry_v54 import SpatialGeometryEncoder
from hierarchical_container_attention_v54 import HierarchicalContainerAttention
from entity_gating_v54 import EntityGatingHead
from numerical_grounding_v54 import NumericalConstraintHead
from goal_coverage_and_state_v54 import DynamicGoalCoverageModule, RecurrentTaskStateEngine
from learned_span_head_v54 import LearnedSpanPointerHead
from candidate_memory_bank_v54 import EpisodicCandidateMemoryBank


class KarvishModelV54(nn.Module):
    """
    Karvish Model V54: Full neural mobile action policy.
    Zero app-specific heuristics; 100% learned neural representations.
    """

    def __init__(
        self,
        hidden_dim: int = 384,
        task_dim: int = 128,
        num_actions: int = 8,
        num_roles: int = 32,
        num_flags: int = 16,
        num_features: int = 4,
        max_seq_len: int = 32,
        max_candidates: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.task_dim = task_dim
        self.num_actions = num_actions

        # 1. Text Encoder (MiniLM-L6 backbone)
        self.text_encoder = MiniLMTextEncoder(hidden_dim=hidden_dim, dropout=dropout)

        # 2. Continuous 2D Spatial Geometry Encoder
        self.spatial_encoder = SpatialGeometryEncoder(input_dim=10, spatial_dim=64, hidden_dim=hidden_dim)

        # 3. Roles and UI State Flags (Pretrained V52 compatible)
        self.role_embed = nn.Embedding(num_roles, 64)
        self.flags_mlp = nn.Sequential(
            nn.Linear(num_flags, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )

        # 4. Candidate Base Fusion (Pretrained V52 compatible)
        self.cand_fuse = nn.Sequential(
            nn.Linear(hidden_dim + 64 + 64, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # 5. Hierarchical Container Attention & Card Context Binding
        self.container_attention = HierarchicalContainerAttention(hidden_dim=hidden_dim, num_heads=4, dropout=dropout)

        # 6. Episodic Candidate Memory Bank
        self.memory_bank = EpisodicCandidateMemoryBank(hidden_dim=hidden_dim, memory_size=16, num_heads=4, dropout=dropout)

        # 7. Dynamic Goal Coverage Attention
        self.goal_coverage = DynamicGoalCoverageModule(hidden_dim=hidden_dim, task_dim=task_dim)

        # 8. Recurrent Task State & Milestone Engine
        self.task_state_engine = RecurrentTaskStateEngine(hidden_dim=hidden_dim, task_dim=task_dim, delta_dim=10)

        # 9. Multiplicative Semantic Entity Gating Head
        self.entity_gater = EntityGatingHead(hidden_dim=hidden_dim)

        # 10. Continuous Scalar Numerical Constraint Head
        self.numerical_head = NumericalConstraintHead(hidden_dim=hidden_dim, num_features=num_features, dropout=dropout)

        # 11. 4-Way Interaction Candidate Ranking Head (Pretrained V52 compatible)
        self.rank_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # 12. Action Type Head (Pretrained V52 compatible)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_actions),
        )

        # 13. Learned Span / Copy-Pointer Head
        self.span_pointer_head = LearnedSpanPointerHead(hidden_dim=hidden_dim, dropout=dropout)

    def forward(
        self,
        goal_ids: torch.Tensor,                                     # [B, L]
        goal_mask: torch.Tensor,                                    # [B, L]
        candidate_label_ids: torch.Tensor,                          # [B, N, max_lbl_len]
        candidate_label_mask: torch.Tensor,                         # [B, N, max_lbl_len]
        candidate_bounds: torch.Tensor,                             # [B, N, 10]
        candidate_roles: torch.Tensor,                              # [B, N]
        candidate_flags: torch.Tensor,                              # [B, N, 16]
        candidate_pad_mask: Optional[torch.Tensor] = None,          # [B, N]
        candidate_num_features: Optional[torch.Tensor] = None,      # [B, N, 4]
        parent_indices: Optional[torch.Tensor] = None,              # [B, N]
        child_embeddings: Optional[torch.Tensor] = None,            # [B, N, max_c, hidden_dim]
        child_pad_mask: Optional[torch.Tensor] = None,              # [B, N, max_c]
        memory_bank: Optional[torch.Tensor] = None,                 # [B, K, hidden_dim]
        memory_mask: Optional[torch.Tensor] = None,                 # [B, K]
        h_prev: Optional[torch.Tensor] = None,                      # [B, task_dim]
        c_prev: Optional[torch.Tensor] = None,                      # [B, L]
        observable_delta: Optional[torch.Tensor] = None,            # [B, 10]
        target_candidate_idx: Optional[torch.Tensor] = None,        # [B] optional teacher forcing
    ) -> Dict[str, Any]:
        B, L = goal_ids.shape
        _, N, max_lbl_len = candidate_label_ids.shape

        dtype = next(self.parameters()).dtype
        if candidate_pad_mask is None:
            candidate_pad_mask = torch.ones(B, N, device=goal_ids.device, dtype=dtype)
        if observable_delta is None:
            observable_delta = torch.zeros(B, 10, device=goal_ids.device, dtype=dtype)

        # Step 1: Encode Goal
        goal_out = self.text_encoder(goal_ids, attention_mask=goal_mask)
        goal_tokens = goal_out["last_hidden_state"]                 # [B, L, hidden_dim]
        goal_pooled = goal_out["pooled_output"]                     # [B, hidden_dim]

        # Step 2: Initialize or Retrieve Task State
        if h_prev is None:
            h_prev = self.task_state_engine.init_state(goal_pooled) # [B, task_dim]

        # Step 3: Dynamic Goal Coverage Attention
        cov_out = self.goal_coverage(
            goal_tokens=goal_tokens,
            task_state=h_prev,
            c_prev=c_prev,
            goal_mask=goal_mask,
        )
        u_active = cov_out["u_active"]                              # [B, hidden_dim]
        c_t = cov_out["c_t"]                                        # [B, L]
        gamma_t = cov_out["gamma_t"]                                # [B]

        # Step 4: Encode Candidate Text
        flat_lbl_ids = candidate_label_ids.view(B * N, max_lbl_len)
        flat_lbl_mask = candidate_label_mask.view(B * N, max_lbl_len)
        lbl_out = self.text_encoder(flat_lbl_ids, attention_mask=flat_lbl_mask)
        e_text = lbl_out["pooled_output"].view(B, N, self.hidden_dim)  # [B, N, hidden_dim]

        # Step 5: Encode Continuous 2D Spatial Geometry & Screen Zones
        e_spatial = self.spatial_encoder(candidate_bounds)          # [B, N, hidden_dim]

        # Step 6: Encode Roles & Flags
        e_role = self.role_embed(candidate_roles)                   # [B, N, 64]
        e_flags = self.flags_mlp(candidate_flags)                   # [B, N, 64]

        # Fuse candidate base representations
        v_base = self.cand_fuse(torch.cat([e_text + e_spatial, e_role, e_flags], dim=-1))  # [B, N, hidden_dim]

        # Step 7: Hierarchical Container Cross-Attention & Parent Card Binding
        v_candidates = self.container_attention(
            candidate_embeddings=v_base,
            goal_embeddings=goal_tokens,
            candidate_pad_mask=candidate_pad_mask,
            parent_indices=parent_indices,
            child_embeddings=child_embeddings,
            child_pad_mask=child_pad_mask,
        )  # [B, N, hidden_dim]

        # Step 8: Episodic Candidate Memory Bank Cross-Attention
        if memory_bank is not None:
            v_candidates, memory_bank = self.memory_bank(
                candidate_embeddings=v_candidates,
                memory_bank=memory_bank,
                memory_mask=memory_mask,
            )

        # Step 9: Multiplicative Semantic Entity Gating
        entity_out = self.entity_gater(
            goal_intent=u_active,
            candidate_embeddings=v_candidates,
            candidate_pad_mask=candidate_pad_mask,
        )
        entity_gate = entity_out["entity_gate"]                     # [B, N] in (0, 1)

        # Step 10: Candidate Ranking (Factored 4-Way Interaction)
        u_exp = u_active.unsqueeze(1).expand(B, N, self.hidden_dim)
        diff = torch.abs(u_exp - v_candidates)
        prod = u_exp * v_candidates
        rank_features = torch.cat([u_exp, v_candidates, diff, prod], dim=-1)  # [B, N, hidden_dim * 4]
        ranking_scores = self.rank_head(rank_features).squeeze(-1)            # [B, N]

        # Step 11: Entity-Gated Numerical & Comparative Constraint Adjustment
        if candidate_num_features is not None:
            num_out = self.numerical_head(
                candidate_num_features=candidate_num_features,
                goal_intent=u_active,
                entity_gate=entity_gate,
            )
            ranking_scores = ranking_scores + num_out["comp_bias"]

        if candidate_pad_mask is not None:
            ranking_scores = ranking_scores.masked_fill(candidate_pad_mask == 0, -1e4)

        # Step 12: Action Type Classification (Conditioned on selected candidate)
        if target_candidate_idx is not None:
            sel_cand_idx = target_candidate_idx.unsqueeze(-1).unsqueeze(-1).expand(B, 1, self.hidden_dim)
        else:
            sel_cand_idx = ranking_scores.argmax(dim=-1).unsqueeze(-1).unsqueeze(-1).expand(B, 1, self.hidden_dim)
        sel_cand_v = v_candidates.gather(1, sel_cand_idx).squeeze(1)         # [B, hidden_dim]
        action_features = torch.cat([u_active, sel_cand_v], dim=-1)          # [B, hidden_dim * 2]
        action_logits = self.action_head(action_features)                     # [B, num_actions]

        # Step 13: Phase D Learned Span / Pointer Argument Extraction
        span_out = self.span_pointer_head(
            goal_tokens=goal_tokens,
            target_candidate_embedding=sel_cand_v,
            goal_mask=goal_mask,
        )

        # Step 14: Update Task State from Screen Observation
        screen_summary = v_candidates.mean(dim=1)                             # [B, hidden_dim]
        state_out = self.task_state_engine(h_prev, screen_summary, observable_delta)

        return {
            "ranking_scores": ranking_scores,               # [B, N]
            "action_logits": action_logits,                 # [B, 8]
            "entity_gate": entity_gate,                     # [B, N]
            "entity_sim": entity_out["entity_sim"],         # [B, N]
            "span_start_logits": span_out["start_logits"],  # [B, L]
            "span_end_logits": span_out["end_logits"],      # [B, L]
            "direction_logits": span_out["direction_logits"], # [B, 4]
            "h_t": state_out["h_t"],                        # [B, task_dim]
            "c_t": c_t,                                     # [B, L]
            "gamma_t": gamma_t,                             # [B]
            "subgoal_logits": state_out["subgoal_logits"],  # [B, 10]
            "progress_scalar": state_out["progress_scalar"],# [B]
            "termination_prob": state_out["termination_prob"], # [B]
            "milestone_prob": state_out["milestone_prob"],  # [B]
            "candidate_embeddings": v_candidates,           # [B, N, hidden_dim]
            "goal_tokens": goal_tokens,                     # [B, L, hidden_dim]
            "goal_pooled": goal_pooled,                     # [B, hidden_dim]
        }
