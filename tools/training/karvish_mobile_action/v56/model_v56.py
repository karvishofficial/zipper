"""
KARVISH UNIVERSAL V56 — UNIVERSAL MOBILE ACTION POLICY MODEL
Full Neural Mobile Policy with:
  1. Universal Semantic Role Perception (32 canonical roles).
  2. Dynamic Contextual Role Compatibility: f(Role, Subgoal, Context, Bounds).
  3. Dual-Slot Action-Conditioned Learned Span Pointer Extraction (Recipient vs. Message Payload).
  4. Multimodal Delta Verification & Closed-Loop Stall Audit.
  5. Evidence-Gated Task Completion (P(complete | evidence=0) < 0.15).
  6. Continuous 2D Spatial Geometry & Numerical Price Constraint Grounding.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path("c:/Users/Karthik S/AndroidStudioProjects/zipper")
V54_DIR = REPO_ROOT / "tools" / "training" / "karvish_mobile_action" / "v54"
V55_DIR = REPO_ROOT / "tools" / "training" / "karvish_mobile_action" / "v55"
SCRIPT_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(V55_DIR))
sys.path.insert(0, str(V54_DIR))

# Preserved V54 modules
from language_backbone_v54 import MiniLMTextEncoder
from spatial_geometry_v54 import SpatialGeometryEncoder
from hierarchical_container_attention_v54 import HierarchicalContainerAttention
from entity_gating_v54 import EntityGatingHead
from numerical_grounding_v54 import NumericalConstraintHead
from goal_coverage_and_state_v54 import DynamicGoalCoverageModule, RecurrentTaskStateEngine
from candidate_memory_bank_v54 import EpisodicCandidateMemoryBank
from learned_span_head_v54 import LearnedSpanPointerHead

# V55 & V56 modules
from role_compatibility_v55 import ContextualRoleCompatibilityModule
from evidence_verifier_v55 import MultimodalDeltaVerifier, EvidenceGatedCompletionHead
from action_conditioned_span_v56 import ActionConditionedSpanHeadV56
from schema_v56 import CANONICAL_ACTIONS, NUM_UNIVERSAL_ROLES, NUM_UNIVERSAL_SUBGOALS


class KarvishModelV56(nn.Module):
    """
    Karvish Model V56: Universal Mobile Action Foundation Policy.
    Zero app-specific heuristics; 100% learned universal neural representations.
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
        self.num_roles = num_roles

        # 1. Text Encoder (MiniLM-L6-v2)
        self.text_encoder = MiniLMTextEncoder()

        # 2. Continuous 2D Spatial Geometry Encoder
        self.spatial_encoder = SpatialGeometryEncoder(hidden_dim=hidden_dim)

        # 3. Universal Semantic UI Role Embedding
        self.role_embed = nn.Embedding(num_roles, 64)

        # 4. Accessibility State Flags MLP
        self.flags_mlp = nn.Sequential(
            nn.Linear(num_flags, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )

        # 5. Candidate Fusion Layer
        self.cand_fuse = nn.Sequential(
            nn.Linear(hidden_dim + 64 + 64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 6. Recurrent Task State & Subgoal Engine
        self.task_state_engine = RecurrentTaskStateEngine(hidden_dim=hidden_dim, task_dim=task_dim)

        # 7. Dynamic Goal Coverage Attention
        self.goal_coverage = DynamicGoalCoverageModule(hidden_dim=hidden_dim, task_dim=task_dim)

        # 8. Hierarchical Container Attention
        self.container_attention = HierarchicalContainerAttention(hidden_dim=hidden_dim, dropout=dropout)

        # 9. Episodic Candidate Memory Bank
        self.memory_bank = EpisodicCandidateMemoryBank(hidden_dim=hidden_dim)

        # 10. Multiplicative Semantic Entity Gating Head
        self.entity_gater = EntityGatingHead(hidden_dim=hidden_dim)

        # 11. Continuous Scalar Numerical Constraint Head
        self.numerical_head = NumericalConstraintHead(hidden_dim=hidden_dim, num_features=num_features)

        # 12. 4-Way Interaction Formulation Rank Head
        self.rank_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # 13. Action Type Head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_actions),
        )

        # 14. Contextual Role Compatibility Engine
        self.role_compatibility = ContextualRoleCompatibilityModule(
            hidden_dim=hidden_dim,
            role_dim=64,
            subgoal_dim=64,
            spatial_dim=64,
            dropout=dropout,
        )

        # 15. Dual-Slot Action-Conditioned Learned Span Head (V56)
        self.action_conditioned_span = ActionConditionedSpanHeadV56(
            hidden_dim=hidden_dim,
            action_dim=64,
            role_dim=64,
            slot_dim=32,
            num_actions=num_actions,
            num_roles=num_roles,
            dropout=dropout,
        )

        # 16. Multimodal Delta Verifier (Closed-loop transition audit)
        self.delta_verifier = MultimodalDeltaVerifier(delta_dim=10, hidden_dim=hidden_dim, task_dim=task_dim)

        # 17. Evidence-Gated Completion Head
        self.evidence_completion = EvidenceGatedCompletionHead(task_dim=task_dim, hidden_dim=hidden_dim)

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
        active_subgoal_override: Optional[torch.Tensor] = None,     # [B] optional subgoal label
        evidence_observed: Optional[torch.Tensor] = None,           # [B] 0 or 1
        slot_idx: Optional[torch.Tensor] = None,                    # [B] 0=recipient, 1=payload
    ) -> Dict[str, Any]:
        B, L = goal_ids.shape
        _, N, max_lbl_len = candidate_label_ids.shape

        dtype = next(self.parameters()).dtype
        if candidate_pad_mask is None:
            candidate_pad_mask = torch.ones(B, N, device=goal_ids.device, dtype=dtype)
        if observable_delta is None:
            observable_delta = torch.zeros(B, 10, device=goal_ids.device, dtype=dtype)
        if candidate_num_features is None:
            candidate_num_features = torch.zeros(B, N, 4, device=goal_ids.device, dtype=dtype)

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
        roles_clamped = candidate_roles.clamp(0, self.num_roles - 1)
        e_role = self.role_embed(roles_clamped)                     # [B, N, 64]
        e_flags = self.flags_mlp(candidate_flags)                   # [B, N, 64]

        # Fuse candidate base representations
        cand_base_in = torch.cat([e_text + e_spatial, e_role, e_flags], dim=-1) # [B, N, hidden_dim + 128]
        v_base = self.cand_fuse(cand_base_in)                                     # [B, N, hidden_dim]

        # Step 7: Hierarchical Container Attention
        v_candidates = self.container_attention(
            candidate_embeddings=v_base,
            goal_embeddings=goal_tokens,
            candidate_pad_mask=candidate_pad_mask,
            parent_indices=parent_indices,
            child_embeddings=child_embeddings,
            child_pad_mask=child_pad_mask,
        )

        # Step 8: Episodic Candidate Memory Bank
        if memory_bank is not None:
            v_candidates, memory_bank = self.memory_bank(
                candidate_embeddings=v_candidates,
                memory_bank=memory_bank,
                memory_mask=memory_mask,
            )

        # Step 9: 4-Way Interaction Formulation (Exact V54 Feature Order)
        u_exp = u_active.unsqueeze(1).expand(B, N, self.hidden_dim)
        diff = torch.abs(u_exp - v_candidates)
        prod = u_exp * v_candidates
        four_way = torch.cat([u_exp, v_candidates, diff, prod], dim=-1)  # [B, N, hidden_dim * 4]
        raw_ranking_scores = self.rank_head(four_way).squeeze(-1)        # [B, N]

        # Step 10: Multiplicative Semantic Entity Gating Head
        entity_out = self.entity_gater(
            goal_intent=u_active,
            candidate_embeddings=v_candidates,
            candidate_pad_mask=candidate_pad_mask,
        )
        entity_gate = entity_out["entity_gate"]                     # [B, N] in (0, 1)

        # Step 11: Continuous Scalar Numerical Constraint Head
        if candidate_num_features is not None:
            num_out = self.numerical_head(
                candidate_num_features=candidate_num_features,
                goal_intent=u_active,
                entity_gate=entity_gate,
            )
            comp_bias = num_out["comp_bias"]                            # [B, N]
        else:
            comp_bias = 0.0

        # Step 12: Multimodal Delta Verifier (Detect NO-OP / Stall)
        delta_audit = self.delta_verifier(observable_delta)
        delta_features = delta_audit["delta_features"]              # [B, task_dim]
        stall_prob = delta_audit["stall_prob"]                      # [B]

        # Step 13: Recurrent Task State & Subgoal Prediction
        screen_summary = v_candidates.mean(dim=1)                   # [B, hidden_dim]
        state_out = self.task_state_engine(h_prev, screen_summary, observable_delta)
        h_t = state_out["h_t"]                                      # [B, task_dim]
        subgoal_logits = state_out["subgoal_logits"]                # [B, 12]

        if active_subgoal_override is not None:
            active_subgoal = active_subgoal_override
        else:
            active_subgoal = torch.argmax(subgoal_logits, dim=-1)   # [B]

        # Step 14: Contextual Role Compatibility Engine: f(Role, Subgoal, Context, Bounds)
        role_compat_out = self.role_compatibility(
            candidate_roles=candidate_roles,
            candidate_bounds=candidate_bounds,
            active_subgoal_idx=active_subgoal,
            u_active=u_active,
        )
        role_compat_bias = role_compat_out["compatibility_bias"]    # [B, N]

        # Combine all components into final ranking scores
        ranking_scores = raw_ranking_scores + comp_bias + role_compat_bias
        ranking_scores = ranking_scores.masked_fill(candidate_pad_mask == 0, -1e4)

        # Step 15: Select Active Candidate & Predict Action Type
        if target_candidate_idx is not None:
            sel_cand_idx = target_candidate_idx.unsqueeze(-1).unsqueeze(-1).expand(B, 1, self.hidden_dim)
            sel_idx_1d = target_candidate_idx
        else:
            sel_idx_1d = ranking_scores.argmax(dim=-1)              # [B]
            sel_cand_idx = sel_idx_1d.unsqueeze(-1).unsqueeze(-1).expand(B, 1, self.hidden_dim)

        sel_cand_v = v_candidates.gather(1, sel_cand_idx).squeeze(1)         # [B, hidden_dim]
        action_features = torch.cat([u_active, sel_cand_v], dim=-1)          # [B, hidden_dim * 2]
        action_logits = self.action_head(action_features)                     # [B, num_actions]
        pred_action_idx = torch.argmax(action_logits, dim=-1)                 # [B]

        # Step 16: Dual-Slot Action-Conditioned Span Extraction
        sel_role_idx = candidate_roles.gather(1, sel_idx_1d.unsqueeze(-1)).squeeze(-1) # [B]
        span_out = self.action_conditioned_span(
            goal_tokens=goal_tokens,
            target_candidate_embedding=sel_cand_v,
            target_action_idx=pred_action_idx,
            target_role_idx=sel_role_idx,
            goal_mask=goal_mask,
            slot_idx=slot_idx,
        )

        # Step 17: Evidence-Gated Task Completion
        evid_out = self.evidence_completion(
            task_state=h_t,
            evidence_observed=evidence_observed,
        )
        completion_prob = evid_out["completion_prob"]               # [B]

        return {
            "ranking_scores": ranking_scores,               # [B, N]
            "raw_ranking_scores": raw_ranking_scores,       # [B, N]
            "action_logits": action_logits,                 # [B, num_actions]
            "span_start_logits": span_out["span_start_logits"], # [B, L]
            "span_end_logits": span_out["span_end_logits"],     # [B, L]
            "stall_prob": stall_prob,                       # [B]
            "termination_prob": completion_prob,            # [B]
            "h_t": h_t,                                     # [B, task_dim]
            "c_t": c_t,                                     # [B, L]
            "subgoal_logits": subgoal_logits,               # [B, 12]
        }
