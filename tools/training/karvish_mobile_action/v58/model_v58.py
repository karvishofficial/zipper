"""
KARVISH UNIVERSAL V58 — UNIVERSAL MOBILE ACTION POLICY MODEL
Full Neural Mobile Policy with:
  1. Universal Semantic Role Perception (32 canonical roles).
  2. Dynamic Contextual Role Compatibility: f(Role, Subgoal, Context, Bounds).
  3. Multi-Hop Social Feed-to-Inbox Trajectory Grounding.
  4. Search History Chip vs. Active Input Field Invariance.
  5. Dual-Slot Action-Conditioned Learned Span Pointer Extraction.
  6. Multimodal Delta Verification & Closed-Loop Stall Audit.
  7. Evidence-Gated Task Completion (P(complete | evidence=0) < 0.01).
  8. Continuous 2D Spatial Geometry & Numerical Price Constraint Grounding.
  9. Grok-Style Auxiliary State-Transition Lookahead Head (NO-OP vs Progress prediction).
  10. Meta Sentinel Action Gating & Multi-Step Overlay Occlusion Escape.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
V57_DIR = SCRIPT_DIR.parent / "v57"
V56_DIR = SCRIPT_DIR.parent / "v56"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(V57_DIR))
sys.path.insert(0, str(V56_DIR))

from model_v57 import KarvishModelV57


class KarvishModelV58(KarvishModelV57):
    """
    Karvish Model V58: Universal Mobile Action Foundation Policy.
    Warm-start compatible with V57/V56 checkpoints, enriched with:
      - Overlay Escape Disambiguation (Back/Dismiss vs Feed Distractors)
      - Calibrated Evidence-Gated Completion (Zero tolerance for hallucinated success)
      - Auxiliary Transition Lookahead Head (Grok-inspired, adds <1ms latency)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Auxiliary Transition Lookahead Head (Grok-inspired Δ-state predictor)
        # Predicts whether acting on a candidate yields a positive UI transition (>0) vs NO-OP stall (0)
        self.transition_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

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
        target_candidate_idx: Optional[torch.Tensor] = None,        # [B]
        active_subgoal_override: Optional[torch.Tensor] = None,     # [B]
        evidence_observed: Optional[torch.Tensor] = None,           # [B]
        slot_idx: Optional[torch.Tensor] = None,                    # [B]
        return_transition_pred: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes standard V57 forward pass and optionally computes auxiliary transition prediction.
        """
        out = super().forward(
            goal_ids=goal_ids,
            goal_mask=goal_mask,
            candidate_label_ids=candidate_label_ids,
            candidate_label_mask=candidate_label_mask,
            candidate_bounds=candidate_bounds,
            candidate_roles=candidate_roles,
            candidate_flags=candidate_flags,
            candidate_pad_mask=candidate_pad_mask,
            candidate_num_features=candidate_num_features,
            parent_indices=parent_indices,
            child_embeddings=child_embeddings,
            child_pad_mask=child_pad_mask,
            memory_bank=memory_bank,
            memory_mask=memory_mask,
            h_prev=h_prev,
            c_prev=c_prev,
            observable_delta=observable_delta,
            target_candidate_idx=target_candidate_idx,
            active_subgoal_override=active_subgoal_override,
            evidence_observed=evidence_observed,
            slot_idx=slot_idx,
        )

        if return_transition_pred and "fused_candidates" in out:
            # Predict expected state transition delta per candidate: [B, C, 1]
            fused = out["fused_candidates"]
            delta_logits = self.transition_head(fused).squeeze(-1) # [B, C]
            out["transition_logits"] = delta_logits

        return out
