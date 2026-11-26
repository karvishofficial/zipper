"""
KARVISH UNIVERSAL V60 — UNIVERSAL ZERO-SHOT MOBILE AGENT & CALIBRATED TERMINATION POLICY
Architecture Specification (AGENTS.md):
  1. Universal Semantic Role Perception (32 canonical roles).
  2. Asymmetric Goal-UI Cross-Attention & Spatial 2D Bounds Integration.
  3. Continuous Scalar Numerical Constraint & Entity Gating Heads.
  4. Dual-Slot Action-Conditioned Learned Span Pointer Extraction.
  5. 15-Class Token-level BIO NLU Slot Head for colloquial slang & multi-item baskets.
  6. Calibrated Goal Termination Verifier Head (Focal-supervised against near-miss false-stops).
  7. Stagnation & Obstacle Slip Recovery Head (Handles zero-delta screens & blocking modals).
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
V59_DIR = SCRIPT_DIR.parent / "v59"
V58_DIR = SCRIPT_DIR.parent / "v58"
V57_DIR = SCRIPT_DIR.parent / "v57"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(V59_DIR))
sys.path.insert(0, str(V58_DIR))
sys.path.insert(0, str(V57_DIR))

from model_v59 import KarvishModelV59
from schema_v60 import NUM_BIO_TAGS


class KarvishModelV60(KarvishModelV59):
    """
    Karvish Model V60: Universal Zero-Shot Mobile Agent with Calibrated Termination.
    Warm-start compatible with V59/V58 weights.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 19. Dedicated Calibrated Goal Termination Verifier
        # Inputs: u_active [384] + h_t [64] = 448D
        self.calibrated_term_head = nn.Sequential(
            nn.Linear(self.hidden_dim + self.task_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

        # 20. Stagnation & Modal Slip Recovery Head
        # Inputs: delta_features [64] + h_t [64] = 128D
        self.slip_recovery_head = nn.Sequential(
            nn.Linear(self.task_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(
        self,
        goal_ids: torch.Tensor,
        goal_mask: torch.Tensor,
        candidate_label_ids: torch.Tensor,
        candidate_label_mask: torch.Tensor,
        candidate_bounds: torch.Tensor,
        candidate_roles: torch.Tensor,
        candidate_flags: torch.Tensor,
        candidate_pad_mask: Optional[torch.Tensor] = None,
        candidate_num_features: Optional[torch.Tensor] = None,
        parent_indices: Optional[torch.Tensor] = None,
        child_embeddings: Optional[torch.Tensor] = None,
        child_pad_mask: Optional[torch.Tensor] = None,
        memory_bank: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        h_prev: Optional[torch.Tensor] = None,
        c_prev: Optional[torch.Tensor] = None,
        observable_delta: Optional[torch.Tensor] = None,
        target_candidate_idx: Optional[torch.Tensor] = None,
        active_subgoal_override: Optional[torch.Tensor] = None,
        evidence_observed: Optional[torch.Tensor] = None,
        slot_idx: Optional[torch.Tensor] = None,
        return_transition_pred: bool = False,
    ) -> Dict[str, Any]:
        
        # Invoke base V59 forward pass
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
            return_transition_pred=return_transition_pred,
        )

        u_active = out["u_active"]  # [B, hidden_dim]
        h_t = out["h_t"]            # [B, task_dim]

        # Step 19: Compute Calibrated Termination Probability
        term_features = torch.cat([u_active, h_t], dim=-1)
        term_logit = self.calibrated_term_head(term_features).squeeze(-1) # [B]
        raw_term_prob = torch.sigmoid(term_logit)

        # Enforce AGENTS.md mathematical bound: without verified physical completion evidence, cap below 0.15
        if evidence_observed is None:
            has_evid = torch.zeros(u_active.size(0), device=u_active.device)
        else:
            has_evid = (evidence_observed.view(-1) > 0.5).float()

        base_term_prob = out["termination_prob"]
        calibrated_term_prob = has_evid * torch.maximum(base_term_prob, raw_term_prob) + (1.0 - has_evid) * torch.clamp(raw_term_prob, max=0.15)

        # Step 20: Compute Slip Recovery Probability
        if observable_delta is None:
            observable_delta = torch.zeros(u_active.size(0), 10, device=u_active.device)
        delta_audit = self.delta_verifier(observable_delta)
        delta_features = delta_audit["delta_features"]
        slip_in = torch.cat([delta_features, h_t], dim=-1)
        slip_logit = self.slip_recovery_head(slip_in).squeeze(-1)
        slip_prob = torch.sigmoid(slip_logit)

        # Overwrite termination_prob with calibrated verifier probability
        out["termination_prob"] = calibrated_term_prob
        out["calibrated_term_logit"] = term_logit
        out["slip_prob"] = slip_prob
        out["slip_logit"] = slip_logit

        return out
