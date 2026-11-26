"""
KARVISH MULTIMODAL UNIVERSAL V61 — MULTIMODAL ACTION & CALIBRATED TERMINATION POLICY
Architecture Specification:
  1. Full 32,522 WordPiece vocabulary support (MiniLM-L6-v2 + 2,000 domain tokens).
  2. 3-Layer Deep Non-Linear Visual Adapter (512 -> 384 -> 384 with GeLU & LayerNorm).
  3. Residual Gated Multimodal Scoring:
     Ranking_Scores = Base_V60_Scores + Visual_Affinity_Bias(Visual_Proj, Goal_U_Active).
  4. Identity Zero-Initialized Layer Stacking (Net2Net):
     Visual affinity head is initialized to ZERO, ensuring 100% numerical parity with V60 at Step 0.
  5. Calibrated Goal Termination Verifier Head + Slip Recovery Head.
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
V60_DIR = SCRIPT_DIR.parent / "v60"
V59_DIR = SCRIPT_DIR.parent / "v59"
V58_DIR = SCRIPT_DIR.parent / "v58"
V57_DIR = SCRIPT_DIR.parent / "v57"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(V60_DIR))
sys.path.insert(0, str(V59_DIR))
sys.path.insert(0, str(V58_DIR))
sys.path.insert(0, str(V57_DIR))

from model_v60 import KarvishModelV60
from schema_v61 import NUM_BIO_TAGS
from vocab_v61 import (
    build_v61_extended_vocab,
    initialize_extended_embeddings,
    BASE_TOKENIZER_ID,
    EXTENDED_VOCAB_SIZE,
    BASE_VOCAB_SIZE,
)
from transformers import AutoTokenizer


class VisualProjectionAdapter(nn.Module):
    """3-layer non-linear visual adapter mapping 512-D MobileNetV4 features to 384-D latent space."""
    def __init__(self, in_dim: int = 512, hidden_dim: int = 384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class KarvishModelV61(KarvishModelV60):
    """
    Karvish Model V61: Multimodal Universal Mobile Agent with Vision Sight and Extended Vocab.
    Warm-started from V60 with Net2Net Identity Zero-Initialization.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 1. Expand Tokenizer and Embedding Table to 32,522 rows
        try:
            self.text_encoder.hf_model.resize_token_embeddings(EXTENDED_VOCAB_SIZE)
        except Exception:
            pass

        # 2. 3-Layer Deep Visual Adapter (512 -> 384)
        self.visual_adapter = VisualProjectionAdapter(in_dim=512, hidden_dim=self.hidden_dim)

        # 3. Multimodal Visual Affinity Head (Goal + Visual Candidate)
        self.visual_affinity_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        # Zero-initialize the visual affinity output projection
        # Guarantees that at Step 0, visual_affinity_head(x) = 0 -> output = Base_V60 (Identity Pass-Through)
        with torch.no_grad():
            nn.init.zeros_(self.visual_affinity_head[-1].weight)
            nn.init.zeros_(self.visual_affinity_head[-1].bias)

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
        candidate_visual_feats: Optional[torch.Tensor] = None,  # [B, N, 512]
    ) -> Dict[str, Any]:

        # Run base V60 forward pass
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

        # Multimodal Visual Fusion
        if candidate_visual_feats is not None:
            B, N, _ = candidate_visual_feats.shape
            vis_proj = self.visual_adapter(candidate_visual_feats)  # [B, N, 384]

            # Compute goal-visual cross-affinity
            u_active = out["u_active"]  # [B, 384]
            u_expanded = u_active.unsqueeze(1).expand(-1, N, -1)  # [B, N, 384]
            joint_pair = torch.cat([vis_proj, u_expanded], dim=-1)  # [B, N, 768]

            vis_bias = self.visual_affinity_head(joint_pair).squeeze(-1)  # [B, N]
            if candidate_pad_mask is not None:
                vis_bias = vis_bias.masked_fill(candidate_pad_mask == 0, 0.0)

            # Refine base ranking scores
            refined_scores = out["ranking_scores"] + vis_bias
            if candidate_pad_mask is not None:
                refined_scores = refined_scores.masked_fill(candidate_pad_mask == 0, -1e4)

            out["ranking_scores"] = refined_scores
            out["candidate_logits"] = refined_scores
            out["visual_affinity_bias"] = vis_bias
        else:
            out["candidate_logits"] = out["ranking_scores"]

        out["is_terminal_complete"] = out["calibrated_term_logit"]
        return out


def load_v60_weights_into_v61(model: KarvishModelV61, v60_ckpt_path: Path) -> KarvishModelV61:
    """
    Step 0 Warm-Start:
    1. Loads V60 checkpoint.
    2. Copies matching weights for candidate encoder, goal encoder, cross-attention, slot heads, and calibrated verifier.
    3. Keeps zero-init on visual affinity head.
    4. Applies subword mean decomposition initialization to new vocab tokens.
    """
    print(f">> [Step 0 Warm-Start] Loading V60 checkpoint from {v60_ckpt_path}...")
    ckpt_data = torch.load(v60_ckpt_path, map_location="cpu", weights_only=False)
    v60_state = ckpt_data.get("model_state_dict", ckpt_data)

    model_state = model.state_dict()
    matched = 0
    unmatched = 0

    for k, v in v60_state.items():
        if k in model_state:
            if model_state[k].shape == v.shape:
                model_state[k].copy_(v)
                matched += 1
            elif "token_embeddings" in k or "word_embeddings" in k:
                rows = min(model_state[k].shape[0], v.shape[0])
                model_state[k][:rows].copy_(v[:rows])
                matched += 1
            else:
                unmatched += 1
        else:
            unmatched += 1

    model.load_state_dict(model_state, strict=False)
    print(f">> [Step 0 Warm-Start] Successfully matched {matched} tensor parameters (Unmatched new layers: {unmatched}).")

    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_TOKENIZER_ID)
        new_tokens, _ = build_v61_extended_vocab()
        emb_layer = model.text_encoder.hf_model.embeddings.word_embeddings
        print(f">> Initializing {len(new_tokens)} extended vocabulary rows with Subword Mean Decomposition...")
        with torch.no_grad():
            for idx, token_str in enumerate(new_tokens):
                row_idx = BASE_VOCAB_SIZE + idx
                if row_idx < emb_layer.weight.shape[0]:
                    sub_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token_str))
                    valid_ids = [i for i in sub_ids if i is not None and 0 <= i < BASE_VOCAB_SIZE]
                    if valid_ids:
                        mean_vec = emb_layer.weight[valid_ids].mean(dim=0)
                        emb_layer.weight[row_idx].copy_(mean_vec)
                    else:
                        emb_layer.weight[row_idx].normal_(mean=0.0, std=0.02)
        print(">> Extended vocabulary embedding rows successfully initialized.")
    except Exception as e:
        print(f">> [Warning] Vocab mean init skipped: {e}")

    return model
