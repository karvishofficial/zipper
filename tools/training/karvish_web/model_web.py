"""
KARVISH-WEB MODEL ARCHITECTURE
Dedicated Compact Web Action Model (14MB INT8).
Engineered specifically for Autonomous Browser Navigation & Form Execution.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from language_backbone import MiniLMTextEncoder
from spatial_geometry import SpatialGeometryEncoder
from hierarchical_container_attention import HierarchicalContainerAttention
from learned_span_head import LearnedSpanPointerHead
from bio_span_head import BIOTokenClassifierHead
from schema_web import CANONICAL_WEB_ACTIONS, WEB_ROLES, WEB_REGIONS


class KarvishWebModel(nn.Module):
    """
    Dedicated On-Device Web Action Intelligence Model.
    Compact ~24M parameters (~12-14 MB in INT8).
    Processes web DOM candidates in parallel in under 20ms on mobile NPU/CPU.
    """

    def __init__(
        self,
        hidden_dim: int = 384,
        num_canonical_actions: int = len(CANONICAL_WEB_ACTIONS),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_actions = num_canonical_actions

        # 1. MiniLM Language Backbone (384-dim, 6 layers, 12 heads)
        self.text_encoder = MiniLMTextEncoder(
            hidden_dim=hidden_dim,
            num_layers=6,
            num_heads=12,
            dim_feedforward=1536,
            dropout=dropout,
        )

        # 2. Continuous 2D Spatial Geometry Encoder
        self.spatial_encoder = SpatialGeometryEncoder(
            input_dim=10,
            spatial_dim=64,
            hidden_dim=64,
        )

        # 3. Web Role & Spatial Region Embeddings
        self.role_embed = nn.Embedding(len(WEB_ROLES) + 4, 64)
        self.region_embed = nn.Embedding(len(WEB_REGIONS) + 4, 64)
        self.flags_mlp = nn.Sequential(
            nn.Linear(8, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )

        # Candidate feature fusion: text(384) + spatial(64) + role(64) + region(64) + flags(64) = 640 -> 384
        self.cand_fuse = nn.Sequential(
            nn.Linear(hidden_dim + 64 + 64 + 64 + 64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # 5. Goal Cross-Attention (Contextualizing Candidates against Goal)
        self.goal_cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=6,
            dropout=dropout,
            batch_first=True,
        )
        self.post_cross_norm = nn.LayerNorm(hidden_dim)

        # 6. Learned Span Pointer Head for text argument extraction from goal
        self.span_pointer_head = LearnedSpanPointerHead(
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # 7. Candidate Ranking Head (Logits per candidate)
        self.rank_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # 8. Web Action Type Head (CLICK, TYPE_TEXT, SUBMIT, SCROLL, SELECT, EXTRACT, WAIT, NAVIGATE)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_canonical_actions),
        )

        # 9. BIO Token Classifier Head for Open-Vocabulary Query Entity Extraction
        self.bio_head = BIOTokenClassifierHead(
            hidden_dim=hidden_dim,
            num_classes=3,
            dropout=dropout,
        )

    def forward(
        self,
        goal_input_ids: torch.Tensor,
        goal_attention_mask: torch.Tensor,
        cand_input_ids: torch.Tensor,
        cand_attention_mask: torch.Tensor,
        cand_spatial: torch.Tensor,
        cand_roles: torch.Tensor,
        cand_regions: torch.Tensor,
        cand_flags: torch.Tensor,
        cand_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for parallel web action prediction.
        Batch size B, Goal tokens L_g, Num candidates K, Candidate tokens L_c.
        """
        B, L_g = goal_input_ids.shape
        _, K, L_c = cand_input_ids.shape

        # 1. Encode Goal
        goal_enc = self.text_encoder(goal_input_ids, goal_attention_mask)
        goal_tokens = goal_enc["last_hidden_state"]  # [B, L_g, hidden_dim]
        goal_pooled = goal_enc["pooled_output"]     # [B, hidden_dim]

        # 2. Encode Candidates Text in parallel
        cand_ids_flat = cand_input_ids.view(B * K, L_c)
        cand_mask_flat = cand_attention_mask.view(B * K, L_c)
        cand_enc = self.text_encoder(cand_ids_flat, cand_mask_flat)
        cand_text_pooled = cand_enc["pooled_output"].view(B, K, self.hidden_dim)

        # 3. Spatial, Role, Region, and Flag Embeddings
        cand_spatial_emb = self.spatial_encoder(cand_spatial)
        cand_role_emb = self.role_embed(cand_roles)
        cand_region_emb = self.region_embed(cand_regions)
        cand_flag_emb = self.flags_mlp(cand_flags)

        # Fuse candidate representations
        cand_features = torch.cat([
            cand_text_pooled,
            cand_spatial_emb,
            cand_role_emb,
            cand_region_emb,
            cand_flag_emb,
        ], dim=-1)
        cand_emb = self.cand_fuse(cand_features)  # [B, K, hidden_dim]

        # 4. Cross-Attention: Candidates attend to Goal Tokens
        attn_out, _ = self.goal_cross_attention(
            query=cand_emb,
            key=goal_tokens,
            value=goal_tokens,
            key_padding_mask=~goal_attention_mask.bool(),
        )
        cand_contextualized = self.post_cross_norm(cand_emb + attn_out)

        # 5. Candidate Ranking Logits
        rank_logits = self.rank_head(cand_contextualized).squeeze(-1)  # [B, K]
        # Mask out padded candidates
        rank_logits = rank_logits.masked_fill(~cand_mask.bool(), -1e4)

        # 6. Action Type Prediction (Conditioned on Best Candidate + Goal)
        # Softmax pooling over candidates to get target candidate representation
        cand_probs = F.softmax(rank_logits, dim=-1).unsqueeze(-1)  # [B, K, 1]
        selected_cand_repr = (cand_contextualized * cand_probs).sum(dim=1)  # [B, hidden_dim]

        action_context = torch.cat([goal_pooled, selected_cand_repr], dim=-1)
        action_logits = self.action_head(action_context)  # [B, num_actions]

        # 7. Span Pointer Head (Start & End tokens in Goal)
        span_out = self.span_pointer_head(
            goal_tokens=goal_tokens,
            target_candidate_embedding=selected_cand_repr,
            goal_mask=goal_attention_mask,
        )

        # 8. BIO Token Classification Logits [B, L, 3]
        bio_logits = self.bio_head(goal_tokens)

        return {
            "rank_logits": rank_logits,
            "action_logits": action_logits,
            "span_start_logits": span_out["start_logits"],
            "span_end_logits": span_out["end_logits"],
            "bio_logits": bio_logits,
        }
