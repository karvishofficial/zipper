"""
KARVISH V52 — PHASE 4: CONTINUOUS 2D SPATIAL GEOMETRY & SCREEN ZONE ENGINE
Replaces crude 18-box static grid buckets with continuous LayoutLM-style relative coordinates.

Key Specifications:
- Inputs: 10-dimensional continuous normalized geometry vector:
  [left, top, right, bottom, cx, cy, width, height, area, aspect_ratio]
- Output: 64-dimensional dense spatial embedding aligned with hidden state.
- Screen Zone Biasing:
  - Header Zone (y < 0.12): Search bars, back arrows, titles.
  - Body Zone (0.12 <= y <= 0.88): Scrollable content cards, lists, feeds.
  - Bottom Nav Zone (y > 0.88): Navigation tabs (Explore, Home, Profile).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialGeometryEncoder(nn.Module):
    """
    Encodes 2D continuous spatial bounding boxes into dense spatial representations.
    Mathematically enforces layout awareness (e.g. Header vs Feed vs Nav Bar).
    """

    def __init__(self, input_dim: int = 10, spatial_dim: int = 64, hidden_dim: int = 384):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.hidden_dim = hidden_dim

        # 2-layer continuous coordinate MLP
        self.coord_mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, spatial_dim),
            nn.LayerNorm(spatial_dim),
        )

        # 3 Screen Zone Embeddings (Header, Body, Bottom Nav)
        self.zone_embedding = nn.Embedding(3, spatial_dim)

        # Projection to model hidden dimension
        self.proj_to_hidden = nn.Linear(spatial_dim * 2, hidden_dim)

    def classify_zone(self, top: torch.Tensor, bottom: torch.Tensor) -> torch.Tensor:
        """
        Classifies screen zone:
        0: Header (top < 0.12)
        1: Body / Viewport (0.12 <= top and bottom <= 0.88)
        2: Bottom Nav Bar (bottom > 0.88)
        """
        is_header = (top < 0.12).long()
        is_bottom = (bottom > 0.88).long()
        # 0 = header, 2 = bottom nav, 1 = body
        zone = torch.ones_like(top, dtype=torch.long)
        zone = torch.where(top < 0.12, torch.zeros_like(zone), zone)
        zone = torch.where(bottom > 0.88, torch.full_like(zone, 2), zone)
        return zone

    def forward(
        self,
        bounds_vectors: torch.Tensor,   # [B, N, 10] or [N, 10]
    ) -> torch.Tensor:
        is_batched = (bounds_vectors.dim() == 3)
        if not is_batched:
            bounds_vectors = bounds_vectors.unsqueeze(0)

        B, N, D = bounds_vectors.shape

        # Extract continuous coordinates
        coord_features = self.coord_mlp(bounds_vectors)  # [B, N, spatial_dim]

        # Extract vertical coordinates for zone classification
        tops = bounds_vectors[..., 1]
        bottoms = bounds_vectors[..., 3]
        zones = self.classify_zone(tops, bottoms)         # [B, N]
        zone_features = self.zone_embedding(zones)       # [B, N, spatial_dim]

        combined = torch.cat([coord_features, zone_features], dim=-1)  # [B, N, spatial_dim * 2]
        spatial_embeddings = self.proj_to_hidden(combined)              # [B, N, hidden_dim]

        if not is_batched:
            spatial_embeddings = spatial_embeddings.squeeze(0)

        return spatial_embeddings
