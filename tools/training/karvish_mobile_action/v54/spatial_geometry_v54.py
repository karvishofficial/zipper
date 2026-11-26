"""
KARVISH V54 — CONTINUOUS 2D SPATIAL GEOMETRY & SCREEN ZONE ENGINE
Continuous normalized layout coordinates aligned with hidden state representation.

Key Specifications:
- Inputs: 10-dimensional continuous normalized geometry vector:
  [left, top, right, bottom, cx, cy, width, height, area, aspect_ratio]
- Output: 384-dimensional dense spatial embedding aligned with hidden state.
- Screen Zone Biasing:
  - Header Zone (y < 0.12): Search bars, back arrows, titles.
  - Body Zone (0.12 <= y <= 0.88): Scrollable content cards, lists, feeds.
  - Bottom Nav Zone (y > 0.88): Navigation tabs (Explore, Home, Profile, Checkout bar).
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
        zones = torch.ones_like(top, dtype=torch.long)
        zones = torch.where(is_header == 1, torch.zeros_like(zones), zones)
        zones = torch.where(is_bottom == 1, torch.full_like(zones, 2), zones)
        return zones

    def forward(self, bounds: torch.Tensor) -> torch.Tensor:
        """
        bounds: [B, N, 10] normalized geometry vectors
        returns: [B, N, hidden_dim] dense spatial embeddings
        """
        B, N, _ = bounds.shape
        coord_features = self.coord_mlp(bounds)  # [B, N, spatial_dim]

        top = bounds[:, :, 1]
        bottom = bounds[:, :, 3]
        zones = self.classify_zone(top, bottom)  # [B, N]
        zone_features = self.zone_embedding(zones)  # [B, N, spatial_dim]

        combined = torch.cat([coord_features, zone_features], dim=-1)  # [B, N, spatial_dim * 2]
        return self.proj_to_hidden(combined)  # [B, N, hidden_dim]
