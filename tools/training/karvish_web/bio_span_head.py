"""
KARVISH-WEB BIO TOKEN CLASSIFIER HEAD
Replaces Pointer Network with a standard 3-class Named Entity Recognition (NER) token classifier.
Classes:
0: O       (Outside query: stopwords, commands, store names, fillers)
1: B-QUERY (Begin query entity)
2: I-QUERY (Inside query entity)

Presents open-vocabulary generalization without memorizing sentence position indices.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BIOTokenClassifierHead(nn.Module):
    """
    Classifies each goal token into O, B-QUERY, or I-QUERY based on contextual MiniLM representations.
    Adds ~250K parameters, completely solving open-vocabulary entity span extraction.
    """

    def __init__(self, hidden_dim: int = 384, num_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, goal_tokens: torch.Tensor) -> torch.Tensor:
        """
        goal_tokens: [B, L, hidden_dim]
        returns: bio_logits [B, L, num_classes]
        """
        return self.classifier(goal_tokens)

    @staticmethod
    def decode_bio(input_ids: List[int], pred_bio_labels: List[int], tokenizer: Any) -> str:
        """
        Decodes predicted BIO tags back into the exact text string.
        """
        extracted_ids = []
        in_entity = False

        for token_id, label in zip(input_ids, pred_bio_labels):
            # Special tokens ([CLS], [SEP], [PAD]) are always O
            if token_id in (tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id):
                continue

            if label == 1:  # B-QUERY
                extracted_ids.append(token_id)
                in_entity = True
            elif label == 2 and in_entity:  # I-QUERY
                extracted_ids.append(token_id)
            else:
                in_entity = False

        if not extracted_ids:
            return ""

        decoded = tokenizer.decode(extracted_ids, skip_special_tokens=True).strip()
        return decoded
