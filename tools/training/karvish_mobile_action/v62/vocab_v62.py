"""
KARVISH V62 VOCABULARY MODULE
Base Tokenizer: sentence-transformers/all-MiniLM-L6-v2 (30,522 WordPiece tokens)
Extended Vocab: 32,522 tokens (+2,000 domain tokens covering Indian grocery, food, transit, Hinglish & abbreviations)
100% backwards-compatible with V61 tokenizer & on-device assets.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
V61_DIR = SCRIPT_DIR.parent / "v61"
sys.path.insert(0, str(V61_DIR))

from vocab_v61 import (
    BASE_TOKENIZER_ID,
    BASE_VOCAB_SIZE,
    EXTENDED_VOCAB_SIZE,
    NEW_TOKENS_COUNT,
    DOMAIN_TERMS,
    generate_full_domain_token_list,
    build_v61_extended_vocab as build_v62_extended_vocab,
    initialize_extended_embeddings as initialize_v62_extended_embeddings,
)

__all__ = [
    "BASE_TOKENIZER_ID",
    "BASE_VOCAB_SIZE",
    "EXTENDED_VOCAB_SIZE",
    "NEW_TOKENS_COUNT",
    "DOMAIN_TERMS",
    "generate_full_domain_token_list",
    "build_v62_extended_vocab",
    "initialize_v62_extended_embeddings",
]
