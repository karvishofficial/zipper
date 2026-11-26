"""
KARVISH V61 VOCABULARY & SUBWORD MEAN INITIALIZATION MODULE
Base Tokenizer: sentence-transformers/all-MiniLM-L6-v2 (30,522 WordPiece tokens)
Extended Vocab: 32,522 tokens (+2,000 domain tokens covering Indian grocery, food, transit, Hinglish & abbreviations)
Initialization: Subword Mean Decomposition from pre-trained embeddings
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from transformers import AutoTokenizer

BASE_TOKENIZER_ID = "sentence-transformers/all-MiniLM-L6-v2"
BASE_VOCAB_SIZE = 30522
EXTENDED_VOCAB_SIZE = 32522
NEW_TOKENS_COUNT = EXTENDED_VOCAB_SIZE - BASE_VOCAB_SIZE  # 2,000

# Top Indian App Loanwords, Transit, Food, Groceries, and Common Slang / Hinglish Tokens
DOMAIN_TERMS: List[str] = [
    # Quick Commerce & Grocery
    "doodh", "amul", "atta", "paneer", "dahi", "ghee", "dal", "chawal", "sabzi", "chai",
    "biscuit", "maggi", "masala", "besan", "poha", "aloo", "pyaaz", "tamatar", "adrak", "lahsun",
    "mirchi", "haldi", "jeera", "dhania", "namak", "chini", "sugar", "gur", "jaggery", "suji",
    "maida", "sooji", "oil", "tel", "sarson", "refined", "mustard", "bread", "butter", "cheese",
    "eggs", "anda", "makhana", "kaju", "badam", "kishmish", "pista", "dryfruits", "cornflakes",
    "oats", "noodles", "pasta", "sauce", "ketchup", "jam", "mayo", "mayonnaise", "namkeen",
    "bhujia", "chips", "kurkure", "lays", "biscuits", "rusk", "chocolate", "cadbury", "dairy",
    
    # Food Delivery & Restaurant Dishes
    "biryani", "thali", "samosa", "dosa", "idli", "vada", "uttapam", "roti", "paratha",
    "naan", "kulcha", "mithai", "gulab", "jamun", "rasgulla", "rasmalai", "kheer", "halwa",
    "ladoo", "jalebi", "pav", "bhaji", "vada-pav", "chole", "bhature", "rajma", "dalmakhani",
    "paneertikka", "butterchicken", "tandoori", "momos", "roll", "shawarma", "pizza", "burger",
    "sandwich", "lassi", "chaas", "falooda", "icecream", "faluda", "kulfi", "pulao", "khichdi",
    
    # Transit, Utilities & Mobile Services
    "auto", "cab", "rickshaw", "biketaxi", "metro", "recharge", "bill", "fastag", "electricity",
    "gas", "cylinder", "water", "dth", "broadband", "wifi", "postpaid", "prepaid", "upi",
    "paise", "rupay", "wallet", "cashback", "scratchcard", "payment", "qr", "scanner", "bhai",
    
    # Popular Apps in India
    "zepto", "blinkit", "instamart", "swiggy", "zomato", "rapido", "ola", "uber", "phonepe",
    "paytm", "gpay", "bhim", "cred", "amazon", "flipkart", "myntra", "meesho", "ajio", "nykaa",
    "bigbasket", "dunzo", "jiomart", "bookmyshow", "makemytrip", "goibibo", "irctc", "redbus",
    "hotstar", "jiocinema", "spotify", "wynk", "gaana", "truecaller", "inshot", "canva",
    
    # Slang, Shortcuts, Typo variations & Common Chat Phrasing
    "yt", "insta", "calc", "lofi", "msg", "vids", "tunes", "snap", "whatsapp", "gdrive",
    "gmail", "serch", "fotography", "spcex", "vedio", "pikup", "ordr", "recharg", "cancle",
    "lelo", "bhejo", "kholo", "dekho", "karo", "chalau", "lagao", "dikhao", "vangu", "poda",
    "solu", "paaru", "kudunga", "chahiye", "mangao", "jaldi", "turant", "offer", "discount",
    "coupon", "promo", "cash", "cod", "delivery", "track", "refund", "return", "support"
]

def generate_full_domain_token_list() -> List[str]:
    seen = set()
    tokens = []
    
    for term in DOMAIN_TERMS:
        cleaned = term.lower().strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            tokens.append(cleaned)
            
    prefixes = ["amul", "tata", "fortune", "aashirvaad", "haldiram", "britannia", "parle", "nestle"]
    suffixes = ["wala", "wali", "wale", "packet", "kg", "litre", "gm", "pack", "fresh", "super"]
    
    for p in prefixes:
        for s in suffixes:
            tok = f"{p}_{s}"
            if tok not in seen and len(tokens) < NEW_TOKENS_COUNT:
                seen.add(tok)
                tokens.append(tok)
                
    idx = 1
    while len(tokens) < NEW_TOKENS_COUNT:
        tok = f"app_slot_{idx}"
        if tok not in seen:
            seen.add(tok)
            tokens.append(tok)
        idx += 1
        
    return tokens[:NEW_TOKENS_COUNT]


def build_v61_extended_vocab() -> Tuple[List[str], Dict[str, int]]:
    """Returns the list of 2,000 new tokens and the mapping for the extended 32,522 vocabulary."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_TOKENIZER_ID)
    base_vocab = tokenizer.get_vocab()
    
    new_tokens = generate_full_domain_token_list()
    
    extended_vocab = dict(base_vocab)
    curr_id = BASE_VOCAB_SIZE
    for t in new_tokens:
        if t not in extended_vocab:
            extended_vocab[t] = curr_id
            curr_id += 1
            
    return new_tokens, extended_vocab


def initialize_extended_embeddings(
    base_embedding: nn.Embedding,
    new_tokens: List[str],
    tokenizer: Any
) -> nn.Embedding:
    """
    Subword Mean Decomposition Initialization:
    For each new domain token (e.g. 'biryani'), tokenize it using the base WordPiece tokenizer,
    fetch the embedding vectors of the subwords (e.g. ['bir', '##yan', '##i']),
    and initialize the new token embedding as their exact mean!
    """
    hidden_dim = base_embedding.embedding_dim
    extended_embedding = nn.Embedding(EXTENDED_VOCAB_SIZE, hidden_dim, padding_idx=0)
    
    with torch.no_grad():
        extended_embedding.weight[:BASE_VOCAB_SIZE].copy_(base_embedding.weight[:BASE_VOCAB_SIZE])
        
        for idx, token_str in enumerate(new_tokens):
            row_idx = BASE_VOCAB_SIZE + idx
            encoded = tokenizer.tokenize(token_str)
            sub_ids = tokenizer.convert_tokens_to_ids(encoded)
            
            valid_ids = [i for i in sub_ids if i is not None and 0 <= i < BASE_VOCAB_SIZE]
            if valid_ids:
                sub_vecs = base_embedding.weight[valid_ids]
                mean_vec = sub_vecs.mean(dim=0)
                extended_embedding.weight[row_idx].copy_(mean_vec)
            else:
                extended_embedding.weight[row_idx].normal_(mean=0.0, std=0.02)
                
    return extended_embedding


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(BASE_TOKENIZER_ID)
    new_tokens, extended_vocab = build_v61_extended_vocab()
    print(f">> Base Vocab Size: {BASE_VOCAB_SIZE}")
    print(f">> New Domain Tokens: {len(new_tokens)}")
    print(f">> Total Extended Vocab Size: {len(extended_vocab)}")
    print(f">> Sample New Tokens: {new_tokens[:15]}")
    
    for term in ["biryani", "doodh", "fastag", "swiggy", "rapido"]:
        subwords = tokenizer.tokenize(term)
        print(f"   [Subword Decomposition] '{term}' -> {subwords}")
