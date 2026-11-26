"""
KARVISH-WEB TRAINING PIPELINE WITH BIO TOKEN CLASSIFIER & 90% LOCKED RANKER
Initializes from the verified 90% locked ranker baseline (karvish_web_ranker_v1_locked.pt)
Trains the BIO Token Classification Head for open-vocabulary entity extraction
while strictly preserving the 90% candidate ranking and action classification heads.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from model_web import KarvishWebModel
from bio_span_head import BIOTokenClassifierHead
from schema_web import CANONICAL_WEB_ACTIONS, WEB_ROLES, WEB_REGIONS

MAX_GOAL_LEN = 32
MAX_CANDIDATES = 24
MAX_CAND_LEN = 16

ROLE_MAP = {r: i for i, r in enumerate(WEB_ROLES)}
REGION_MAP = {r: i for i, r in enumerate(WEB_REGIONS)}


class WebDataset(Dataset):
    def __init__(self, jsonl_path: Path, tokenizer: Any, is_train: bool = False):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
        self.tokenizer = tokenizer
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        goal = sample["goal"]
        cands = list(sample["candidates"][:MAX_CANDIDATES])
        target_idx = min(sample["target_idx"], len(cands) - 1)
        action_type = sample["action_type"]
        span_start = min(sample.get("span_start", 0), MAX_GOAL_LEN - 1)
        span_end = min(sample.get("span_end", 0), MAX_GOAL_LEN - 1)

        # Candidate Permutation Augmentation for training
        if self.is_train and len(cands) > 1:
            indices = list(range(len(cands)))
            random.shuffle(indices)
            new_cands = [cands[i] for i in indices]
            target_idx = indices.index(target_idx)
            cands = new_cands

        # 1. Goal Encoding
        g_enc = self.tokenizer(
            goal,
            max_length=MAX_GOAL_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        goal_ids = g_enc["input_ids"].squeeze(0)
        goal_mask = g_enc["attention_mask"].squeeze(0)

        # 2. Candidate Encoding (Padded to MAX_CANDIDATES for batch stacking)
        num_real = min(len(cands), MAX_CANDIDATES)
        cand_ids = torch.zeros((MAX_CANDIDATES, MAX_CAND_LEN), dtype=torch.long)
        cand_attn = torch.zeros((MAX_CANDIDATES, MAX_CAND_LEN), dtype=torch.long)
        cand_spatial = torch.zeros((MAX_CANDIDATES, 10), dtype=torch.float32)
        cand_roles = torch.zeros(MAX_CANDIDATES, dtype=torch.long)
        cand_regions = torch.zeros(MAX_CANDIDATES, dtype=torch.long)
        cand_flags = torch.zeros((MAX_CANDIDATES, 8), dtype=torch.float32)
        cand_mask = torch.zeros(MAX_CANDIDATES, dtype=torch.long)
        cand_mask[:num_real] = 1

        for i in range(num_real):
            c = cands[i]
            c_text = c.get("label", "")
            c_enc = self.tokenizer(
                c_text,
                max_length=MAX_CAND_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            cand_ids[i] = c_enc["input_ids"].squeeze(0)
            cand_attn[i] = c_enc["attention_mask"].squeeze(0)

            spatial = c.get("spatial", [0.0] * 10)
            cand_spatial[i] = torch.tensor(spatial[:10], dtype=torch.float32)

            role_str = c.get("role", "BUTTON")
            cand_roles[i] = ROLE_MAP.get(role_str, 0)

            region_str = c.get("region", "BODY")
            cand_regions[i] = REGION_MAP.get(region_str, 1)

            flags = c.get("flags", [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
            cand_flags[i] = torch.tensor(flags[:8], dtype=torch.float32)

        # 3. BIO Label Construction [MAX_GOAL_LEN] (0: O, 1: B-QUERY, 2: I-QUERY, -100: PAD)
        bio_labels = torch.full((MAX_GOAL_LEN,), -100, dtype=torch.long)
        for i in range(MAX_GOAL_LEN):
            if goal_mask[i] == 1:
                bio_labels[i] = 0  # Default to O for active tokens

        if span_start > 0 or span_end > 0:
            if span_start < MAX_GOAL_LEN:
                bio_labels[span_start] = 1  # B-QUERY
            for i in range(span_start + 1, min(span_end + 1, MAX_GOAL_LEN)):
                bio_labels[i] = 2  # I-QUERY

        return {
            "goal_input_ids": goal_ids,
            "goal_attention_mask": goal_mask,
            "cand_input_ids": cand_ids,
            "cand_attention_mask": cand_attn,
            "cand_spatial": cand_spatial,
            "cand_roles": cand_roles,
            "cand_regions": cand_regions,
            "cand_flags": cand_flags,
            "cand_mask": cand_mask,
            "target_idx": torch.tensor(target_idx, dtype=torch.long),
            "action_type": torch.tensor(action_type, dtype=torch.long),
            "span_start": torch.tensor(span_start, dtype=torch.long),
            "span_end": torch.tensor(span_end, dtype=torch.long),
            "bio_labels": bio_labels,
        }


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Karvish-Web with BIO Head on device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    data_dir = SCRIPT_DIR / "data"
    train_ds = WebDataset(data_dir / "web_train.jsonl", tokenizer, is_train=True)
    val_ds = WebDataset(data_dir / "web_val.jsonl", tokenizer, is_train=False)

    batch_size = 32 if torch.cuda.is_available() else 16
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())

    model = KarvishWebModel().to(device)

    # Initialize from the LOCKED 90% Ranker Checkpoint
    checkpoint_dir = SCRIPT_DIR / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    locked_ckpt = checkpoint_dir / "karvish_web_ranker_v1_locked.pt"

    if locked_ckpt.exists():
        state = torch.load(locked_ckpt, map_location=device)
        loaded_dict = state.get("model_state_dict", state)
        model.load_state_dict(loaded_dict, strict=False)
        print(f"[*] Initialized model from LOCKED 90% Ranker baseline: {locked_ckpt}")

    # Learning rates: Protect the 90% Ranker by keeping ranker/backbone lr low, train bio_head with high lr
    bio_params = list(model.bio_head.parameters())
    backbone_params = list(model.text_encoder.parameters())
    ranker_params = (
        list(model.rank_head.parameters()) +
        list(model.action_head.parameters()) +
        list(model.spatial_encoder.parameters()) +
        list(model.cand_fuse.parameters())
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": bio_params, "lr": 5e-4},
            {"params": backbone_params, "lr": 5e-6},
            {"params": ranker_params, "lr": 1e-5},
        ],
        weight_decay=0.01,
    )

    epochs = 6  # 6 focused epochs for BIO convergence
    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    criterion_rank = nn.CrossEntropyLoss()
    criterion_action = nn.CrossEntropyLoss()
    criterion_bio = nn.CrossEntropyLoss(ignore_index=-100)

    best_val_rank_acc = 0.0
    best_ckpt_path = checkpoint_dir / "karvish_web_best.pt"

    print("\n" + "=" * 75)
    print(f"STARTING BIO HEAD TRAINING WHILE PRESERVING 90% RANKER ({epochs} EPOCHS)")
    print(f"Dataset: Train={len(train_ds)} | Val={len(val_ds)} | Batch={batch_size}")
    print("=" * 75)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        train_rank_correct = 0
        train_act_correct = 0
        train_bio_correct = 0
        train_bio_total = 0
        train_total = 0

        for batch in train_loader:
            optimizer.zero_grad()

            goal_ids = batch["goal_input_ids"].to(device)
            goal_mask = batch["goal_attention_mask"].to(device)
            cand_ids = batch["cand_input_ids"].to(device)
            cand_mask_t = batch["cand_attention_mask"].to(device)
            cand_spatial = batch["cand_spatial"].to(device)
            cand_roles = batch["cand_roles"].to(device)
            cand_regions = batch["cand_regions"].to(device)
            cand_flags = batch["cand_flags"].to(device)
            cand_active = batch["cand_mask"].to(device)

            targets = batch["target_idx"].to(device)
            action_targets = batch["action_type"].to(device)
            bio_targets = batch["bio_labels"].to(device)

            out = model(
                goal_input_ids=goal_ids,
                goal_attention_mask=goal_mask,
                cand_input_ids=cand_ids,
                cand_attention_mask=cand_mask_t,
                cand_spatial=cand_spatial,
                cand_roles=cand_roles,
                cand_regions=cand_regions,
                cand_flags=cand_flags,
                cand_mask=cand_active,
            )

            scaled_rank_logits = out["rank_logits"] / 0.07
            loss_rank = criterion_rank(scaled_rank_logits, targets)
            loss_action = criterion_action(out["action_logits"], action_targets)
            loss_bio = criterion_bio(out["bio_logits"].view(-1, 3), bio_targets.view(-1))

            loss = loss_rank * 1.0 + loss_action * 1.0 + loss_bio * 2.5

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            train_rank_correct += (out["rank_logits"].argmax(dim=-1) == targets).sum().item()
            train_act_correct += (out["action_logits"].argmax(dim=-1) == action_targets).sum().item()

            active_tokens = (bio_targets != -100)
            pred_bio = out["bio_logits"].argmax(dim=-1)
            train_bio_correct += ((pred_bio == bio_targets) & active_tokens).sum().item()
            train_bio_total += active_tokens.sum().item()
            train_total += targets.size(0)

        # Validation
        model.eval()
        val_rank_correct = 0
        val_act_correct = 0
        val_bio_correct = 0
        val_bio_total = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                goal_ids = batch["goal_input_ids"].to(device)
                goal_mask = batch["goal_attention_mask"].to(device)
                cand_ids = batch["cand_input_ids"].to(device)
                cand_mask_t = batch["cand_attention_mask"].to(device)
                cand_spatial = batch["cand_spatial"].to(device)
                cand_roles = batch["cand_roles"].to(device)
                cand_regions = batch["cand_regions"].to(device)
                cand_flags = batch["cand_flags"].to(device)
                cand_active = batch["cand_mask"].to(device)

                targets = batch["target_idx"].to(device)
                action_targets = batch["action_type"].to(device)
                bio_targets = batch["bio_labels"].to(device)

                out = model(
                    goal_input_ids=goal_ids,
                    goal_attention_mask=goal_mask,
                    cand_input_ids=cand_ids,
                    cand_attention_mask=cand_mask_t,
                    cand_spatial=cand_spatial,
                    cand_roles=cand_roles,
                    cand_regions=cand_regions,
                    cand_flags=cand_flags,
                    cand_mask=cand_active,
                )

                val_rank_correct += (out["rank_logits"].argmax(dim=-1) == targets).sum().item()
                val_act_correct += (out["action_logits"].argmax(dim=-1) == action_targets).sum().item()

                active_tokens = (bio_targets != -100)
                pred_bio = out["bio_logits"].argmax(dim=-1)
                val_bio_correct += ((pred_bio == bio_targets) & active_tokens).sum().item()
                val_bio_total += active_tokens.sum().item()
                val_total += targets.size(0)

        val_rank_acc = (val_rank_correct / val_total) * 100
        val_act_acc = (val_act_correct / val_total) * 100
        val_bio_acc = (val_bio_correct / max(1, val_bio_total)) * 100
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] | "
            f"Rank Top-1: {val_rank_acc:.2f}% | Act Acc: {val_act_acc:.2f}% | BIO Token Acc: {val_bio_acc:.2f}%"
        )

        # Save checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_rank_acc": val_rank_acc,
            "val_act_acc": val_act_acc,
            "val_bio_acc": val_bio_acc,
        }, str(best_ckpt_path))

    print("=" * 75)
    print(f"Training Complete! Saved to: {best_ckpt_path}")
    print("=" * 75)


if __name__ == "__main__":
    train()
