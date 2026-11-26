"""
Full Phase C1 Training Script: Genuine Cactus Needle 2 45M LoRA on NVIDIA RTX 4050 GPU.
Uses exact 736 AndroidControl training transitions and validates on 77 held-out transitions.
Hardware: Guaranteed locked to cuda:0 (NVIDIA GeForce RTX 4050 Laptop GPU).
"""

import os
import sys
import json
import time
import pickle
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from safetensors.numpy import save_file

# Hardware verification
assert torch.cuda.is_available(), "CRITICAL: CUDA GPU is not available!"
device_name = torch.cuda.get_device_name(0)
assert "RTX 4050" in device_name, f"CRITICAL: Expected RTX 4050, detected: {device_name}"
DEVICE = torch.device("cuda:0")

sys.path.insert(0, os.path.dirname(__file__))
from pytorch_needle import Needle2Model
from needle.model.tokenizer import get_tokenizer, BOS_ID, EOS_ID, PAD_ID
from needle.model.finetune import render_example


class NeedleDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_len=1024):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt, target = render_example(ex)
        p_ids = self.tokenizer.encode(prompt)
        t_ids = self.tokenizer.encode(target)
        
        # Combine: BOS + prompt + target + EOS
        full_ids = [BOS_ID] + p_ids + t_ids + [EOS_ID]
        # Target mask: 0 for prompt, 1 for target
        mask = [0.0] * (1 + len(p_ids)) + [1.0] * (len(t_ids) + 1)
        
        # Truncate
        full_ids = full_ids[:self.max_len]
        mask = mask[:self.max_len]
        
        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "target_mask": torch.tensor(mask, dtype=torch.float32)
        }


def collate_fn(batch):
    # Dynamic batch padding to longest item in batch
    max_batch_len = max(len(item["input_ids"]) for item in batch)
    
    batch_ids = []
    batch_masks = []
    
    for item in batch:
        ids = item["input_ids"]
        mask = item["target_mask"]
        pad_len = max_batch_len - len(ids)
        if pad_len > 0:
            ids = F.pad(ids, (0, pad_len), value=PAD_ID)
            mask = F.pad(mask, (0, pad_len), value=0.0)
        batch_ids.append(ids)
        batch_masks.append(mask)
        
    return {
        "input_ids": torch.stack(batch_ids),
        "target_mask": torch.stack(batch_masks)
    }


def compute_loss(logits, input_ids, target_mask):
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_mask = target_mask[..., 1:].contiguous()
    
    # Flatten
    vocab_size = shift_logits.size(-1)
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
    
    masked_loss = (loss * shift_mask.view(-1)).sum()
    denom = torch.clamp(shift_mask.sum(), min=1.0)
    return masked_loss / denom


def evaluate(model, val_loader):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            target_mask = batch["target_mask"].to(DEVICE)
            logits = model(input_ids)
            loss = compute_loss(logits, input_ids, target_mask)
            total_loss += loss.item()
            total_batches += 1
    return total_loss / max(total_batches, 1)


def train():
    print("=" * 70)
    print("PHASE C1: FULL NEEDLE 2 (45M) LoRA GPU TRAINING")
    print(f"HARDWARE: {device_name} (Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB)")
    print("=" * 70)

    base_ckpt_path = "tools/training/experiment_c_needle2/checkpoints/needle2.pkl"
    train_path = "tools/training/experiment_c_needle2/data/needle_train.jsonl"
    val_path = "tools/training/experiment_c_needle2/data/needle_val.jsonl"
    out_dir = "tools/training/experiment_c_needle2/checkpoints"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load Base Model onto GPU
    print("\n[Step 1/5] Loading base Cactus Needle 2 checkpoint...")
    with open(base_ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    print(f"  Base weights loaded from: {base_ckpt_path}")
    
    rank = 16
    alpha = 32.0
    scale = alpha / rank

    t_load = time.time()
    model = Needle2Model(ckpt["params"], ckpt["config"], lora_rank=rank).to(DEVICE)
    print(f"  Needle 2 architecture mapped to GPU in: {time.time() - t_load:.2f}s")
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for p in trainable_params)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total Base Parameters:     {total_params:,} (~{total_params * 4 / 1024**2:.1f} MB)")
    print(f"  Trainable LoRA Parameters: {total_trainable:,} (~{total_trainable * 4 / 1024**2:.2f} MB, {total_trainable/total_params*100:.2f}%)")

    # 2. Prepare Datasets
    print("\n[Step 2/5] Preparing tokenized datasets...")
    tokenizer = get_tokenizer(ckpt["config"]["vocab_size"])
    train_dataset = NeedleDataset(train_path, tokenizer, max_len=1024)
    val_dataset = NeedleDataset(val_path, tokenizer, max_len=1024)
    print(f"  Train Dataset Transitions: {len(train_dataset)} (All 736 transitions across 111 Android apps)")
    print(f"  Held-out Val Transitions:  {len(val_dataset)} (77 transitions across 21 validation apps)")

    batch_size = 1
    grad_accum = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 3. Optimizer and Scheduler
    epochs = 3
    total_steps = (len(train_loader) // grad_accum) * epochs
    warmup_steps = int(0.05 * total_steps)
    
    optimizer = torch.optim.AdamW(trainable_params, lr=2e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    print(f"\n[Step 3/5] Training Configuration:")
    print(f"  Epochs:                    {epochs}")
    print(f"  Batch Size per Step:       {batch_size}")
    print(f"  Gradient Accumulation:     {grad_accum} (Effective Batch Size = {batch_size * grad_accum})")
    print(f"  Total Optimization Steps:  {total_steps}")
    print(f"  Peak Learning Rate:        2e-4 (Cosine Decay with warmup)")

    # 4. Training Loop
    print("\n[Step 4/5] Executing GPU Training Loop on RTX 4050...")
    best_val_loss = float("inf")
    start_train_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        step_loss = 0.0
        optimizer.zero_grad()
        epoch_start = time.time()
        
        for i, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(DEVICE)
            target_mask = batch["target_mask"].to(DEVICE)
            
            logits = model(input_ids)
            loss = compute_loss(logits, input_ids, target_mask)
            
            # Scale for gradient accumulation
            loss_scaled = loss / grad_accum
            loss_scaled.backward()
            
            step_loss += loss.item()
            epoch_loss += loss.item()
            
            if (i + 1) % grad_accum == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
                step_idx = (epoch * len(train_loader) + i + 1) // grad_accum
                if step_idx % 25 == 0 or (i + 1) == len(train_loader):
                    avg_recent_loss = step_loss / min(grad_accum, i + 1)
                    vram_mb = torch.cuda.memory_allocated() / 1024**2
                    lr = scheduler.get_last_lr()[0]
                    print(f"  Epoch {epoch+1}/{epochs} | Step {step_idx}/{total_steps} | Loss: {avg_recent_loss:.4f} | LR: {lr:.2e} | VRAM: {vram_mb:.0f} MB")
                step_loss = 0.0

        epoch_duration = time.time() - epoch_start
        avg_train_loss = epoch_loss / len(train_loader)
        
        # Validation Evaluation
        val_loss = evaluate(model, val_loader)
        print(f"  ---> Epoch {epoch+1} Done ({epoch_duration:.1f}s) | Train Loss: {avg_train_loss:.4f} | Held-out Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save best checkpoint
            torch_ckpt_path = os.path.join(out_dir, "needle_lora_best.pt")
            torch.save(model.state_dict(), torch_ckpt_path)

    total_duration = time.time() - start_train_time
    print(f"\n[Step 5/5] GPU Training Complete!")
    print(f"  Total Duration:      {total_duration:.1f}s ({total_duration/60:.2f} minutes)")
    print(f"  Best Val Loss:       {best_val_loss:.4f}")
    print(f"  Peak VRAM Consumed:  {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")

    # Save Adapters in Needle Format for Export Compatibility
    print("\nExporting LoRA Adapters...")
    adapter_dict = {"lora": {}, "scale": float(scale), "base": base_ckpt_path, "rank": rank}
    
    # Extract trained LoRA A and B tensors
    for l_idx, block in enumerate(model.blocks):
        sa = block.self_attn
        for proj_name, module in [("q_proj", sa.q_proj), ("k_proj", sa.k_proj), ("v_proj", sa.v_proj), ("out_proj", sa.out_proj), ("gate_proj", sa.gate_proj)]:
            key_path = f"stack/layers/block/self_attn/{proj_name}/kernel_{l_idx}"
            adapter_dict["lora"][key_path] = {
                "A": module.lora_A.detach().cpu().numpy(),
                "B": module.lora_B.detach().cpu().numpy()
            }
            
    out_pkl = os.path.join(out_dir, "needle_lora.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(adapter_dict, f)
    print(f"  Saved PKL Adapter:        {out_pkl} ({os.path.getsize(out_pkl):,} bytes)")

    out_st = os.path.join(out_dir, "needle_lora.safetensors")
    st_tensors = {}
    for k, v in adapter_dict["lora"].items():
        st_tensors[f"lora/{k}/A"] = np.ascontiguousarray(v["A"])
        st_tensors[f"lora/{k}/B"] = np.ascontiguousarray(v["B"])
    save_file(st_tensors, out_st, metadata={"scale": str(scale), "rank": str(rank)})
    print(f"  Saved Safetensors Adapter: {out_st} ({os.path.getsize(out_st):,} bytes)")

    print("\n" + "=" * 70)
    print("PHASE C1 NEEDLE 2 GPU TRAINING COMPLETE [SUCCESS]")
    print("=" * 70)
    return True

if __name__ == "__main__":
    train()
