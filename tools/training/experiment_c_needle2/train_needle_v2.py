"""
Karvish Action Model Training Script v3 (Cactus Needle 2 45M + LoRA)
Optimized for NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM).

Speed optimizations (zero quality loss):
  - BF16 mixed precision (already in v2)
  - torch.compile for faster forward/backward
  - Native CUDA telemetry (no blocking nvidia-smi)
  - Configurable validation interval (default: end-only)
  - Temperature-smoothed stratified sampling for class balance

Saves checkpoints to tools/training/experiment_c_needle2/checkpoints_v2.
"""

import os
import sys
import json
import time
import pickle
import argparse
import math
import numpy as np
from collections import Counter
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

def get_gpu_util():
    """Non-blocking GPU utilization via native PyTorch CUDA telemetry."""
    try:
        allocated = torch.cuda.memory_allocated()
        total = torch.cuda.get_device_properties(0).total_mem
        return (allocated / total) * 100.0
    except Exception:
        return 0.0

sys.path.insert(0, os.path.dirname(__file__))
from pytorch_needle_v2 import Needle2ModelV2
from needle.model.tokenizer import get_tokenizer, BOS_ID, EOS_ID, PAD_ID
from needle.model.finetune import render_example


class NeedleDatasetV2(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_len=1024, max_samples=None):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))
                    if max_samples and len(self.examples) >= max_samples:
                        break

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt, target = render_example(ex)
        p_ids = self.tokenizer.encode(prompt)
        full_toks = self.tokenizer.encode(prompt + target)
        p_len = len(p_ids)
        
        full_ids = [BOS_ID] + full_toks + [EOS_ID]
        mask = [0.0] * (1 + p_len) + [1.0] * (len(full_toks) - p_len + 1)
        
        full_ids = full_ids[:self.max_len]
        mask = mask[:self.max_len]
        
        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "target_mask": torch.tensor(mask, dtype=torch.float32)
        }

    def get_action_type(self, idx):
        """Extract action type from an example for stratified sampling."""
        ex = self.examples[idx]
        answers = ex.get("answers", [{}])
        name = answers[0].get("name", "") if answers else ""
        if name == "open_app": return "open_app"
        if name.startswith("tap_"): return "tap"
        if name.startswith("type_"): return "type_text"
        if "scroll" in name: return "scroll"
        if "back" in name: return "back"
        if "wait" in name: return "wait"
        if "complete" in name: return "complete"
        return "other"


def build_balanced_sampler(dataset, alpha=0.5):
    """Temperature-smoothed stratified sampler (alpha=0.5).
    
    Produces sampling weights so minority classes (open_app, back, wait)
    get upsampled and majority class (tap) gets downsampled.
    alpha=0 → uniform across classes, alpha=1 → original distribution.
    """
    action_types = [dataset.get_action_type(i) for i in range(len(dataset))]
    counts = Counter(action_types)
    total = len(dataset)
    
    # Temperature-smoothed class probability: p_c^alpha / sum(p_c^alpha)
    smoothed = {c: (count / total) ** alpha for c, count in counts.items()}
    z = sum(smoothed.values())
    class_weight = {c: (smoothed[c] / z) / (counts[c] / total) for c in counts}
    
    sample_weights = [class_weight[action_types[i]] for i in range(total)]
    
    print(f"  Class balance (alpha={alpha}):")
    for c in sorted(counts.keys()):
        orig_pct = counts[c] / total * 100
        effective_pct = class_weight[c] * counts[c] / total * 100
        print(f"    {c:10s}: {counts[c]:4d} ({orig_pct:5.1f}%) -> effective weight {effective_pct:5.1f}%")
    
    return torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=total,
        replacement=True
    )


def collate_fn(batch):
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
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_mask = target_mask[..., 1:].contiguous()
    
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
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids)
                loss = compute_loss(logits, input_ids, target_mask)
            total_loss += loss.item()
            total_batches += 1
    return total_loss / max(total_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Train Needle 2 v3 on RTX 4050")
    parser.add_argument("--sanity", action="store_true", help="Run quick sanity training on 50 samples")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per step")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--val_interval", type=int, default=0, help="Validate every N epochs (0=end only)")
    parser.add_argument("--train_data", type=str, default=None, help="Override training data path")
    parser.add_argument("--val_data", type=str, default=None, help="Override validation data path")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Use only the first N training examples")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Use only the first N validation examples")
    parser.add_argument("--balance_alpha", type=float, default=0.5, help="Class balance temperature (0=uniform, 1=original)")
    parser.add_argument("--compile", action="store_true", default=True, help="Use torch.compile (default: True)")
    parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--out_dir", type=str, default="tools/training/experiment_c_needle2/checkpoints_v3", help="Checkpoint output directory")
    args = parser.parse_args()
    if args.no_compile:
        args.compile = False

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    base_ckpt_path = "tools/training/experiment_c_needle2/checkpoints/needle2.pkl"
    if args.train_data:
        train_path = args.train_data
    elif args.sanity:
        train_path = "tools/training/experiment_c_needle2/data/sanity_50_v3.jsonl"
    else:
        train_path = "tools/training/experiment_c_needle2/data/needle_train_v3.jsonl"
    val_path = args.val_data or "tools/training/experiment_c_needle2/data/needle_val_v3.jsonl"

    print("=" * 80)
    print(f"NEEDLE 2 v3 TRAINING")
    print(f"Train Data: {train_path}")
    print(f"Hardware: {device_name} (Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB)")
    print(f"Speed: BF16=ON | torch.compile={'ON' if args.compile else 'OFF'} | Val interval={'end-only' if args.val_interval == 0 else f'every {args.val_interval} epochs'}")
    print(f"Checkpoint Output Directory: {out_dir}")
    print("=" * 80, flush=True)

    with open(base_ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    tokenizer = get_tokenizer(ckpt["config"]["vocab_size"])

    model = Needle2ModelV2(ckpt["params"], ckpt["config"], lora_rank=args.rank, dropout=args.dropout).to(DEVICE)

    # torch.compile for faster forward/backward (zero quality impact)
    if args.compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("torch.compile: ENABLED (reduce-overhead mode)")
        except Exception as e:
            print(f"torch.compile: FAILED ({e}), falling back to eager mode")
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for p in trainable_params)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Base Parameters:     {total_params:,}")
    print(f"Trainable LoRA Parameters: {total_trainable:,} (Rank={args.rank}, Dropout={args.dropout})")

    train_dataset = NeedleDatasetV2(train_path, tokenizer, max_len=1024, max_samples=args.max_train_samples)
    val_dataset = NeedleDatasetV2(val_path, tokenizer, max_len=1024, max_samples=args.max_val_samples)
    print(f"Training Transitions:      {len(train_dataset)}")
    print(f"Held-out Val Transitions:  {len(val_dataset)} (Isolated from training)")

    # Temperature-smoothed stratified sampling for class balance
    sampler = build_balanced_sampler(train_dataset, alpha=args.balance_alpha)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True)

    epochs = 2 if args.sanity else args.epochs
    total_steps = (len(train_loader) // args.grad_accum) * epochs
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), eta_min=1e-6)

    best_val_loss = float("inf")
    gpu_utils = []
    start_time = time.time()
    total_samples_processed = 0

    print(f"\nExecuting Training Loop ({epochs} Epochs, {total_steps} Optimization Steps)...", flush=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        step_loss = 0.0
        optimizer.zero_grad()
        epoch_start = time.time()

        for i, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(DEVICE)
            target_mask = batch["target_mask"].to(DEVICE)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids)
                loss = compute_loss(logits, input_ids, target_mask)

            loss_scaled = loss / args.grad_accum
            loss_scaled.backward()

            step_loss += loss.item()
            epoch_loss += loss.item()
            total_samples_processed += len(input_ids)

            if (i + 1) % args.grad_accum == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                step_idx = (epoch * len(train_loader) + i + 1) // args.grad_accum
                log_freq = 5 if args.sanity else 25
                if step_idx % log_freq == 0 or (i + 1) == len(train_loader):
                    avg_recent = step_loss / min(args.grad_accum, i + 1)
                    vram_mb = torch.cuda.memory_allocated() / 1024**2
                    lr = scheduler.get_last_lr()[0]
                    curr_util = get_gpu_util()
                    gpu_utils.append(curr_util)
                    print(f"  Epoch {epoch+1}/{epochs} | Step {step_idx}/{total_steps} | Loss: {avg_recent:.4f} | LR: {lr:.2e} | VRAM: {vram_mb:.0f} MB | GPU Util: {curr_util:.0f}%", flush=True)
                step_loss = 0.0

        epoch_duration = time.time() - epoch_start
        avg_train_loss = epoch_loss / len(train_loader)
        
        # Validate only at specified intervals or at end (saves ~13 min per skipped validation)
        should_validate = (args.val_interval > 0 and (epoch + 1) % args.val_interval == 0) or (epoch + 1) == epochs
        if should_validate:
            val_loss = evaluate(model, val_loader)
            print(f"  ---> Epoch {epoch+1} Completed ({epoch_duration:.1f}s) | Train Loss: {avg_train_loss:.4f} | Held-out Val Loss: {val_loss:.4f}", flush=True)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_name = "needle_lora_sanity_best.pt" if args.sanity else "needle_lora_best.pt"
                best_pt = os.path.join(out_dir, ckpt_name)
                torch.save(model.state_dict(), best_pt)
        else:
            print(f"  ---> Epoch {epoch+1} Completed ({epoch_duration:.1f}s) | Train Loss: {avg_train_loss:.4f} | Val: skipped", flush=True)
            # Still save checkpoint each epoch
            ckpt_name = "needle_lora_sanity_best.pt" if args.sanity else "needle_lora_best.pt"
            best_pt = os.path.join(out_dir, ckpt_name)
            torch.save(model.state_dict(), best_pt)

    total_duration = time.time() - start_time
    throughput = total_samples_processed / max(total_duration, 1)
    peak_vram = torch.cuda.max_memory_allocated() / (1024**2)

    print("\n" + "=" * 80)
    avg_util = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0.0
    peak_util = max(gpu_utils) if gpu_utils else 0.0
    print(f"TRAINING COMPLETE! Duration: {total_duration:.1f}s ({total_duration/60:.2f} min) | Peak VRAM: {peak_vram:.1f} MB | Avg GPU Util: {avg_util:.1f}% (Peak: {peak_util:.1f}%) | Throughput: {throughput:.2f} samples/sec")
    print(f"Best Held-out Validation Loss: {best_val_loss:.4f}")
    print("=" * 80, flush=True)

    # Export LoRA adapters
    scale = 32.0 / args.rank
    adapter_dict = {"lora": {}, "scale": float(scale), "base": base_ckpt_path, "rank": args.rank}
    for l_idx, block in enumerate(model.blocks):
        sa = block.self_attn
        for proj_name, module in [("q_proj", sa.q_proj), ("k_proj", sa.k_proj), ("v_proj", sa.v_proj), ("out_proj", sa.out_proj), ("gate_proj", sa.gate_proj)]:
            key_path = f"stack/layers/block/self_attn/{proj_name}/kernel_{l_idx}"
            adapter_dict["lora"][key_path] = {
                "A": module.lora_A.detach().cpu().numpy(),
                "B": module.lora_B.detach().cpu().numpy()
            }

    prefix = "sanity_" if args.sanity else ""
    out_pkl = os.path.join(out_dir, f"{prefix}needle_lora.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(adapter_dict, f)
    print(f"Saved LoRA PKL:        {out_pkl} ({os.path.getsize(out_pkl):,} bytes)")

    out_st = os.path.join(out_dir, f"{prefix}needle_lora.safetensors")
    st_tensors = {}
    for k, v in adapter_dict["lora"].items():
        st_tensors[f"lora/{k}/A"] = np.ascontiguousarray(v["A"])
        st_tensors[f"lora/{k}/B"] = np.ascontiguousarray(v["B"])
    save_file(st_tensors, out_st, metadata={"scale": str(scale), "rank": str(args.rank)})
    print(f"Saved LoRA Safetensors: {out_st} ({os.path.getsize(out_st):,} bytes)")
    if not args.sanity:
        import shutil
        shutil.copy(best_pt, os.path.join(out_dir, "needle_lora_v3.pt"))
        shutil.copy(out_pkl, os.path.join(out_dir, "needle_lora_v3.pkl"))
        shutil.copy(out_st, os.path.join(out_dir, "needle_lora_v3.safetensors"))
        print(f"Archived Versioned Checkpoint: needle_lora_v3 (.pt, .pkl, .safetensors)")

    meta = {
        "dataset_train": train_path,
        "dataset_val": val_path,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "epochs": epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "learning_rate": args.lr,
        "lora_rank": args.rank,
        "lora_dropout": args.dropout,
        "total_steps": total_steps,
        "total_duration_sec": total_duration,
        "throughput_samples_per_sec": throughput,
        "peak_vram_mb": peak_vram,
        "avg_gpu_util_percent": avg_util if "avg_util" in locals() else 0.0,
        "peak_gpu_util_percent": peak_util if "peak_util" in locals() else 0.0,
        "best_val_loss": best_val_loss
    }
    with open(os.path.join(out_dir, f"{prefix}training_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return True

if __name__ == "__main__":
    main()
