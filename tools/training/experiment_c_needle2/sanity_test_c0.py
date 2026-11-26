"""
Phase C0: Needle 2 LoRA Pipeline Sanity Check
Runs on a small 50-example subset of AndroidControl to verify:
- Base checkpoint loading
- LoRA parameter attachment
- Forward/backward gradients & loss decrease
- Absence of NaN/Inf values
- Precision & optimizer convergence
- Adapter saving and reloading
"""

import sys
import os
import json
import pickle
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import jax
import jax.numpy as jnp
import optax
from safetensors.numpy import save_file, load_file

from needle.model.architecture import SimpleAttentionNetwork, TransformerConfig
from needle.model.tokenizer import get_tokenizer
from needle.model.finetune import (
    lora_target_paths, init_lora, merge_lora,
    fit_max_len, load_jsonl
)

def run_sanity_check():
    print("=" * 60)
    print("PHASE C0: NEEDLE 2 LoRA SANITY TEST (50 EXAMPLES)")
    print("=" * 60)

    ckpt_path = "tools/training/experiment_c_needle2/checkpoints/needle2.pkl"
    sanity_data_path = "tools/training/experiment_c_needle2/data/sanity_50.jsonl"
    adapter_out_pkl = "tools/training/experiment_c_needle2/checkpoints/needle_lora_sanity.pkl"
    adapter_out_safetensors = "tools/training/experiment_c_needle2/checkpoints/needle_lora_sanity.safetensors"

    # 1. Base Checkpoint Loading
    print("\n[Step 1/6] Loading base Needle 2 checkpoint...")
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    print(f"  Loaded from: {ckpt_path}")
    raw_cfg = ckpt["config"]
    print(f"  Config: {raw_cfg['num_layers']} layers, {raw_cfg['d_model']} d_model, {raw_cfg['vocab_size']} vocab")
    
    cfg_dict = dict(raw_cfg)
    cfg_dict["dtype"] = "float32"
    config = TransformerConfig(**cfg_dict)
    config.flash = False
    config.remat = False

    model = SimpleAttentionNetwork(config)
    params = jax.tree.map(lambda a: jnp.asarray(a).astype(jnp.float32), ckpt["params"])
    total_base_params = sum(p.size for p in jax.tree.leaves(params))
    print(f"  Total base parameters: {total_base_params:,} (~{total_base_params*4/1024/1024:.1f} MB float32)")

    # 2. LoRA Parameter Attachment
    print("\n[Step 2/6] Attaching LoRA adapters (Rank 16, Alpha 32)...")
    paths = lora_target_paths(params)
    print(f"  Target module groups: {len(paths)}")
    for p in paths:
        print(f"    - {'/'.join(p)}")
    
    rank = 16
    alpha = 32.0
    scale = alpha / rank
    seed = 42
    lora = init_lora(params, paths, rank=rank, key=jax.random.PRNGKey(seed))
    total_lora_params = sum(v["A"].size + v["B"].size for v in lora.values())
    print(f"  Trainable LoRA parameters: {total_lora_params:,} ({total_lora_params*4/1024/1024:.2f} MB float32)")
    print(f"  Trainable parameter ratio: {(total_lora_params / total_base_params) * 100:.2f}%")

    # 3. Data Loading & Tokenization
    print("\n[Step 3/6] Loading and encoding sanity dataset (50 examples)...")
    tokenizer = get_tokenizer(config.vocab_size)
    max_len = fit_max_len(sanity_data_path, tokenizer, cap=1024)
    seqs, masks = load_jsonl(sanity_data_path, tokenizer, max_len)
    print(f"  Examples: {len(seqs)}, Max sequence length: {max_len}")
    assert len(seqs) == 50, f"Expected 50 examples, got {len(seqs)}"

    # 4. Training Loop (2 Epochs)
    print("\n[Step 4/6] Executing forward/backward passes & optimizer updates...")
    batch_size = 4
    count = len(seqs)
    steps_per_epoch = -(-count // batch_size)
    epochs = 2
    total_steps = epochs * steps_per_epoch

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=1e-4, warmup_steps=2, decay_steps=total_steps
    )
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule))
    opt_state = optimizer.init(lora)

    def loss_fn(l_weights, ids, mask):
        merged = merge_lora(params, l_weights, scale)
        logits = model.apply({"params": merged}, ids, quant=False)
        logits, targets, mask_slice = logits[:, :-1], ids[:, 1:], mask[:, 1:]
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        return (ce * mask_slice).sum() / jnp.maximum(mask_slice.sum(), 1.0)

    @jax.jit
    def train_step(l_weights, state, batch_ids, batch_mask):
        loss, grads = jax.value_and_grad(loss_fn)(l_weights, batch_ids, batch_mask)
        updates, new_state = optimizer.update(grads, state, l_weights)
        new_lora = optax.apply_updates(l_weights, updates)
        return new_lora, new_state, loss, grads

    step_losses = []
    has_nan_or_inf = False
    start_time = time.time()

    step_idx = 0
    for epoch in range(epochs):
        epoch_losses = []
        for start in range(0, count, batch_size):
            b_ids = jnp.asarray(seqs[start:start + batch_size])
            b_mask = jnp.asarray(masks[start:start + batch_size])
            
            lora, opt_state, loss_val, grads = train_step(lora, opt_state, b_ids, b_mask)
            l_float = float(loss_val)
            epoch_losses.append(l_float)
            step_losses.append(l_float)

            # Check for NaN / Inf
            if np.isnan(l_float) or np.isinf(l_float):
                has_nan_or_inf = True
                print(f"  [ERROR] NaN/Inf detected at step {step_idx}!")
                break

            for g_dict in grads.values():
                for g_tensor in g_dict.values():
                    if jnp.isnan(g_tensor).any() or jnp.isinf(g_tensor).any():
                        has_nan_or_inf = True
                        print(f"  [ERROR] NaN/Inf in gradients at step {step_idx}!")
                        break

            step_idx += 1
            if step_idx % 5 == 0 or step_idx == total_steps:
                print(f"  Step {step_idx:2d}/{total_steps} | Loss: {l_float:.4f}")

        avg_epoch_loss = np.mean(epoch_losses)
        print(f"  Epoch {epoch + 1}/{epochs} Complete | Avg Loss: {avg_epoch_loss:.4f}")

    train_duration = time.time() - start_time
    print(f"  Sanity training duration: {train_duration:.2f}s ({train_duration/total_steps:.2f}s/step)")

    # 5. Check Loss Progression
    initial_loss = step_losses[0]
    final_loss = step_losses[-1]
    loss_decreased = final_loss < initial_loss
    print(f"\n[Step 5/6] Loss Verification:")
    print(f"  Initial Step Loss: {initial_loss:.4f}")
    print(f"  Final Step Loss:   {final_loss:.4f}")
    print(f"  Loss Reduction:    {initial_loss - final_loss:.4f}")
    print(f"  Loss Decreased:    {loss_decreased}")
    print(f"  Zero NaN/Inf:      {not has_nan_or_inf}")

    # 6. Save & Reload Checkpoint Verification
    print("\n[Step 6/6] Saving & Reloading Adapter Checkpoints...")
    adapter_dict = {
        "lora": {"/".join(p): {"A": np.asarray(v["A"]), "B": np.asarray(v["B"])}
                 for p, v in lora.items()},
        "scale": float(scale),
        "base": ckpt_path,
        "rank": rank,
        "seed": seed,
    }
    with open(adapter_out_pkl, "wb") as f:
        pickle.dump(adapter_dict, f)
    print(f"  Saved PKL adapter: {adapter_out_pkl} ({os.path.getsize(adapter_out_pkl):,} bytes)")

    # Also save safetensors
    st_tensors = {}
    for name, v in adapter_dict["lora"].items():
        st_tensors[f"lora/{name}/A"] = np.ascontiguousarray(v["A"])
        st_tensors[f"lora/{name}/B"] = np.ascontiguousarray(v["B"])
    save_file(st_tensors, adapter_out_safetensors, metadata={"scale": str(scale), "rank": str(rank)})
    print(f"  Saved Safetensors adapter: {adapter_out_safetensors} ({os.path.getsize(adapter_out_safetensors):,} bytes)")

    # Reload and test merge
    with open(adapter_out_pkl, "rb") as f:
        reloaded = pickle.load(f)
    assert len(reloaded["lora"]) == len(paths), "Reloaded adapter tensor count mismatch"
    reloaded_lora = {tuple(k.split("/")): {"A": jnp.asarray(v["A"]), "B": jnp.asarray(v["B"])}
                     for k, v in reloaded["lora"].items()}
    merged_params = merge_lora(params, reloaded_lora, reloaded["scale"])
    
    # Run forward pass on merged model
    test_ids = jnp.asarray(seqs[:1])
    test_mask = jnp.asarray(masks[:1])
    test_logits = model.apply({"params": merged_params}, test_ids, quant=False)
    test_ce = optax.softmax_cross_entropy_with_integer_labels(test_logits[:, :-1], test_ids[:, 1:])
    test_loss = float((test_ce * test_mask[:, 1:]).sum() / jnp.maximum(test_mask[:, 1:].sum(), 1.0))
    print(f"  Reloaded adapter evaluation loss: {test_loss:.4f} (Valid float, No NaN: {not np.isnan(test_loss)})")

    all_passed = loss_decreased and (not has_nan_or_inf) and (not np.isnan(test_loss))
    print("\n" + "=" * 60)
    print(f"PHASE C0 RESULT: {'PASSED [OK]' if all_passed else 'FAILED [ERROR]'}")
    print("=" * 60)
    return all_passed

if __name__ == "__main__":
    success = run_sanity_check()
    sys.exit(0 if success else 1)
