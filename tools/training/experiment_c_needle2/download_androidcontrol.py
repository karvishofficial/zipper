"""
Download and convert the full AndroidControl dataset from HuggingFace.
Outputs the data in Karvish TrainingExample JSONL format (same schema as train.jsonl).

This gives us ~15k training examples instead of our current 736.
The existing 77 validation examples are kept FROZEN and never mixed with training data.
"""

import json
import os
import sys
import hashlib

def load_androidcontrol():
    """Load AndroidControl from HuggingFace datasets library."""
    from datasets import load_dataset
    
    print("Downloading AndroidControl dataset from HuggingFace...")
    print("This may take a few minutes on first run (dataset is cached after).")
    
    # Try the community mirror first
    try:
        ds = load_dataset("leosltl/Android-Control")
        print(f"Loaded from leosltl/Android-Control")
        return ds
    except Exception as e:
        print(f"leosltl/Android-Control failed: {e}")
    
    # Try smolagents mirror
    try:
        ds = load_dataset("smolagents/android-control")
        print(f"Loaded from smolagents/android-control")
        return ds
    except Exception as e:
        print(f"smolagents/android-control failed: {e}")
    
    raise RuntimeError("Could not load AndroidControl from any HuggingFace mirror")


def inspect_dataset(ds):
    """Print dataset structure for understanding the schema."""
    print("\n=== Dataset Structure ===")
    if hasattr(ds, 'keys'):
        for split in ds.keys():
            print(f"\nSplit: {split}")
            print(f"  Num rows: {len(ds[split])}")
            print(f"  Columns: {ds[split].column_names}")
            if len(ds[split]) > 0:
                first = ds[split][0]
                print(f"  First record keys: {list(first.keys()) if isinstance(first, dict) else 'N/A'}")
                for k, v in first.items():
                    if isinstance(v, str):
                        print(f"    {k}: {v[:200]}...")
                    elif isinstance(v, (int, float, bool)):
                        print(f"    {k}: {v}")
                    elif isinstance(v, dict):
                        print(f"    {k}: dict with keys {list(v.keys())[:10]}")
                    elif isinstance(v, list):
                        print(f"    {k}: list of {len(v)} items")
                    elif isinstance(v, bytes):
                        print(f"    {k}: bytes ({len(v)} bytes)")
                    else:
                        print(f"    {k}: {type(v).__name__}")
    else:
        print(f"  Type: {type(ds)}")


def load_existing_val_hashes(val_path):
    """Load hashes of existing validation examples to avoid leaking them into training."""
    hashes = set()
    if not os.path.exists(val_path):
        return hashes
    with open(val_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                key = f"{rec.get('goal', '')}|{rec.get('packageName', '')}"
                hashes.add(hashlib.md5(key.encode()).hexdigest())
    return hashes


if __name__ == '__main__':
    OUT_DIR = 'tools/training/data'
    VAL_PATH = os.path.join(OUT_DIR, 'val_unseen.jsonl')
    
    # Step 1: Load existing validation hashes to prevent leakage
    val_hashes = load_existing_val_hashes(VAL_PATH)
    print(f"Loaded {len(val_hashes)} validation example hashes (will exclude from training)")
    
    # Step 2: Load dataset
    ds = load_androidcontrol()
    
    # Step 3: Inspect structure
    inspect_dataset(ds)
    
    print("\n=== Schema inspection complete. Check output above to finalize conversion. ===")
