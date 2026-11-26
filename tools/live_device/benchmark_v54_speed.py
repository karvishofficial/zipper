"""
KARVISH V54 — SPEED & LATENCY BENCHMARK SUITE
Benchmarks:
1. Neural Model Inference Speed (ONNX FP32 & INT8 on CPU with 4 threads matching mobile CPU)
2. Latency percentiles (Average, p50, p90, p95, Min, Max)
3. Throughput (decisions / sec)
"""

import time
import json
import numpy as np
from pathlib import Path
import onnxruntime as ort

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
EXPORT_DIR = REPO_ROOT / "tools" / "training" / "karvish_mobile_action" / "v54" / "export"
ASSETS_MODEL = REPO_ROOT / "app" / "src" / "main" / "assets" / "models" / "karvish_v54.onnx"

def benchmark_onnx_v54(model_path: Path, num_trials: int = 50):
    if not model_path.exists():
        print(f"Skipping {model_path.name} (not found at {model_path})")
        return None

    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"\n>> Benchmarking {model_path.name} ({size_mb:.2f} MB)...")
    
    # Configure CPU options (similar to mobile 4-core allocation)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    
    t0 = time.perf_counter()
    session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
    cold_init_ms = (time.perf_counter() - t0) * 1000.0
    print(f"   Cold initialization: {cold_init_ms:.2f} ms")

    B = 1
    L = 16  # Typical goal length ("Open WhatsApp and send hi to Rekha")
    N = 20  # Typical UI screen candidates on mobile
    max_lbl = 16

    inputs = {
        "goal_ids": np.random.randint(1, 1000, (B, L), dtype=np.int64),
        "goal_mask": np.ones((B, L), dtype=np.float32),
        "candidate_label_ids": np.random.randint(1, 1000, (B, N, max_lbl), dtype=np.int64),
        "candidate_label_mask": np.ones((B, N, max_lbl), dtype=np.float32),
        "candidate_bounds": np.random.rand(B, N, 10).astype(np.float32),
        "candidate_roles": np.random.randint(0, 18, (B, N), dtype=np.int64),
        "candidate_flags": np.zeros((B, N, 16), dtype=np.float32),
        "candidate_pad_mask": np.ones((B, N), dtype=np.float32),
        "candidate_num_features": np.zeros((B, N, 4), dtype=np.float32),
    }

    # Warmup
    for _ in range(5):
        session.run(None, inputs)

    # Benchmark trials
    latencies = []
    for _ in range(num_trials):
        t_start = time.perf_counter()
        session.run(None, inputs)
        lat_ms = (time.perf_counter() - t_start) * 1000.0
        latencies.append(lat_ms)

    latencies.sort()
    avg_ms = float(np.mean(latencies))
    p50_ms = float(np.percentile(latencies, 50))
    p90_ms = float(np.percentile(latencies, 90))
    p95_ms = float(np.percentile(latencies, 95))
    min_ms = float(min(latencies))
    max_ms = float(max(latencies))
    fps = 1000.0 / avg_ms

    print(f"   Results over {num_trials} decisions:")
    print(f"   - Average Latency: {avg_ms:.2f} ms ({fps:.1f} decisions/sec)")
    print(f"   - Median (p50):    {p50_ms:.2f} ms")
    print(f"   - 90th percentile: {p90_ms:.2f} ms")
    print(f"   - 95th percentile: {p95_ms:.2f} ms")
    print(f"   - Min / Max:       {min_ms:.2f} ms / {max_ms:.2f} ms")

    return {
        "file": model_path.name,
        "size_mb": round(size_mb, 2),
        "cold_init_ms": round(cold_init_ms, 2),
        "avg_ms": round(avg_ms, 2),
        "p50_ms": round(p50_ms, 2),
        "p90_ms": round(p90_ms, 2),
        "p95_ms": round(p95_ms, 2),
        "min_ms": round(min_ms, 2),
        "max_ms": round(max_ms, 2),
        "throughput_decisions_per_sec": round(fps, 1),
    }

def main():
    print("=" * 70)
    print("KARVISH V54 — SPEED & LATENCY BENCHMARK SUITE")
    print("=" * 70)

    fp32_model = EXPORT_DIR / "karvish_v54.onnx"
    int8_model = EXPORT_DIR / "karvish_v54_int8.onnx"

    results = {}
    if fp32_model.exists():
        results["fp32"] = benchmark_onnx_v54(fp32_model)
    if int8_model.exists():
        results["int8"] = benchmark_onnx_v54(int8_model)
    if ASSETS_MODEL.exists():
        results["deployed_asset"] = benchmark_onnx_v54(ASSETS_MODEL)

    out_file = SCRIPT_DIR / "v54_speed_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[BENCHMARK SAVED] -> {out_file}")

if __name__ == "__main__":
    main()
