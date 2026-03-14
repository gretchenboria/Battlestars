"""
Benchmark RoPE kernel vs baselines.

Baselines:
  1. Eager PyTorch
  2. torch.compile
  3. Helion (default autotuning)
  4. Helion + LLM autotuner (our contribution)
  5. Helion + CompileIQ booster pack (if available)
"""

import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.rope import rope_forward, rope_backward, precompute_freqs
from autotuner.llm_tuner import LLMAutotuner


# ---------------------------------------------------------------------------
# Baseline: Eager PyTorch
# ---------------------------------------------------------------------------

def rope_eager(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Standard eager implementation used in Llama/HuggingFace."""
    batch, seq_len, num_heads, head_dim = x.shape
    half = head_dim // 2
    cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq, 1, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(2)
    x1 = x[..., :half]
    x2 = x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


def rope_compile(x, cos, sin):
    return rope_eager(x, cos, sin)


# ---------------------------------------------------------------------------
# Timing utility
# ---------------------------------------------------------------------------

def benchmark(fn, *args, warmup=10, rep=50):
    """Returns median latency in ms."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rep


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_rope_benchmark(
    batch: int = 8,
    seq_len: int = 4096,
    num_heads: int = 32,
    head_dim: int = 128,
    dtype: torch.dtype = torch.float16,
    run_llm_tuner: bool = True,
    llm_trials: int = 15,
    use_compileiq: bool = True,
):
    device = torch.device("cuda")
    print(f"\nRoPE Benchmark")
    print(f"Shape: [{batch}, {seq_len}, {num_heads}, {head_dim}], dtype={dtype}")
    bytes_rw = (
        batch * seq_len * num_heads * head_dim * torch.finfo(dtype).bits // 8 * 2  # read + write x
        + seq_len * head_dim * 4 * 2  # cos + sin (float32)
    )
    print(f"Memory traffic: {bytes_rw / 1e9:.2f} GB\n")

    # Create inputs
    x = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device)
    cos, sin = precompute_freqs(head_dim, seq_len, device=device)
    cos = cos.to(dtype)
    sin = sin.to(dtype)

    results = {}

    # 1. Eager
    t = benchmark(rope_eager, x, cos, sin)
    gbs = bytes_rw / (t * 1e-3) / 1e9
    results["eager"] = t
    print(f"Eager PyTorch:       {t:.3f}ms  ({gbs:.0f} GB/s)")

    # 2. torch.compile
    compiled = torch.compile(rope_compile)
    compiled(x, cos, sin)  # trigger compilation
    t = benchmark(compiled, x, cos, sin)
    gbs = bytes_rw / (t * 1e-3) / 1e9
    results["torch_compile"] = t
    print(f"torch.compile:       {t:.3f}ms  ({gbs:.0f} GB/s)")

    # 3. Helion default (quick autotune)
    os.environ["HELION_AUTOTUNE_EFFORT"] = "quick"
    rope_forward(x, cos, sin)  # trigger autotune
    t = benchmark(rope_forward, x, cos, sin)
    gbs = bytes_rw / (t * 1e-3) / 1e9
    results["helion_default"] = t
    print(f"Helion (default):    {t:.3f}ms  ({gbs:.0f} GB/s)")

    # 4. Helion + CompileIQ
    if use_compileiq:
        booster_path = "/opt/booster_pack"
        if os.path.exists(booster_path):
            acf_files = [f for f in os.listdir(booster_path) if f.endswith(".acf")]
            if acf_files:
                import helion
                best_ciq_t = float("inf")
                best_acf = None
                for acf in acf_files[:5]:  # try a few
                    try:
                        cfg = helion.Config(advanced_controls_file=os.path.join(booster_path, acf))
                        t_ciq = benchmark(lambda: rope_forward(x, cos, sin, config=cfg))
                        if t_ciq < best_ciq_t:
                            best_ciq_t = t_ciq
                            best_acf = acf
                    except Exception:
                        pass
                if best_acf:
                    gbs = bytes_rw / (best_ciq_t * 1e-3) / 1e9
                    results["helion_compileiq"] = best_ciq_t
                    print(f"Helion + CompileIQ:  {best_ciq_t:.3f}ms  ({gbs:.0f} GB/s)  [{best_acf}]")

    # 5. LLM-guided autotuner
    if run_llm_tuner:
        def bytes_accessed(inputs):
            x_in, cos_in, sin_in = inputs
            return (
                x_in.numel() * x_in.element_size() * 2  # read + write
                + cos_in.numel() * cos_in.element_size()
                + sin_in.numel() * sin_in.element_size()
            )

        tuner = LLMAutotuner(
            kernel_fn=rope_forward,
            sample_inputs=[x, cos, sin],
            bytes_accessed_fn=bytes_accessed,
        )
        best = tuner.run(n_trials=llm_trials, verbose=True)
        results["llm_tuned"] = best.latency_ms
        gbs = bytes_rw / (best.latency_ms * 1e-3) / 1e9
        print(f"\nHelion + LLM:        {best.latency_ms:.3f}ms  ({gbs:.0f} GB/s)")

    # Summary table
    print("\n" + "="*60)
    print("SUMMARY (speedup vs eager)")
    print("="*60)
    baseline = results["eager"]
    for name, t in results.items():
        speedup = baseline / t
        print(f"  {name:<25} {t:.3f}ms  {speedup:.2f}x")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--llm-trials", type=int, default=15)
    parser.add_argument("--no-compileiq", action="store_true")
    args = parser.parse_args()

    run_rope_benchmark(
        batch=args.batch,
        seq_len=args.seq_len,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        run_llm_tuner=not args.no_llm,
        llm_trials=args.llm_trials,
        use_compileiq=not args.no_compileiq,
    )
