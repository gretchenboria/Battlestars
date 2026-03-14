"""
Benchmark SwiGLU kernel vs baselines.

Used in Llama 2/3, Mistral, Mixtral, Gemma FFN layers.
intermediate_dim typically 2.67x hidden_dim (e.g. 14336 for Llama-3 70B).
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.swiglu import swiglu_forward
from autotuner.llm_tuner import LLMAutotuner


def swiglu_eager(x, gate):
    return x * gate * torch.sigmoid(gate)


def benchmark(fn, *args, warmup=10, rep=50):
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


def run_swiglu_benchmark(
    tokens: int = 4096,
    intermediate_dim: int = 14336,  # Llama-3 70B FFN dim
    dtype: torch.dtype = torch.float16,
    run_llm_tuner: bool = True,
    llm_trials: int = 15,
):
    device = torch.device("cuda")
    print(f"\nSwiGLU Benchmark")
    print(f"Shape: [{tokens}, {intermediate_dim}], dtype={dtype}")
    bytes_rw = tokens * intermediate_dim * torch.finfo(dtype).bits // 8 * 3  # 2 reads + 1 write
    print(f"Memory traffic: {bytes_rw / 1e9:.2f} GB\n")

    x = torch.randn(tokens, intermediate_dim, dtype=dtype, device=device)
    gate = torch.randn(tokens, intermediate_dim, dtype=dtype, device=device)

    results = {}

    # 1. Eager
    t = benchmark(swiglu_eager, x, gate)
    gbs = bytes_rw / (t * 1e-3) / 1e9
    results["eager"] = t
    print(f"Eager PyTorch:       {t:.3f}ms  ({gbs:.0f} GB/s)")

    # 2. torch.compile
    compiled = torch.compile(swiglu_eager)
    compiled(x, gate)
    t = benchmark(compiled, x, gate)
    gbs = bytes_rw / (t * 1e-3) / 1e9
    results["torch_compile"] = t
    print(f"torch.compile:       {t:.3f}ms  ({gbs:.0f} GB/s)")

    # 3. Helion default
    os.environ["HELION_AUTOTUNE_EFFORT"] = "quick"
    swiglu_forward(x, gate)
    t = benchmark(swiglu_forward, x, gate)
    gbs = bytes_rw / (t * 1e-3) / 1e9
    results["helion_default"] = t
    print(f"Helion (default):    {t:.3f}ms  ({gbs:.0f} GB/s)")

    # 4. LLM-guided
    if run_llm_tuner:
        tuner = LLMAutotuner(
            kernel_fn=swiglu_forward,
            sample_inputs=[x, gate],
            bytes_accessed_fn=lambda inputs: sum(
                t.numel() * t.element_size() for t in inputs
            ) + inputs[0].numel() * inputs[0].element_size(),
        )
        best = tuner.run(n_trials=llm_trials, verbose=True)
        results["llm_tuned"] = best.latency_ms
        gbs = bytes_rw / (best.latency_ms * 1e-3) / 1e9
        print(f"\nHelion + LLM:        {best.latency_ms:.3f}ms  ({gbs:.0f} GB/s)")

    print("\n" + "="*60)
    print("SUMMARY (speedup vs eager)")
    print("="*60)
    baseline = results["eager"]
    for name, t in results.items():
        print(f"  {name:<25} {t:.3f}ms  {baseline/t:.2f}x")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--intermediate-dim", type=int, default=14336)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--llm-trials", type=int, default=15)
    args = parser.parse_args()

    run_swiglu_benchmark(
        tokens=args.tokens,
        intermediate_dim=args.intermediate_dim,
        run_llm_tuner=not args.no_llm,
        llm_trials=args.llm_trials,
    )
