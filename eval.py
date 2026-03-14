#!/usr/bin/env python3
"""
Local evaluation + ncu profiling harness for fp8_quant.

Usage:
  # Correctness check + benchmark:
  python eval.py

  # Profile with Nsight Compute (ncu):
  ncu --target-processes all \
      --capture-range cudaProfilerApi \
      --capture-range-end stop \
      --set full \
      -o profile.ncu-rep \
      python eval.py --profile

  # View profile:
  ncu-ui profile.ncu-rep
"""

import argparse
import os
import sys
import torch
import helion
import helion.language as hl

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="fp8_quant eval & profiling harness")
parser.add_argument("--profile", action="store_true",
                    help="Wrap kernels with cudaProfilerStart/Stop for ncu capture")
parser.add_argument("--shape", type=str, default="all",
                    help="Shape key to run, e.g. '4096,7168,128', or 'all'")
parser.add_argument("--warmup", type=int, default=10)
parser.add_argument("--rep", type=int, default=50)
parser.add_argument("--dtype", type=str, default="float16",
                    choices=["float16", "bfloat16"])
args = parser.parse_args()

dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

# ---------------------------------------------------------------------------
# Mock task module (not available locally)
# ---------------------------------------------------------------------------
input_t  = tuple   # (x: Tensor, x_q: Tensor, x_s: Tensor)
output_t = tuple   # (x_q: Tensor, x_s: Tensor)

# ---------------------------------------------------------------------------
# Kernel (same as submission.py but no static_shapes for flexibility)
# ---------------------------------------------------------------------------

MAX_VAL = 448.0

@helion.kernel()
def fp8_quant_kernel(
    data: torch.Tensor,
    scales_out: torch.Tensor,
) -> torch.Tensor:
    nrows = data.size(0)
    ncols = hl.specialize(data.size(1))
    qout = torch.empty(nrows, ncols, dtype=torch.float32, device=data.device)
    for rr in hl.tile(nrows):
        row = data[rr, :].to(torch.float32)
        amax = torch.amax(torch.abs(row), -1).clamp(min=1e-10)
        scale = amax / MAX_VAL
        qout[rr, :] = torch.clamp(row / scale[:, None], -MAX_VAL, MAX_VAL)
        scales_out[rr] = scale
    return qout


# Reference implementation (eager PyTorch, no Helion)
def fp8_quant_reference(data: torch.Tensor, scales_out: torch.Tensor) -> torch.Tensor:
    x = data.to(torch.float32)
    amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    scale = (amax / MAX_VAL).squeeze(-1)
    qout = (x / amax * MAX_VAL).clamp(-MAX_VAL, MAX_VAL)
    scales_out.copy_(scale)
    return qout


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------

def benchmark(fn, *args, warmup=10, rep=50):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rep


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------

def check_correctness(T, H, gsz, verbose=True):
    N = T * (H // gsz)
    x = torch.randn(N, gsz, dtype=dtype, device="cuda")
    s_ref = torch.zeros(N, dtype=torch.float32, device="cuda")
    s_hel = torch.zeros(N, dtype=torch.float32, device="cuda")

    q_ref = fp8_quant_reference(x, s_ref)
    fp8_quant_kernel.reset()
    q_hel = fp8_quant_kernel(x, s_hel)

    q_ok = torch.allclose(q_ref, q_hel, atol=1e-2, rtol=1e-2)
    s_ok = torch.allclose(s_ref, s_hel, atol=1e-5, rtol=1e-5)
    status = "PASS" if (q_ok and s_ok) else "FAIL"
    if verbose:
        print(f"  Correctness ({T},{H},{gsz}): {status}  "
              f"[q_max_err={( q_ref - q_hel ).abs().max():.4f}, "
              f"s_max_err={(s_ref - s_hel).abs().max():.6f}]")
    return q_ok and s_ok


# ---------------------------------------------------------------------------
# All shapes
# ---------------------------------------------------------------------------

ALL_SHAPES = [
    # (T,    H,    gsz)
    (1,    256,    64),
    (4,    512,   128),
    (16,  1024,    64),
    (1,   4096,   128),
    (8,   4096,   128),
    (16,  4096,   128),
    (256, 4096,   128),
    (256, 8192,   128),
    (4096, 7168,  128),
]

if args.shape != "all":
    parts = [int(p) for p in args.shape.split(",")]
    ALL_SHAPES = [tuple(parts)]

device = torch.device("cuda")

# ---------------------------------------------------------------------------
# Correctness sweep
# ---------------------------------------------------------------------------

print(f"\nfp8_quant eval harness  (dtype={dtype}, device=cuda)")
print("="*60)
print("Correctness checks:")
for shape in ALL_SHAPES:
    check_correctness(*shape)

# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

print("\nBenchmark (warmup={}, rep={}):".format(args.warmup, args.rep))
print(f"{'Shape':>25}  {'N×G':>12}  {'Ref ms':>8}  {'Helion ms':>10}  {'GB/s':>7}  {'Speedup':>7}")

results = {}
for T, H, gsz in ALL_SHAPES:
    N = T * (H // gsz)
    x = torch.randn(N, gsz, dtype=dtype, device=device)
    s = torch.zeros(N, dtype=torch.float32, device=device)

    bytes_rw = N * gsz * torch.finfo(dtype).bits // 8  \
             + N * gsz * 4  \
             + N * 4

    fp8_quant_kernel.reset()
    t_ref = benchmark(fp8_quant_reference, x, s, warmup=args.warmup, rep=args.rep)
    t_hel = benchmark(fp8_quant_kernel,    x, s, warmup=args.warmup, rep=args.rep)
    gbs   = bytes_rw / (t_hel * 1e-3) / 1e9

    results[(T, H, gsz)] = dict(ref_ms=t_ref, helion_ms=t_hel, gbs=gbs)
    print(f"  ({T:5},{H:5},{gsz:4})  {N*gsz:>12,}  {t_ref:8.3f}  {t_hel:10.3f}  {gbs:7.0f}  {t_ref/t_hel:6.2f}x")

# ---------------------------------------------------------------------------
# NCU profile mode
# ---------------------------------------------------------------------------

if args.profile:
    print("\n" + "="*60)
    print("Profiling mode: capturing with cudaProfilerStart/Stop")
    print("Run this script with:")
    print("  ncu --target-processes all \\")
    print("      --capture-range cudaProfilerApi \\")
    print("      --capture-range-end stop \\")
    print("      --set full \\")
    print("      -o profile.ncu-rep \\")
    print("      python eval.py --profile")
    print("="*60 + "\n")

    # Warm up outside the capture range
    for T, H, gsz in ALL_SHAPES:
        N = T * (H // gsz)
        x = torch.randn(N, gsz, dtype=dtype, device=device)
        s = torch.zeros(N, dtype=torch.float32, device=device)
        for _ in range(5):
            fp8_quant_kernel(x, s)
            fp8_quant_reference(x, s)
    torch.cuda.synchronize()

    # Start profiler capture
    torch.cuda.cudart().cudaProfilerStart()

    for T, H, gsz in ALL_SHAPES:
        N = T * (H // gsz)
        x = torch.randn(N, gsz, dtype=dtype, device=device)
        s = torch.zeros(N, dtype=torch.float32, device=device)

        range_name = f"fp8_quant_helion_{T}x{H}x{gsz}"
        torch.cuda.nvtx.range_push(range_name)
        for _ in range(3):
            fp8_quant_kernel(x, s)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push(f"fp8_quant_eager_{T}x{H}x{gsz}")
        for _ in range(3):
            fp8_quant_reference(x, s)
        torch.cuda.nvtx.range_pop()

    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print("Profile captured. Open with: ncu-ui profile.ncu-rep")
