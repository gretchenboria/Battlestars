#!/usr/bin/env python3
"""
Autotune fp8_quant kernel for all benchmark shapes on B200.
Run on the server: python autotune_fp8.py
"""

import os
import sys
import time
import json
import torch
import helion
import helion.language as hl

MAX_VAL = 448.0

# ---------------------------------------------------------------------------
# Kernel definition (no static_shapes so autotune can explore freely)
# ---------------------------------------------------------------------------

@helion.kernel()
def fp8_quant_kernel(
    data: torch.Tensor,        # [N, G] rows × group_size
    scales_out: torch.Tensor,  # [N] output scales (written in-place)
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


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def bench_config(cfg, data, scales, warmup=5, rep=20):
    """Return latency_ms for a given config."""
    # rebuild kernel with explicit config
    @helion.kernel(config=cfg)
    def _k(d: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        nrows = d.size(0)
        ncols = hl.specialize(d.size(1))
        out = torch.empty(nrows, ncols, dtype=torch.float32, device=d.device)
        for rr in hl.tile(nrows):
            row = d[rr, :].to(torch.float32)
            amax = torch.amax(torch.abs(row), -1).clamp(min=1e-10)
            scale = amax / MAX_VAL
            out[rr, :] = torch.clamp(row / scale[:, None], -MAX_VAL, MAX_VAL)
            s[rr] = scale
        return out

    try:
        for _ in range(warmup):
            _k(data, scales)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(rep):
            _k(data, scales)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / rep
    except Exception as e:
        return float("inf"), str(e)


# ---------------------------------------------------------------------------
# Shapes to autotune
# ---------------------------------------------------------------------------

SHAPES = [
    # (T, H, gsz)  →  N=T*(H//gsz), group_size=gsz
    (1,    256,   64),
    (4,    512,  128),
    (16,  1024,   64),
    (1,   4096,  128),
    (8,   4096,  128),
    (16,  4096,  128),
    (256, 4096,  128),
    (256, 8192,  128),
    (4096, 7168, 128),
]

# CompileIQ booster pack ACF files
BOOSTER_PACK = "/opt/booster_pack"
ACF_FILES = []
if os.path.isdir(BOOSTER_PACK):
    ACF_FILES = sorted(
        os.path.join(BOOSTER_PACK, f)
        for f in os.listdir(BOOSTER_PACK)
        if f.startswith("fp8_group_quant") and f.endswith(".acf")
    )
    print(f"Found {len(ACF_FILES)} CompileIQ ACF files: {[os.path.basename(f) for f in ACF_FILES]}")


# ---------------------------------------------------------------------------
# Main autotune loop
# ---------------------------------------------------------------------------

best_configs = {}

for T, H, gsz in SHAPES:
    N = T * (H // gsz)
    data = torch.randn(N, gsz, dtype=torch.float16, device="cuda")
    scales = torch.zeros(N, dtype=torch.float32, device="cuda")

    print(f"\n{'='*60}")
    print(f"Shape ({T}, {H}, {gsz})  →  input [{N}, {gsz}]  dtype=float16")
    bytes_rw = N * gsz * 2 + N * gsz * 4 + N * 4  # read f16 + write f32 + write f32 scales
    print(f"Est. memory traffic: {bytes_rw/1e6:.1f} MB")

    # 1. Helion built-in autotuner
    fp8_quant_kernel.reset()
    t0 = time.time()
    best_config = fp8_quant_kernel.autotune([data, scales], force=True)
    elapsed = time.time() - t0
    print(f"Helion autotune done in {elapsed:.1f}s")
    print(f"  Best config: {dict(best_config)}")

    # Benchmark the winning config
    ms = bench_config(best_config, data, scales)
    gbs = bytes_rw / (ms * 1e-3) / 1e9
    print(f"  Latency: {ms:.3f}ms  ({gbs:.0f} GB/s)")
    best_latency = ms
    best_acf_config = best_config

    # 2. Try CompileIQ ACF files on top of the best config
    for acf_path in ACF_FILES:
        acf_name = os.path.basename(acf_path)
        try:
            cfg_acf = helion.Config(**{**dict(best_config), "advanced_controls_file": acf_path})
            ms_acf = bench_config(cfg_acf, data, scales)
            gbs_acf = bytes_rw / (ms_acf * 1e-3) / 1e9
            marker = " *** NEW BEST ***" if ms_acf < best_latency else ""
            print(f"  + {acf_name}: {ms_acf:.3f}ms  ({gbs_acf:.0f} GB/s){marker}")
            if ms_acf < best_latency:
                best_latency = ms_acf
                best_acf_config = cfg_acf
        except Exception as e:
            print(f"  + {acf_name}: FAILED ({e})")

    best_configs[(T, H, gsz)] = best_acf_config
    print(f"→ Final best for {(T,H,gsz)}: {dict(best_acf_config)}  ({best_latency:.3f}ms)")


# ---------------------------------------------------------------------------
# Print final SHAPE_CONFIGS dict for copy-paste into submission.py
# ---------------------------------------------------------------------------

print("\n\n" + "="*60)
print("# Paste this into submission.py SHAPE_CONFIGS:")
print("="*60)
print("SHAPE_CONFIGS: dict[tuple, helion.Config] = {")
for shape, cfg in best_configs.items():
    d = {k: v for k, v in dict(cfg).items() if v is not None}
    args_str = ", ".join(f"{k}={v!r}" for k, v in d.items())
    print(f"    {shape}: helion.Config({args_str}),")
print("}")

# Also dump as JSON for easy parsing
output_path = "/home/ubuntu/best_configs.json"
serializable = {
    str(shape): {k: v for k, v in dict(cfg).items() if v is not None}
    for shape, cfg in best_configs.items()
}
with open(output_path, "w") as f:
    json.dump(serializable, f, indent=2)
print(f"\nSaved to {output_path}")
