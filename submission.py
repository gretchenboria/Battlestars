#!POPCORN leaderboard fp8_quant
#!POPCORN gpu B200_Nebius

from task import input_t, output_t

import torch
import helion
import helion.language as hl

# ---------------------------------------------------------------------------
# Per-shape configs — hardcoded, fast-compiling, NO runtime autotuning.
# Selected for minimal JIT compile time while maintaining good throughput.
# (Complex ACF/TMA configs cause 12min leaderboard timeouts.)
# ---------------------------------------------------------------------------

SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # (num_tokens, hidden_dim, group_size)
    # Test shapes
    (1,    256,   64): helion.Config(block_sizes=[16],  num_warps=4,  num_stages=2),
    (4,    512,  128): helion.Config(block_sizes=[32],  num_warps=4,  num_stages=2),
    (16,  1024,   64): helion.Config(block_sizes=[32],  num_warps=4,  num_stages=2),
    (1,   4096,  128): helion.Config(block_sizes=[32],  num_warps=4,  num_stages=2),
    (8,   4096,  128): helion.Config(block_sizes=[64],  num_warps=8,  num_stages=2),
    # Benchmark shapes
    (16,  4096,  128): helion.Config(block_sizes=[64],  num_warps=8,  num_stages=2),
    (256, 4096,  128): helion.Config(block_sizes=[128], num_warps=16, num_stages=2),
    (256, 8192,  128): helion.Config(block_sizes=[128], num_warps=16, num_stages=2),
    (4096, 7168, 128): helion.Config(block_sizes=[128], num_warps=16, num_stages=2),
}

# ---------------------------------------------------------------------------
# Kernel definition
# ---------------------------------------------------------------------------

def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, config=config)
    def kernel(
        data: torch.Tensor,        # [N, G]  float16 rows
        scales_out: torch.Tensor,  # [N]     float32 scales (written in-place)
    ) -> torch.Tensor:
        nrows = data.size(0)
        ncols = hl.specialize(data.size(1))
        MAX_VAL = 448.0

        qout = torch.empty(nrows, ncols, dtype=torch.float32, device=data.device)

        for rr in hl.tile(nrows):
            row   = data[rr, :].to(torch.float32)
            amax  = torch.amax(torch.abs(row), -1).clamp(min=1e-10)
            scale = amax / MAX_VAL
            qout[rr, :]   = torch.clamp(row / scale[:, None], -MAX_VAL, MAX_VAL)
            scales_out[rr] = scale

        return qout

    return kernel

# Lazy init — compile each kernel only when first needed (avoids 12min import timeout)
_KERNELS: dict[tuple, object] = {}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def custom_kernel(data: input_t) -> output_t:
    x, x_q, x_s = data          # x:[T,H] f16, x_q:[T,H] f32, x_s:[T,G] f32
    T, H = x.shape
    G   = x_s.shape[1]
    gsz = H // G
    N   = T * G
    key = (T, H, gsz)

    if key not in _KERNELS:
        _KERNELS[key] = _make_kernel(SHAPE_CONFIGS[key])
    kernel = _KERNELS[key]

    flat_in = x.reshape(N, gsz)
    flat_s  = x_s.reshape(N)

    flat_q = kernel(flat_in, flat_s)

    x_q[...] = flat_q.reshape(T, H)
    x_s[...] = flat_s.reshape(T, G)
    return x_q, x_s
