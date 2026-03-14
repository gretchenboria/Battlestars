#!POPCORN leaderboard gated_deltanet_chunk_fwd_o
#!POPCORN gpu B200_Nebius

from task import input_t, output_t
import torch
import helion
import helion.language as hl
import glob

# Optimization Strategy:
# 1. Math simplification for `exp(g_diff)`:
#    exp(g_i - g_j) = exp(g_i) * exp(-g_j). 
#    We multiply Q by exp(g_i) and K by exp(-g_j) before the dot product.
#    This eliminates the memory-heavy broadcasting of `g_diff` and the large `torch.exp` on a 2D matrix.
# 2. Avoid redundant scaling logic.
# 3. Block sizing and configuration are appropriately mapped.

acf_files = glob.glob("/opt/booster_pack/chunk_fwd_o_*.acf")

config_params = {
    "block_sizes": [], # Using default tiling of [1, 64]
    "num_warps": [2, 4, 8],
    "num_stages": [2, 3, 4]
}

if acf_files:
    config_params["advanced_controls_file"] = acf_files

TUNE_CONFIG = helion.Config(**config_params)

# --- HARDCODED LEADERBOARD CONFIGS ---
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes
    (1, 64, 2, 64, 64): helion.Config(block_sizes=[], num_warps=2, num_stages=2),
    (2, 128, 4, 64, 64): helion.Config(block_sizes=[], num_warps=4, num_stages=2),
    (1, 256, 4, 64, 128): helion.Config(block_sizes=[], num_warps=4, num_stages=2),
    # Benchmark shapes
    (1, 64, 1, 64, 64): helion.Config(block_sizes=[], num_warps=2, num_stages=2),
    (2, 512, 3, 64, 64): helion.Config(block_sizes=[], num_warps=4, num_stages=2),
    (2, 1024, 3, 64, 64): helion.Config(block_sizes=[], num_warps=4, num_stages=2),
    (3, 1024, 4, 100, 100): helion.Config(block_sizes=[], num_warps=4, num_stages=2),
    (4, 1024, 4, 128, 128): helion.Config(block_sizes=[], num_warps=4, num_stages=2),
    (2, 1536, 4, 128, 128): helion.Config(block_sizes=[], num_warps=4, num_stages=2),
    (4, 2048, 8, 64, 64): helion.Config(block_sizes=[], num_warps=8, num_stages=2),
}

DEFAULT_CONFIG = helion.Config(block_sizes=[], num_warps=4, num_stages=2)

def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, dot_precision="ieee", config=config)
    def kernel(
        q: torch.Tensor,     # [B, T, H, K]
        k: torch.Tensor,     # [B, T, H, K]
        v: torch.Tensor,     # [B, T, H, V]
        h: torch.Tensor,     # [B, NT, H, K, V]
        g: torch.Tensor,     # [B, T, H]
        scale: float,
    ) -> torch.Tensor:
        B, T, H, K = q.shape
        V = v.shape[-1]
        C = 64
        K = hl.specialize(K)
        V = hl.specialize(V)

        out = torch.empty_like(v)
        BH = B * H

        for flat_bh, tile_t in hl.tile([BH, T], block_size=[1, C]):
            b_idx = flat_bh.begin // H
            h_idx = flat_bh.begin % H
            c_idx = tile_t.begin // C

            # Load gate values
            g_vals = g[b_idx, tile_t, h_idx].to(torch.float32)
            g_exp = torch.exp(g_vals)
            g_exp_inv = torch.exp(-g_vals)

            q_tile = q[b_idx, tile_t, h_idx, :]
            k_tile = k[b_idx, tile_t, h_idx, :]
            v_tile = v[b_idx, tile_t, h_idx, :]

            # Optimization 1: scale Q and K independently instead of computing exp(g_diff)
            q_s = q_tile.to(torch.float32) * g_exp[:, None]
            k_s = k_tile.to(torch.float32) * g_exp_inv[:, None]

            # Intra-chunk (local attention)
            sim = hl.dot(q_s, k_s.T)
            idx = hl.arange(tile_t.block_size)
            causal_mask = idx[:, None] >= idx[None, :]
            sim = torch.where(causal_mask, sim, 0.0)
            local_out = hl.dot(sim.to(v.dtype), v_tile)

            # Inter-chunk (global attention)
            h_state = h[b_idx, c_idx, h_idx, :, :]
            global_out = hl.dot(q_s.to(q.dtype), h_state)

            # Combine and apply scale factor
            out[b_idx, tile_t, h_idx, :] = ((global_out + local_out) * scale).to(out.dtype)

        return out

    return kernel

_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}
_DEFAULT_KERNEL = _make_kernel(DEFAULT_CONFIG)

def custom_kernel(data: input_t) -> output_t:
    q, k, v_new, h, g = data
    B, T, H, K = q.shape
    V = v_new.shape[-1]
    scale = K ** -0.5
    
    kernel = _KERNELS.get((B, T, H, K, V), _DEFAULT_KERNEL)
    return kernel(q, k, v_new, h, g, scale)
