#!POPCORN leaderboard causal_conv1d
#!POPCORN gpu B200_Nebius

from task import input_t, output_t
import torch
import torch.nn.functional as F
import helion
import helion.language as hl
import glob

# Optimization:
# 1. Use F.pad (faster than cat)
# 2. 3D Tiling (B, D, N) for better hardware utilization
# 3. Pre-load weights and bias into registers per tile
# 4. Localized accumulation for better register reuse

# Grab all the secret NVIDIA tuning files
acf_files = glob.glob("/opt/booster_pack/causal_conv_*.acf")

# Build the config parameters with 3D block sizes
config_params = {
    "block_sizes": [1, 16, 128], # [Batch, Depth, Seq]
    "num_warps": [4, 8], 
    "num_stages": [2, 3, 4]
}

if acf_files:
    config_params["advanced_controls_file"] = acf_files

TUNE_CONFIG = helion.Config(**config_params)

# --- HARDCODED LEADERBOARD CONFIGS ---
# NOTE: These now use 3D block_sizes [Batch, Depth, Seq]
# Update these with your tuned outputs after running benchmark on remote
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    (1, 64, 64, 4): helion.Config(block_sizes=[1, 32, 64], num_warps=2, num_stages=2),
    (2, 128, 128, 4): helion.Config(block_sizes=[1, 32, 128], num_warps=4, num_stages=2),
    (1, 256, 256, 3): helion.Config(block_sizes=[1, 32, 128], num_warps=4, num_stages=2),
    (1, 128, 64, 8): helion.Config(block_sizes=[1, 32, 64], num_warps=2, num_stages=2),
    (4, 64, 128, 4): helion.Config(block_sizes=[1, 32, 64], num_warps=2, num_stages=2),
    
    # Benchmark shapes
    (1, 768, 512, 4): helion.Config(block_sizes=[1, 32, 128], num_warps=4, num_stages=2),
    (1, 768, 2048, 4): helion.Config(block_sizes=[1, 32, 256], num_warps=4, num_stages=3),
    (1, 1536, 2048, 4): helion.Config(block_sizes=[1, 32, 256], num_warps=8, num_stages=3),
    (1, 2560, 2048, 4): helion.Config(block_sizes=[1, 32, 256], num_warps=8, num_stages=3),
    (1, 2560, 4096, 4): helion.Config(block_sizes=[1, 32, 256], num_warps=8, num_stages=3),
}

def _make_kernel(config: helion.Config):
    # CRITICAL: Do NOT pass autotune_search_acf or lists here during KernelBot submission
    @helion.kernel(static_shapes=True, config=config)
    def kernel(
        x_pad: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        B, D, L = x_pad.size(0), x_pad.size(1), x_pad.size(2)
        W = hl.specialize(w.size(1))
        N = L - W + 1
        y = torch.empty(B, D, N, dtype=x_pad.dtype, device=x_pad.device)

        # Tile over Batch, Channel (Depth), and Sequence
        # Using [1, None, None] to allow the autotuner/config to control D and N tiling
        for rb, rd, rs in hl.tile([B, D, N], block_size=[1, None, None]):
            bi = rb.begin
            
            # Pre-load bias for this tile of channels
            bias_tile = b[rd].to(torch.float32)
            
            # Local registers for weights across the j-loop
            # Loading weights once per tile instead of once per j-iteration
            weights = hl.zeros([rd, W], dtype=torch.float32)
            for j in range(W):
                weights[:, j] = w[rd, j].to(torch.float32)
            
            # Localized accumulator for the tile
            acc = hl.zeros([rd, rs], dtype=torch.float32)
            
            for j in range(W):
                # Load input chunk for this j-offset
                x_val = hl.load(x_pad, [bi, rd, rs.index + j]).to(torch.float32)
                # Compute
                acc = acc + x_val * weights[:, j, None]
                
            # Add bias and cast back
            acc = acc + bias_tile[:, None]
            y[rb, rd, rs] = acc.to(y.dtype)
        return y
    return kernel

_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}

def custom_kernel(data: input_t) -> output_t:
    x, weight, bias = data
    B, D, S = x.shape
    W = weight.shape[1]
    
    # Use F.pad for faster padding than cat
    padded = F.pad(x, (W - 1, 0))
    
    # Must use a strictly hardcoded config for KernelBot
    kernel = _KERNELS[(B, D, S, W)]
        
    return kernel(padded, weight, bias)
