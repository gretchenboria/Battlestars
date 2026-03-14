"""
RoPE (Rotary Position Embedding) kernels in Helion.

Every modern LLM (Llama, Mistral, DeepSeek, Qwen) uses RoPE.
Not covered in the Helion benchmark suite — this is a gap we're filling.

Eager PyTorch baseline:
    x1, x2 = x[..., :H//2], x[..., H//2:]
    return x * cos + cat([-x2, x1], dim=-1) * sin

Helion fuses all ops: no intermediate allocations, single kernel.
Expected speedup: 3-5x over eager (memory-bandwidth bound on B200).
"""

import torch
import helion
import helion.language as hl


@helion.kernel()
def rope_forward(
    x: torch.Tensor,    # [batch, seq_len, num_heads, head_dim]
    cos: torch.Tensor,  # [seq_len, head_dim]
    sin: torch.Tensor,  # [seq_len, head_dim]
) -> torch.Tensor:
    batch, seq_len, num_heads, head_dim = x.size()
    out = torch.empty_like(x)
    half = head_dim // 2

    for tile_b, tile_s, tile_h in hl.tile([batch, seq_len, num_heads]):
        # Load tile: [tile_b, tile_s, tile_h, head_dim]
        x_tile = x[tile_b, tile_s, tile_h, :].to(torch.float32)
        cos_tile = cos[tile_s, :].to(torch.float32)  # [tile_s, head_dim]
        sin_tile = sin[tile_s, :].to(torch.float32)

        # Broadcast cos/sin across batch and head dims
        # cos_tile: [tile_s, head_dim] -> [tile_b, tile_s, tile_h, head_dim]
        cos_tile = cos_tile.unsqueeze(0).unsqueeze(2)
        sin_tile = sin_tile.unsqueeze(0).unsqueeze(2)

        x1 = x_tile[..., :half]   # [tile_b, tile_s, tile_h, half]
        x2 = x_tile[..., half:]   # [tile_b, tile_s, tile_h, half]

        cos1 = cos_tile[..., :half]
        cos2 = cos_tile[..., half:]
        sin1 = sin_tile[..., :half]
        sin2 = sin_tile[..., half:]

        # Apply rotation: [x1*cos - x2*sin, x2*cos + x1*sin]
        out1 = x1 * cos1 - x2 * sin1
        out2 = x2 * cos2 + x1 * sin2

        result = torch.cat([out1, out2], dim=-1).to(x.dtype)
        out[tile_b, tile_s, tile_h, :] = result

    return out


@helion.kernel()
def rope_backward(
    grad_out: torch.Tensor,  # [batch, seq_len, num_heads, head_dim]
    cos: torch.Tensor,       # [seq_len, head_dim]
    sin: torch.Tensor,       # [seq_len, head_dim]
) -> torch.Tensor:
    """
    Backward pass for RoPE.
    RoPE is an orthogonal rotation, so the backward is the inverse rotation:
    inverse RoPE = apply RoPE with -sin (i.e., transpose of the rotation matrix).
    grad_x = rope(grad_out, cos, -sin)
    """
    batch, seq_len, num_heads, head_dim = grad_out.size()
    grad_x = torch.empty_like(grad_out)
    half = head_dim // 2

    for tile_b, tile_s, tile_h in hl.tile([batch, seq_len, num_heads]):
        g_tile = grad_out[tile_b, tile_s, tile_h, :].to(torch.float32)
        cos_tile = cos[tile_s, :].to(torch.float32).unsqueeze(0).unsqueeze(2)
        sin_tile = sin[tile_s, :].to(torch.float32).unsqueeze(0).unsqueeze(2)

        g1 = g_tile[..., :half]
        g2 = g_tile[..., half:]

        cos1 = cos_tile[..., :half]
        cos2 = cos_tile[..., half:]
        sin1 = sin_tile[..., :half]
        sin2 = sin_tile[..., half:]

        # Inverse rotation: transpose of [cos, -sin; sin, cos] = [cos, sin; -sin, cos]
        dx1 = g1 * cos1 + g2 * sin2
        dx2 = g2 * cos2 - g1 * sin1

        result = torch.cat([dx1, dx2], dim=-1).to(grad_out.dtype)
        grad_x[tile_b, tile_s, tile_h, :] = result

    return grad_x


class RoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin):
        ctx.save_for_backward(cos, sin)
        return rope_forward(x, cos, sin)

    @staticmethod
    def backward(ctx, grad_out):
        cos, sin = ctx.saved_tensors
        grad_x = rope_backward(grad_out, cos, sin)
        return grad_x, None, None


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Differentiable RoPE with Helion forward+backward kernels."""
    return RoPEFunction.apply(x, cos, sin)


def precompute_freqs(head_dim: int, seq_len: int, theta: float = 10000.0,
                     device: torch.device = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables for RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, freqs)  # [seq_len, head_dim//2]
    freqs = torch.cat([freqs, freqs], dim=-1)  # [seq_len, head_dim]
    return freqs.cos(), freqs.sin()
