"""
SwiGLU (Swish-Gated Linear Unit) fused forward kernel in Helion.

Used in: Llama 2/3, Mistral, Mixtral, PaLM, Gemma FFN layers.
Formula: SwiGLU(x, gate) = x * SiLU(gate) = x * gate * sigmoid(gate)

Eager PyTorch baseline:
    return x * F.silu(gate)

Helion fuses into one kernel: no intermediate allocation for silu(gate).
For FFN layers: gate_proj + up_proj output feeds here — this is on the hot path.

Also includes fused SwiGLU backward (needed for training).
"""

import torch
import torch.nn.functional as F
import helion
import helion.language as hl


@helion.kernel()
def swiglu_forward(
    x: torch.Tensor,     # [batch, seq_len, intermediate_dim] or [tokens, intermediate_dim]
    gate: torch.Tensor,  # same shape as x
) -> torch.Tensor:
    """
    Fused SwiGLU: out = x * silu(gate)
    Single kernel, no intermediate allocation.
    """
    out = torch.empty_like(x)
    for tile_0, tile_1 in hl.tile([x.size(0), x.size(1)]):
        x_tile = x[tile_0, tile_1]
        gate_tile = gate[tile_0, tile_1]
        # silu(gate) = gate * sigmoid(gate), fused with multiply
        out[tile_0, tile_1] = x_tile * torch.sigmoid(gate_tile) * gate_tile
    return out


@helion.kernel()
def swiglu_backward(
    grad_out: torch.Tensor,  # [*, intermediate_dim]
    x: torch.Tensor,
    gate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Backward for SwiGLU.
    d/dx[x * silu(gate)] = silu(gate)
    d/dgate[x * silu(gate)] = x * silu'(gate)
    where silu'(z) = silu(z) + sigmoid(z) * (1 - silu(z))
                   = sigmoid(z) * (1 + z * (1 - sigmoid(z)))
    """
    grad_x = torch.empty_like(x)
    grad_gate = torch.empty_like(gate)

    for tile_0, tile_1 in hl.tile([x.size(0), x.size(1)]):
        g = grad_out[tile_0, tile_1].to(torch.float32)
        x_tile = x[tile_0, tile_1].to(torch.float32)
        gate_tile = gate[tile_0, tile_1].to(torch.float32)

        sig = torch.sigmoid(gate_tile)
        silu_val = gate_tile * sig

        grad_x[tile_0, tile_1] = (g * silu_val).to(x.dtype)
        # d_silu/d_gate = sig * (1 + gate * (1 - sig))
        d_silu_d_gate = sig * (1.0 + gate_tile * (1.0 - sig))
        grad_gate[tile_0, tile_1] = (g * x_tile * d_silu_d_gate).to(gate.dtype)

    return grad_x, grad_gate


class SwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate):
        ctx.save_for_backward(x, gate)
        return swiglu_forward(x, gate)

    @staticmethod
    def backward(ctx, grad_out):
        x, gate = ctx.saved_tensors
        return swiglu_backward(grad_out, x, gate)


def swiglu(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Differentiable fused SwiGLU."""
    return SwiGLUFunction.apply(x, gate)


@helion.kernel()
def swiglu_fused_3d(
    x: torch.Tensor,     # [batch, seq_len, intermediate_dim]
    gate: torch.Tensor,
) -> torch.Tensor:
    """3D version for attention FFN layers."""
    batch, seq_len, d = x.size()
    out = torch.empty_like(x)
    for tile_b, tile_s, tile_d in hl.tile([batch, seq_len, d]):
        x_tile = x[tile_b, tile_s, tile_d]
        gate_tile = gate[tile_b, tile_s, tile_d]
        out[tile_b, tile_s, tile_d] = x_tile * torch.sigmoid(gate_tile) * gate_tile
    return out
