#!POPCORN leaderboard causal_conv1d
#!POPCORN gpu B200_Nebius

from task import input_t, output_t

import torch
import helion
import helion.language as hl

# ---------------------------------------------------------------------------
# Per-shape configs: autotuned on B200 (Helion LFBOTreeSearch + CompileIQ ACFs)
# All hardcoded — NO runtime autotuning (server would time out).
# ---------------------------------------------------------------------------

SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # (B, D, S, W)
    # Test shapes
    (1, 64, 64, 4):   helion.Config(block_sizes=[16, 64],  num_warps=4,  num_stages=2),
    (2, 128, 128, 4): helion.Config(block_sizes=[32, 128], num_warps=4,  num_stages=2),
    (1, 256, 256, 3): helion.Config(block_sizes=[32, 128], num_warps=4,  num_stages=2),
    (1, 128, 64, 8):  helion.Config(block_sizes=[16, 64],  num_warps=4,  num_stages=2),
    (4, 64, 128, 4):  helion.Config(block_sizes=[16, 128], num_warps=4,  num_stages=2),
    # Benchmark shapes
    (1, 768, 512, 4):   helion.Config(block_sizes=[64, 256],  num_warps=8,  num_stages=3),
    (1, 768, 2048, 4):  helion.Config(block_sizes=[64, 512],  num_warps=8,  num_stages=3),
    (1, 1536, 2048, 4): helion.Config(block_sizes=[128, 256], num_warps=16, num_stages=3),
    (1, 2560, 2048, 4): helion.Config(block_sizes=[128, 256], num_warps=16, num_stages=3),
    (1, 2560, 4096, 4): helion.Config(block_sizes=[128, 512], num_warps=16, num_stages=3),
}


# ---------------------------------------------------------------------------
# Kernel: clean single-pass depthwise causal conv1d
# ---------------------------------------------------------------------------

def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, config=config)
    def kernel(
        x_pad: torch.Tensor,  # [B, D, L]  left-padded input (W-1 zeros prepended)
        w: torch.Tensor,      # [D, W]     filter weights
        b: torch.Tensor,      # [D]        bias
    ) -> torch.Tensor:
        B = x_pad.size(0)
        D = x_pad.size(1)
        L = x_pad.size(2)
        W = hl.specialize(w.size(1))
        N = L - W + 1  # output length (== original S)

        y = torch.empty(B, D, N, dtype=x_pad.dtype, device=x_pad.device)

        for rb, rd, rs in hl.tile([B, D, N], block_size=[1, None, None]):
            bi = rb.begin
            acc = hl.zeros([rd, rs], dtype=torch.float32)
            for j in range(W):
                c   = w[rd, j].to(torch.float32)
                xt  = hl.load(x_pad, [bi, rd, rs.index + j]).to(torch.float32)
                acc = acc + xt * c[:, None]
            acc = acc + b[rd].to(torch.float32)[:, None]
            y[rb, rd, rs] = acc[None, :, :].to(y.dtype)

        return y

    return kernel


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def custom_kernel(data: input_t) -> output_t:
    x, weight, bias = data          # x: [B,D,S] f32, weight: [D,W] f32, bias: [D] f32
    B, D, S = x.shape
    W = weight.shape[1]
    kernel = _KERNELS[(B, D, S, W)]
    pad_zeros = torch.zeros(B, D, W - 1, dtype=x.dtype, device=x.device)
    padded = torch.cat([pad_zeros, x], dim=2)  # [B, D, S+W-1]
    return kernel(padded, weight, bias)


# ---------------------------------------------------------------------------
# Autotune (run directly: python submission.py --autotune)
# This block does NOT execute when imported by the leaderboard runner.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os, sys, json, time

    print("causal_conv1d autotune on B200")
    print("Usage: python submission.py [--autotune] [--test]")

    mode = sys.argv[1] if len(sys.argv) > 1 else "--autotune"

    device = torch.device("cuda")

    ALL_SHAPES = list(SHAPE_CONFIGS.keys())

    if mode == "--test":
        # Quick correctness check
        import torch.nn.functional as F
        print("\nCorrectness checks:")
        for B, D, S, W in ALL_SHAPES:
            x      = torch.randn(B, D, S, dtype=torch.float32, device=device)
            weight = torch.randn(D, W,   dtype=torch.float32, device=device)
            bias   = torch.randn(D,      dtype=torch.float32, device=device)

            # Reference
            x_pad_ref = F.pad(x, (W - 1, 0))
            ref = F.conv1d(x_pad_ref, weight.unsqueeze(1), bias=bias, groups=D)

            # Ours
            out = custom_kernel((x, weight, bias))

            ok = torch.allclose(ref, out, atol=1e-3, rtol=1e-3)
            print(f"  ({B},{D},{S},{W}): {'PASS' if ok else 'FAIL'}  "
                  f"max_err={( ref - out ).abs().max():.5f}")

    elif mode == "--autotune":
        # Autotune all shapes and print best configs
        @helion.kernel()
        def _autotune_kernel(
            x_pad: torch.Tensor,
            w: torch.Tensor,
            b: torch.Tensor,
        ) -> torch.Tensor:
            B  = x_pad.size(0)
            D  = x_pad.size(1)
            L  = x_pad.size(2)
            W2 = hl.specialize(w.size(1))
            N  = L - W2 + 1
            y  = torch.empty(B, D, N, dtype=x_pad.dtype, device=x_pad.device)
            for rb, rd, rs in hl.tile([B, D, N], block_size=[1, None, None]):
                bi = rb.begin
                acc = hl.zeros([rd, rs], dtype=torch.float32)
                for j in range(W2):
                    c   = w[rd, j].to(torch.float32)
                    x2  = hl.load(x_pad, [bi, rd, rs.index + j]).to(torch.float32)
                    acc = acc + x2 * c[:, None]
                acc = acc + b[rd].to(torch.float32)[:, None]
                y[rb, rd, rs] = acc[None, :, :].to(y.dtype)
            return y

        ACF_FILES = sorted(
            f for f in (
                "/opt/booster_pack/causal_conv_0.acf",
                "/opt/booster_pack/causal_conv_1.acf",
                "/opt/booster_pack/causal_conv_2.acf",
            ) if os.path.exists(f)
        )
        print(f"CompileIQ ACF files: {[os.path.basename(f) for f in ACF_FILES]}")

        best_configs = {}

        def bench(fn, *args, warmup=5, rep=20):
            for _ in range(warmup):
                fn(*args)
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(rep):
                fn(*args)
            e.record()
            torch.cuda.synchronize()
            return s.elapsed_time(e) / rep

        for B, D, S, W in ALL_SHAPES:
            print(f"\n{'='*60}")
            print(f"Shape ({B},{D},{S},{W})")

            x_pad = torch.randn(B, D, S + W - 1, dtype=torch.float32, device=device)
            w_t   = torch.randn(D, W,             dtype=torch.float32, device=device)
            b_t   = torch.randn(D,                dtype=torch.float32, device=device)

            _autotune_kernel.reset()
            t0 = time.time()
            best_cfg = _autotune_kernel.autotune([x_pad, w_t, b_t], force=True)
            print(f"Autotune done in {time.time()-t0:.1f}s  config: {dict(best_cfg)}")

            # Benchmark best
            def _run(x_p, w_, b_):
                @helion.kernel(config=best_cfg)
                def _k(xp, ww, bb):
                    B2 = xp.size(0)
                    D2 = xp.size(1)
                    L2 = xp.size(2)
                    W3 = hl.specialize(ww.size(1))
                    N2 = L2 - W3 + 1
                    yy = torch.empty(B2, D2, N2, dtype=xp.dtype, device=xp.device)
                    for rb, rd, rs in hl.tile([B2, D2, N2], block_size=[1, None, None]):
                        bi = rb.begin
                        acc = hl.zeros([rd, rs], dtype=torch.float32)
                        for j in range(W3):
                            c   = ww[rd, j].to(torch.float32)
                            x3  = hl.load(xp, [bi, rd, rs.index + j]).to(torch.float32)
                            acc = acc + x3 * c[:, None]
                        acc = acc + bb[rd].to(torch.float32)[:, None]
                        yy[rb, rd, rs] = acc[None, :, :].to(yy.dtype)
                    return yy
                return _k(x_p, w_, b_)

            ms = bench(lambda: _run(x_pad, w_t, b_t))
            best_ms  = ms
            best_acf = None

            # Try ACF booster files
            for acf in ACF_FILES:
                cfg_acf = helion.Config(**{**dict(best_cfg), "advanced_controls_file": acf})
                try:
                    def _run_acf(xp, ww, bb, cfg=cfg_acf):
                        @helion.kernel(config=cfg)
                        def _k(xp2, ww2, bb2):
                            B2 = xp2.size(0); D2 = xp2.size(1); L2 = xp2.size(2)
                            W3 = hl.specialize(ww2.size(1)); N2 = L2 - W3 + 1
                            yy = torch.empty(B2, D2, N2, dtype=xp2.dtype, device=xp2.device)
                            for rb, rd, rs in hl.tile([B2, D2, N2], block_size=[1, None, None]):
                                bi = rb.begin
                                acc = hl.zeros([rd, rs], dtype=torch.float32)
                                for j in range(W3):
                                    c   = ww2[rd, j].to(torch.float32)
                                    x3  = hl.load(xp2, [bi, rd, rs.index + j]).to(torch.float32)
                                    acc = acc + x3 * c[:, None]
                                acc = acc + bb2[rd].to(torch.float32)[:, None]
                                yy[rb, rd, rs] = acc[None, :, :].to(yy.dtype)
                            return yy
                        return _k(xp, ww2, bb2)
                    ms_acf = bench(lambda: _run_acf(x_pad, w_t, b_t))
                    marker = " *** NEW BEST ***" if ms_acf < best_ms else ""
                    print(f"  {os.path.basename(acf)}: {ms_acf:.3f}ms{marker}")
                    if ms_acf < best_ms:
                        best_ms  = ms_acf
                        best_acf = acf
                except Exception as e:
                    print(f"  {os.path.basename(acf)}: FAILED ({e})")

            final_cfg = helion.Config(**{**dict(best_cfg), **({"advanced_controls_file": best_acf} if best_acf else {})})
            best_configs[(B, D, S, W)] = final_cfg
            print(f"→ Best: {dict(final_cfg)}  ({best_ms:.3f}ms)")

        print("\n\n# Paste into SHAPE_CONFIGS:")
        print("SHAPE_CONFIGS: dict[tuple, helion.Config] = {")
        for shape, cfg in best_configs.items():
            d = {k: v for k, v in dict(cfg).items() if v is not None}
            print(f"    {shape}: helion.Config({', '.join(f'{k}={v!r}' for k,v in d.items())}),")
        print("}")

        with open("/home/ubuntu/causal_conv1d_best_configs.json", "w") as f:
            json.dump({str(s): {k: v for k, v in dict(c).items() if v is not None}
                       for s, c in best_configs.items()}, f, indent=2)
        print("Saved to /home/ubuntu/causal_conv1d_best_configs.json")
