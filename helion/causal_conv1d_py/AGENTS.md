# AI Agent Knowledge Base: Helion Hackathon

## Compiler & DSL Lessons

### 1. Rank & Dimension Matching
When assigning a 2D accumulator (`acc`) to a 3D output slice (`y[rb, rd, rs]`), Helion requires the ranks to match exactly.
- **Incorrect:** `y[rb, rd, rs] = acc` (Rank mismatch)
- **Correct:** `y[rb, rd, rs] = acc[None, :, :]` (Adds dummy batch dimension)

### 2. hl.tile vs helion.Config
The `block_sizes` list in `helion.Config` must align **only** with the `None` values in `hl.tile`.
- **Example:** `hl.tile([B, D, N], block_size=[1, None, None])`
- **Rule:** `helion.Config(block_sizes=[Depth_Tile, Seq_Tile])` (Exactly 2 values)

### 3. Device Tensor Assignments
You cannot assign to subscripts of device tensors created inside the kernel.
- **Illegal:** `weights[:, j] = w[rd, j]`
- **Legal:** Load directly from the source tensor inside the compute loop.

### 4. Optimized Patterns
- **Padding:** Use `torch.nn.functional.pad` instead of `torch.cat`.
- **Broadcasting:** When multiplying a `[Tile_D, Tile_S]` data block by a `[Tile_D]` weight/bias vector, use `vector[:, None]` to trigger efficient broadcasting.

## Submission Workflow Checklist
- [ ] `#!POPCORN` headers at the top.
- [ ] `static_shapes=True` in `@helion.kernel`.
- [ ] Hardcoded `SHAPE_CONFIGS` dictionary.
- [ ] No `autotune_search_acf` or `autotune_effort` in the final submission code.
- [ ] Logic verified against `python ../eval.py test .` on remote B200.
