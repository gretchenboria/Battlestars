# Battlestars
Helion Hackathon Team Battlestars

## Current Status
We are competing in the PyTorch Helion Hackathon. The goal is to write high-performance GPU kernels using the Python-based Helion DSL to maximize Correctness and Performance points on the official leaderboard. We are currently SSH'd into an Ubuntu Nebius B200 GPU instance (`helion-dear-emu`).

### Tooling
* `popcorn` CLI is installed, authenticated, and configured.

### Current Problem
We are working on the `fp8_quant` warm-up problem (100 correctness points, 0 performance points).
* **Workspace:** The problem was scaffolded using `popcorn setup`, creating a directory with `submission.py`, `reference.py`, and `task.yml`.
* **Code:** We optimized the intentionally slow starter `submission.py` to remove redundant math operations (calculating `amax` only once instead of three times). The file currently contains the required `#!POPCORN leaderboard fp8_quant` and `#!POPCORN gpu B200_Nebius` headers.
* **Issue Encountered:** Helion's default `fork` precompilation caused the terminal to hang during local testing.

## Instructions & Submission Workflow

When generating or testing Helion kernels, you **MUST** follow these rules to avoid server timeouts and kernel crashes:

1. **Prevent Compiler Hangs:** Before running any test, set the spawn environment variable.
   ```bash
   export HELION_AUTOTUNE_PRECOMPILE=spawn
   ```

2. **No Server-Side Autotuning:** You MUST hardcode the best configurations (block sizes, warps, stages, etc.) into the `SHAPE_CONFIGS` dictionary inside `submission.py`. If you rely on Helion's autotuner during a KernelBot leaderboard submission, the server will time out and fail.

3. **Local Autotuning:** To find the best configs to hardcode, run local benchmarks on the Nebius GPU first using `autotune_effort="quick"` or `autotune_effort="full"`, or by searching the NVIDIA CompileIQ ACF files located at `/opt/booster_pack/*.acf`.

4. **Submission Commands:**
   * To verify correctness locally (do this first):
     ```bash
     popcorn submit --mode test submission.py
     ```
   * To benchmark performance locally:
     ```bash
     popcorn submit --mode benchmark submission.py
     ```
   * To submit officially to the hackathon leaderboard:
     ```bash
     popcorn submit --mode leaderboard submission.py
     ```

## Next Steps
Once `fp8_quant` successfully posts to the leaderboard, use `popcorn setup` to pull down the next problem (e.g., `causal_conv1d`). These subsequent problems are fully scored for performance (up to 1000 points per kernel). Optimize them heavily, test locally, hardcode the winning configs, and submit.
