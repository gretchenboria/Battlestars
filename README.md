# Battlestars
Helion Hackathon Team Battlestars

## Current Status
We are competing in the PyTorch Helion Hackathon. The goal is to write high-performance GPU kernels using the Python-based Helion DSL to maximize Correctness and Performance points on the official leaderboard. We are currently SSH'd into an Ubuntu Nebius B200 GPU instance (`helion-dear-emu`).

### Current Problem
We are working on the `fp8_quant` warm-up problem (100 correctness points, 0 performance points) and preparing for fully-scored performance problems like `causal_conv1d`.

---

## Instructions & Submission Workflow

When generating or testing Helion kernels, you **MUST** follow these rules to avoid server timeouts and kernel crashes:

1. **Setup/Scaffold:** Navigate to `helion/<problem_name>_py` or run `popcorn setup` to get the initial `submission.py` and `task.yml`.
   - **CRITICAL NOTE:** ONLY modify `submission.py` (your solution file) in each project. This is the only file that gets evaluated and submitted. Do not modify `reference.py`, `eval.py`, or `task.yml`.
2. **Algorithm Development:**
   - Minimize redundant math (e.g., don't calculate `amax` or `acc` multiple times).
   - Use `hl.tile()` to load data into SRAM once and sliding-window over it.
3. **Correctness Check:**
   - **Crucial:** Run `export HELION_AUTOTUNE_PRECOMPILE=spawn` to prevent terminal hangs.
   - Run local validation: `popcorn submit --mode test submission.py` or `python ../eval.py test .` to verify math against the reference.
4. **Benchmarking & Autotuning:**
   - Run `export HELION_AUTOTUNE_EFFORT=full`
   - Use ACFs: `autotune_search_acf=glob.glob("/opt/booster_pack/*.acf")`
   - Run `popcorn submit --mode benchmark submission.py` or `python ../eval.py benchmark .`
5. **Hardcoding Configs:** **DO NOT** submit with autotuning enabled. You must manually paste the `helion.Config(...)` values found during tuning into the `SHAPE_CONFIGS` dictionary for every benchmark shape.
6. **Submission:** Run the official leaderboard command:
   - `popcorn submit --mode leaderboard submission.py`

### **Environment Variables & Rules**
- `HELION_AUTOTUNE_PRECOMPILE=spawn` (Crucial for preventing compiler hangs)
- `ENABLE_TILE=1` and `HELION_BACKEND=tileir` (Test for extra 1.6x speedup)
- Use `#!POPCORN` headers at the top of `submission.py` for `leaderboard` and `gpu`.
- Only your best submission counts.
- Deadline is 6:00 PM today.

## Next Steps
Once `fp8_quant` successfully posts to the leaderboard, use `popcorn setup` to pull down the next problem (e.g., `causal_conv1d`). These subsequent problems are fully scored for performance (up to 1000 points per kernel). Optimize them heavily, test locally, hardcode the winning configs, and submit.
