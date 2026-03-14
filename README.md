# Battlestars
Helion Hackathon Team Battlestars

## Current Status
We are competing in the PyTorch Helion Hackathon. The goal is to write high-performance GPU kernels using the Python-based Helion DSL to maximize Correctness and Performance points on the official leaderboard. We are currently SSH'd into an Ubuntu Nebius B200 GPU instance (`helion-dear-emu`).

### Current Problem
We are working on the `fp8_quant` warm-up problem (100 correctness points, 0 performance points) and preparing for fully-scored performance problems like `causal_conv1d`.

---

## Team Roles & Workflows

We are splitting responsibilities between two agents to maximize efficiency before the 6:00 PM deadline.

### 🤖 Gemini: GPU Lab & Optimization Lead
**Mission:** Use the Nebius B200 GPU instance to write high-performance Helion kernels, autotune hardware parameters, and find the absolute fastest configurations.

**Workflow:**
1. **Navigate to Problem:** `cd reference-kernels/problems/helion/<problem_name>_py`
2. **Algorithm Development:**
   - Minimize redundant math (e.g., don't calculate `amax` or `acc` multiple times).
   - Use `hl.tile()` to load data into SRAM once and sliding-window over it.
3. **Correctness Check:**
   - Run `export HELION_AUTOTUNE_PRECOMPILE=spawn` to prevent terminal hangs.
   - Run `python ../eval.py test .` to verify math against the reference.
4. **The "1000 Point" Benchmarking:**
   - Run `export HELION_AUTOTUNE_EFFORT=full`
   - Use ACFs: `autotune_search_acf=glob.glob("/opt/booster_pack/*.acf")`
   - Run `python ../eval.py benchmark .`
5. **Reporting:** Once a shape is tuned, copy the `helion.Config(...)` string and send it to the Leaderboard Lead (Laptop).

**Crucial Environment Variables:**
- `HELION_AUTOTUNE_PRECOMPILE=spawn` (Fixes terminal hangs)
- `ENABLE_TILE=1` and `HELION_BACKEND=tileir` (Test for extra 1.6x speedup)

### 🤖 Claude: Mission Control & Leaderboard Lead
**Mission:** Use the Local Laptop to manage the codebase, sync with the GPU Lab, and perform official submissions to KernelBot.

**Workflow:**
1. **Scaffold:** Run `popcorn setup` to get the initial `submission.py` and `task.yml`.
2. **Git Sync:** Ensure the optimized code and `SHAPE_CONFIGS` from the GPU Lab (Gemini) are pulled into your local branch.
3. **Hardcoding:** **DO NOT** submit with autotuning enabled. You must manually paste the `helion.Config(...)` values provided by the GPU Lab into the `SHAPE_CONFIGS` dictionary for every benchmark shape.
4. **Validation:** Run a quick `popcorn submit --mode test submission.py` to ensure the final file is formatted correctly.
5. **Submission:** Run the official leaderboard command:
   - `popcorn submit --mode leaderboard submission.py`

**Submission Rules:**
- Use `#!POPCORN` headers at the top of the file for `leaderboard` and `gpu`.
- Only your best submission counts.
- Deadline is 6:00 PM today.