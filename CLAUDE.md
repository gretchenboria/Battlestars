# AI Agent Workflow: Helion Hackathon

**Your Mission:** Use the local machine and/or the Nebius B200 GPU instance to write high-performance Helion kernels, autotune hardware parameters, and submit the absolute fastest configurations to KernelBot.

### **The Workflow**
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
