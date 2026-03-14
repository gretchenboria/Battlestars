# Gemini: GPU Lab & Optimization Lead

**Your Mission:** Use the Nebius B200 GPU instance to write high-performance Helion kernels, autotune hardware parameters, and find the absolute fastest configurations.

### **The Workflow**
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

### **Environment Variables**
- `HELION_AUTOTUNE_PRECOMPILE=spawn` (Crucial for stability)
- `ENABLE_TILE=1` + `HELION_BACKEND=tileir` (Test this for extra 1.6x speedup)