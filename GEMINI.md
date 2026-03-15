# AI Agent Workflow: Helion Hackathon

**Your Mission:** Write high-performance Helion kernels, evaluate/benchmark them exclusively on the remote Nebius B200 GPU instance, and submit the absolute fastest configurations.

### **The Workflow**
1. **Develop Code:** Write and optimize your Helion kernels locally in `helion/<problem_name>_py/submission.py`.
   - **CRITICAL NOTE:** ONLY modify `submission.py` (your solution file) in each project. This is the only file that gets evaluated and submitted.
   - Algorithm strategy: Minimize redundant math, use `hl.tile()` to load data into SRAM once, and sliding-window over it.
2. **Push Code:** Commit and push your local changes to the Git repository.
3. **Sync to GPU Lab:** SSH into the Nebius B200 GPU instance (using `./login.sh`). Pull your latest code from the repository.
4. **Test & Benchmark on GPU:** (Perform these steps on the SSH session)
   - **Crucial:** Run `export HELION_AUTOTUNE_PRECOMPILE=spawn` to prevent terminal hangs.
   - Run validation: `python ../eval.py test .`
   - Run benchmarking: `export HELION_AUTOTUNE_EFFORT=full` and `python ../eval.py benchmark .`
5. **Hardcoding Configs:** Take the best `helion.Config(...)` values found during the remote benchmark and manually paste them into the `SHAPE_CONFIGS` dictionary in `submission.py`. Commit and push this updated config.
6. **Submission:** Run the official leaderboard command from the remote GPU instance:
   - `popcorn submit --mode leaderboard submission.py`

### **Environment Variables & Rules (For Remote GPU)**
- `HELION_AUTOTUNE_PRECOMPILE=spawn` (Crucial for preventing compiler hangs)
- `ENABLE_TILE=1` and `HELION_BACKEND=tileir` (Test for extra 1.6x speedup)
- Use `#!POPCORN` headers at the top of `submission.py` for `leaderboard` and `gpu`.
- Deadline is 6:00 PM today.
