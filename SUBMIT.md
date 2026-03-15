# Remote GPU Submission Guide

Follow these steps exactly on your **remote B200 instance** to evaluate and submit your kernels.

## 1. Sync Latest Code
Always start by pulling the latest optimizations from the repository.
```bash
git pull origin hackathon
```

## 2. Environment Setup
Run these to ensure the compiler doesn't hang and the autotuner works effectively.
```bash
export HELION_AUTOTUNE_PRECOMPILE=spawn
export HELION_AUTOTUNE_EFFORT=full
```

## 3. Verify & Tune (The "Loop")
Navigate to the problem directory (e.g., `causal_conv1d_py`).
```bash
cd helion/causal_conv1d_py
```

### A. Check Correctness
```bash
python ../eval.py test .
```

### B. Find Fastest Configs & Benchmark
Run the benchmark to see how fast your kernel is.
```bash
python ../eval.py benchmark .
```

**What to expect:**
- If the agent has *already* hardcoded the tuned configs for you (like in `causal_conv1d`), the benchmark will just print out the fast execution times (e.g., `0.0302 ms`). **You are ready to submit! Skip to Step 4.**
- If you are running an untuned kernel that is still autotuning, the terminal will print out the fastest config for each shape at the end of the run, looking like this:
  `One can hardcode the best config and skip autotuning with: @helion.kernel(config=helion.Config(...))`

### C. HARDCODE THE CONFIGS (If not already done)
KernelBot will timeout and fail if you submit while autotuning is enabled. 
If Step 3B printed out new `helion.Config` recommendations, you must:
1. Open `submission.py` on your remote machine (using `nano submission.py` or `vim submission.py`).
2. Find the `SHAPE_CONFIGS` dictionary.
3. Replace the `helion.Config(...)` for each benchmark shape with the fastest config printed in Step 3B.
4. Save the file.

*(Optional but recommended: `git commit` and `git push` these tuned configs back to the repo so you don't lose them!)*

## 4. Final Leaderboard Submission
Once the configs are hardcoded in `submission.py`, run the official popcorn command to get on the leaderboard!
```bash
popcorn submit --mode leaderboard submission.py --no-tui
```

---

## Troubleshooting
- **KernelBot Timeouts:** You forgot Step 3C! Make sure `SHAPE_CONFIGS` has the hardcoded `helion.Config(...)` values and that your `_make_kernel` function does NOT have `autotune_search_acf=acf_files` or lists for `block_sizes`. (The code I pushed handles this safely for you).
- **401 Unauthorized:** Ensure you are running `popcorn` on the remote instance, not locally.
- **Compiler Hangs:** Double-check that `export HELION_AUTOTUNE_PRECOMPILE=spawn` was executed in your current shell.
