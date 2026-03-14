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

### B. Find Fastest Configs
Run the benchmark. Look for the "One can hardcode the best config" output at the end of the run.
```bash
python ../eval.py benchmark .
```

## 4. Final Leaderboard Submission
Once you have verified correctness and performance, run the official popcorn command.
```bash
popcorn submit --mode leaderboard submission.py
```

---

## Troubleshooting
- **401 Unauthorized:** Ensure you are running `popcorn` on the remote instance, not locally.
- **Compiler Hangs:** Double-check that `export HELION_AUTOTUNE_PRECOMPILE=spawn` was executed in your current shell.
- **OOM/Crashes:** If the machine freezes, a hard reboot may be required via the dashboard.
