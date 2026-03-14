# Claude: Mission Control & Leaderboard Lead

**Your Mission:** Use the Local Laptop to manage the codebase, sync with the GPU Lab, and perform official submissions to KernelBot.

### **The Workflow**
1. **Scaffold:** Run `popcorn setup` to get the initial `submission.py` and `task.yml`.
2. **Git Sync:** Ensure the optimized code and `SHAPE_CONFIGS` from the GPU Lab (Gemini) are pulled into your local branch.
3. **Hardcoding:** **DO NOT** submit with autotuning enabled. You must manually paste the `helion.Config(...)` values provided by the GPU Lab into the `SHAPE_CONFIGS` dictionary for every benchmark shape.
4. **Validation:** Run a quick `popcorn submit --mode test submission.py` to ensure the final file is formatted correctly.
5. **Submission:** Run the official leaderboard command:
   - `popcorn submit --mode leaderboard submission.py`

### **Submission Rules**
- Use `#!POPCORN` headers at the top of the file for `leaderboard` and `gpu`.
- Only your best submission counts.
- Deadline is 6:00 PM today.