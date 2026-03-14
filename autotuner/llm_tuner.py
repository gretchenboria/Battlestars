"""
LLM-guided autotuner for Helion kernels.

Helion's roadmap explicitly lists "LLM-guided search" as a future direction.
We're shipping it at the hackathon.

The idea:
  - Standard Helion autotuning (LFBO pattern search) explores configs blindly.
  - An LLM can reason about the kernel structure and hardware to propose
    smarter configs: "this is memory-bandwidth bound, so maximize block_size
    along the reduction dim and use tensor_descriptor indexing for TMA."
  - We use Claude as the surrogate model, feeding it benchmark history and
    asking it to propose the next config to try.

Architecture:
  LLMAutotuner
    .run(n_trials)
      -> propose_config(history) -> Claude API
      -> benchmark_config(config)
      -> update history
      -> return best config found
"""

import inspect
import json
import time
import os
from dataclasses import dataclass, asdict
from typing import Any, Callable

import torch
import helion

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False


# ---------------------------------------------------------------------------
# Config space description (mirrors helion.Config fields)
# ---------------------------------------------------------------------------

CONFIG_SPACE_DESCRIPTION = """
Helion Config Parameters (for NVIDIA B200 GPU):

BLOCK SIZES:
- block_sizes: list of ints, one per hl.tile() call in the kernel
  Powers of 2: [16, 32, 64, 128, 256, 512, 1024]
  Larger = better compute utilization but more shared memory pressure.
  For memory-bound kernels: larger block sizes amortize memory latency.
  For reduction dims: match to reduction loop size.

INDEXING:
- indexing: "pointer" | "block_ptr" | "tensor_descriptor"
  - "pointer": classic Triton pointer arithmetic, most compatible
  - "block_ptr": better for strided access, enables some optimizations
  - "tensor_descriptor": uses TMA (Tensor Memory Accelerator) on H100/B200,
    best for large contiguous tiles, enables async memory transfers

LOOP CONFIGURATION:
- loop_orders: list of [dim_order] per hl.tile(), controls iteration order
  e.g. [[1, 0]] swaps the two tile dimensions for better cache reuse
- l2_groupings: int (e.g. 4, 8, 32), groups blocks for L2 cache locality
  Important for matmul-like kernels with 2D output
- flatten_loops: bool, True=merge dims into single loop (better for simple kernels)
- pid_type: "flat" | "xyz" | "persistent_interleaved" | "persistent_blocked"
  - "flat": standard, one program per tile
  - "persistent_*": SM-based scheduling, better occupancy for small grids

PARALLELISM:
- num_warps: 4 | 8 | 16 | 32 (default 4)
  More warps = better latency hiding for memory-bound kernels
- num_stages: 1 | 2 | 3 | 4 | 5 (pipeline stages for memory ops)
  Higher = better pipelining but more shared memory

REDUCTION:
- reduction_loops: list of block sizes for reduction dims (e.g. [64], [128])
  Controls how reductions are chunked; None = load entire reduction at once

RANGE TUNABLES (per inner hl.tile loop):
- range_num_stages: list of ints per range loop
- range_warp_specializes: list of bool per range loop
- range_unroll_factors: list of ints (0=no unroll, 1=unroll)
- range_multi_buffers: list of bool (double buffering)
- range_flattens: list of bool

NVIDIA EXTRAS:
- advanced_controls_file: path to CompileIQ ACF file in /opt/booster_pack/
"""

SYSTEM_PROMPT = """You are an expert GPU kernel optimization engineer specializing in
Triton and Helion kernels on NVIDIA hardware, especially the B200 (Blackwell) GPU.

Your task is to propose Helion kernel configurations (helion.Config objects) that will
maximize performance. You understand:
- Memory bandwidth vs compute bound analysis
- Roofline model for B200 (HBM3e ~8TB/s, FP16 ~5 PFLOPS)
- How block sizes affect register pressure, shared memory, and occupancy
- When to use TMA (tensor_descriptor) vs pointer indexing
- Pipeline staging for hiding memory latency
- Persistent kernels for small grids

You will receive:
1. The kernel source code
2. The hardware (B200 GPU)
3. Previous benchmark results (config -> latency_ms)

Respond with a JSON object containing a single helion.Config specification.
Your response must be valid JSON with this structure:
{
  "block_sizes": [128, 64],
  "indexing": "block_ptr",
  "num_warps": 8,
  "num_stages": 3,
  "pid_type": "flat",
  "l2_groupings": [4],
  "loop_orders": [[0, 1]],
  "flatten_loops": [false],
  "reduction_loops": null,
  "range_num_stages": null,
  "range_warp_specializes": null,
  "range_unroll_factors": null,
  "range_multi_buffers": null,
  "range_flattens": null,
  "advanced_controls_file": null,
  "reasoning": "Brief explanation of why this config should be fast"
}

Only include fields you want to set. Omit fields to use Helion defaults.
The "reasoning" field is required — explain your optimization hypothesis.
"""


@dataclass
class TrialResult:
    config_dict: dict
    latency_ms: float
    throughput_gbs: float
    valid: bool
    error: str | None = None
    reasoning: str | None = None


class LLMAutotuner:
    """
    LLM-guided configuration search for Helion kernels.

    Implements the "LLM-guided search" roadmap item from the Helion presentation.
    Uses Claude to propose kernel configurations based on:
      1. Kernel source code analysis
      2. Hardware characteristics (B200)
      3. Previous benchmark results (exploit/explore tradeoff)
    """

    def __init__(
        self,
        kernel_fn: Callable,
        sample_inputs: list,
        bytes_accessed_fn: Callable | None = None,
        model: str = "claude-opus-4-6",
        warmup: int = 5,
        rep: int = 20,
    ):
        """
        Args:
            kernel_fn: The @helion.kernel()-decorated function
            sample_inputs: List of torch.Tensor inputs for benchmarking
            bytes_accessed_fn: Optional fn(inputs) -> int for GB/s calculation
            model: Claude model to use
            warmup: Warmup iterations for benchmarking
            rep: Repetitions for timing
        """
        if not _ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package required: pip install anthropic")

        self.kernel_fn = kernel_fn
        self.sample_inputs = sample_inputs
        self.bytes_accessed_fn = bytes_accessed_fn
        self.model = model
        self.warmup = warmup
        self.rep = rep

        self.client = anthropic.Anthropic()
        self.history: list[TrialResult] = []
        self.kernel_source = inspect.getsource(kernel_fn)

    def _build_user_prompt(self) -> str:
        """Build the prompt for Claude with full context."""
        lines = [
            "# Kernel to optimize",
            "```python",
            self.kernel_source,
            "```",
            "",
            "# Hardware",
            "NVIDIA B200 (Blackwell) GPU",
            "  - HBM3e memory bandwidth: ~8 TB/s",
            "  - FP16 peak: ~5 PFLOPS",
            "  - FP8 peak: ~10 PFLOPS",
            "  - 160 SMs, TMA support",
            "",
            "# Input shapes",
        ]

        for i, inp in enumerate(self.sample_inputs):
            if isinstance(inp, torch.Tensor):
                lines.append(f"  input[{i}]: shape={list(inp.shape)}, dtype={inp.dtype}")

        lines.append("")
        lines.append("# Config space")
        lines.append(CONFIG_SPACE_DESCRIPTION)

        if self.history:
            lines.append("")
            lines.append(f"# Previous trials ({len(self.history)} total)")
            lines.append("Sorted by latency (best first):")
            sorted_history = sorted(
                [r for r in self.history if r.valid],
                key=lambda r: r.latency_ms
            )
            for i, result in enumerate(sorted_history[:10]):
                cfg_str = json.dumps(result.config_dict, indent=None)
                lines.append(
                    f"  [{i+1}] {result.latency_ms:.3f}ms ({result.throughput_gbs:.1f} GB/s) "
                    f"| reasoning: {result.reasoning or 'N/A'}"
                )
                lines.append(f"       config: {cfg_str}")

            if len(sorted_history) > 0:
                best = sorted_history[0]
                lines.append(f"\nCurrent best: {best.latency_ms:.3f}ms")
                lines.append("Propose a config you expect to beat the current best, or explore a different region.")
        else:
            lines.append("\nNo previous trials yet. Propose a strong initial configuration.")

        lines.append("\nPropose the next configuration to benchmark:")
        return "\n".join(lines)

    def _parse_claude_response(self, response_text: str) -> tuple[dict, str]:
        """Extract JSON config from Claude's response."""
        # Try to find JSON in the response
        text = response_text.strip()

        # Handle ```json ... ``` blocks
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            text = text[start:end].strip()

        # Find outermost JSON object
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]

        config_dict = json.loads(text)
        reasoning = config_dict.pop("reasoning", "")
        # Remove None values to use Helion defaults
        config_dict = {k: v for k, v in config_dict.items() if v is not None}
        return config_dict, reasoning

    def _dict_to_config(self, config_dict: dict) -> helion.Config:
        """Convert a dict to a helion.Config object."""
        return helion.Config(**config_dict)

    def _benchmark(self, config: helion.Config) -> tuple[float, float]:
        """
        Benchmark a kernel with given config.
        Returns (latency_ms, throughput_gbs).
        """
        # Re-decorate with specific config
        # Helion allows passing config directly when calling the kernel
        fn = self.kernel_fn

        # Warmup
        for _ in range(self.warmup):
            try:
                out = fn(*self.sample_inputs, config=config)
                torch.cuda.synchronize()
            except Exception as e:
                raise RuntimeError(f"Kernel failed during warmup: {e}") from e

        # Timed runs
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(self.rep):
            fn(*self.sample_inputs, config=config)
        end.record()
        torch.cuda.synchronize()

        latency_ms = start.elapsed_time(end) / self.rep

        # Calculate throughput
        if self.bytes_accessed_fn:
            bytes_accessed = self.bytes_accessed_fn(self.sample_inputs)
            throughput_gbs = bytes_accessed / (latency_ms * 1e-3) / 1e9
        else:
            throughput_gbs = 0.0

        return latency_ms, throughput_gbs

    def propose_config(self) -> tuple[dict, str]:
        """Ask Claude to propose the next config to try."""
        user_prompt = self._build_user_prompt()

        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        response_text = message.content[0].text
        return self._parse_claude_response(response_text)

    def run(
        self,
        n_trials: int = 20,
        verbose: bool = True,
    ) -> TrialResult:
        """
        Run LLM-guided autotuning for n_trials.

        Returns the best TrialResult found.
        """
        print(f"\n{'='*60}")
        print(f"LLM Autotuner: {self.kernel_fn.__name__}")
        print(f"Model: {self.model} | Trials: {n_trials}")
        print(f"{'='*60}\n")

        for trial in range(n_trials):
            print(f"Trial {trial+1}/{n_trials}:")

            # Get LLM proposal
            t0 = time.time()
            try:
                config_dict, reasoning = self.propose_config()
                llm_time = time.time() - t0
                if verbose:
                    print(f"  LLM reasoning: {reasoning}")
                    print(f"  Config: {json.dumps(config_dict)}")
                    print(f"  (LLM response: {llm_time:.1f}s)")
            except Exception as e:
                print(f"  LLM error: {e}")
                continue

            # Benchmark
            try:
                config = self._dict_to_config(config_dict)
                latency_ms, throughput_gbs = self._benchmark(config)
                result = TrialResult(
                    config_dict=config_dict,
                    latency_ms=latency_ms,
                    throughput_gbs=throughput_gbs,
                    valid=True,
                    reasoning=reasoning,
                )
                print(f"  Result: {latency_ms:.3f}ms ({throughput_gbs:.1f} GB/s)")
            except Exception as e:
                result = TrialResult(
                    config_dict=config_dict,
                    latency_ms=float("inf"),
                    throughput_gbs=0.0,
                    valid=False,
                    error=str(e),
                    reasoning=reasoning,
                )
                print(f"  FAILED: {e}")

            self.history.append(result)

            # Show best so far
            valid = [r for r in self.history if r.valid]
            if valid:
                best = min(valid, key=lambda r: r.latency_ms)
                print(f"  Best so far: {best.latency_ms:.3f}ms")
            print()

        # Final summary
        valid = [r for r in self.history if r.valid]
        if not valid:
            raise RuntimeError("No valid configs found")

        best = min(valid, key=lambda r: r.latency_ms)
        print(f"\n{'='*60}")
        print(f"BEST CONFIG: {best.latency_ms:.3f}ms ({best.throughput_gbs:.1f} GB/s)")
        print(f"Config: {json.dumps(best.config_dict, indent=2)}")
        print(f"Reasoning: {best.reasoning}")
        print(f"{'='*60}\n")

        return best

    def compare_vs_baseline(
        self,
        baseline_fn: Callable,
        verbose: bool = True,
    ) -> dict:
        """
        Run autotuning and compare best result vs a baseline function.
        Returns speedup statistics.
        """
        # Benchmark baseline
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        for _ in range(self.warmup):
            baseline_fn(*self.sample_inputs)
        torch.cuda.synchronize()

        start.record()
        for _ in range(self.rep):
            baseline_fn(*self.sample_inputs)
        end.record()
        torch.cuda.synchronize()

        baseline_ms = start.elapsed_time(end) / self.rep

        # Find best Helion config
        valid = [r for r in self.history if r.valid]
        if not valid:
            raise RuntimeError("Run autotuning first with .run()")

        best = min(valid, key=lambda r: r.latency_ms)
        speedup = baseline_ms / best.latency_ms

        if verbose:
            print(f"\nBaseline: {baseline_ms:.3f}ms")
            print(f"LLM-tuned Helion: {best.latency_ms:.3f}ms")
            print(f"Speedup: {speedup:.2f}x")

        return {
            "baseline_ms": baseline_ms,
            "helion_ms": best.latency_ms,
            "speedup": speedup,
            "best_config": best.config_dict,
            "n_trials": len(self.history),
            "n_valid": len(valid),
        }
