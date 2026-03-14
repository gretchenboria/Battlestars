# Causal Depthwise 1D Convolution (Reference Logic)

This file summarizes the ground-truth mathematical logic provided in `reference.py`. To earn correctness points in the hackathon, your custom Helion implementation must replicate this logic within the defined numerical tolerances.

## 1. Input Specifications
The evaluation harness prepares three tensors in `float32` on the GPU:
*   **`x` (Input):** Shape `[B, D, S]` — (Batch Size, Features/Dimensions, Sequence Length).
*   **`weight`:** Shape `[D, W]` — (Features, Convolution Window Size).
*   **`bias`:** Shape `[D]` — One bias value per feature channel.

## 2. Mathematical Logic
The reference implementation performs the convolution in three distinct conceptual steps:

### A. Causal Padding
```python
x_padded = F.pad(x, (W - 1, 0))
```
To ensure "Causality," the output at time $t$ must only depend on inputs at time $t$ and earlier ($t, t-1, ... t-W+1$). This is achieved by padding the **left side** (start) of the sequence with $W-1$ zeros.

### B. Depthwise Processing
```python
output = F.conv1d(..., groups=D)
```
The convolution is **Depthwise**, meaning each feature channel is processed independently. Channel $i$ of the input is convolved with row $i$ of the weight matrix. The channels do not interact or mix in this operation.

### C. Weight Transformation
The reference logic unsqueezes the weights from `[D, W]` to `[D, 1, W]` to satisfy PyTorch's `conv1d` requirement for depthwise groups, where the second dimension represents `In_Channels / Groups`.

## 3. Correctness Requirements
Your submission must match this implementation's output within:
*   **Relative Tolerance (`rtol`):** `1e-4`
*   **Absolute Tolerance (`atol`):** `1e-4`

The `DeterministicContext()` is active during verification to ensure that any non-deterministic GPU optimizations are disabled for stable comparison.
