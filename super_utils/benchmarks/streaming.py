"""
Streaming accumulator benchmark - models MFVEP kernel averaging pattern.

Memory-bound workload: nested loop accumulator with minimal arithmetic.
Tests memory bandwidth and cache efficiency.
"""

import numpy as np
from .base import BenchmarkBase


class StreamingAccumulator(BenchmarkBase):
    """
    Nested loop accumulator benchmark.

    Simulates the pattern from MFVEP kernel computation:
    - Outer loop over trials
    - Inner loop over epochs
    - Accumulate into output array

    This is memory-bound because the computation (simple addition) is trivial
    compared to memory access patterns.
    """

    name = "streaming"
    description = "Nested loop accumulator (models MFVEP kernel averaging)"
    workload_type = "memory-bound"

    def setup(self) -> None:
        """Initialize input arrays and output accumulator."""
        # Size selection
        size_map = {
            "small": (1000, 100, 60),    # ~5 MB
            "medium": (5000, 200, 60),   # ~50 MB
            "large": (16000, 500, 60),   # ~500 MB (close to real MFVEP)
        }
        n_trials, n_epochs, n_sectors = size_map.get(self.size, size_map["medium"])

        # Input data: trial-by-epoch-by-sector
        self._input = np.random.randn(n_trials, n_epochs, n_sectors).astype(np.float32)

        # Output accumulator: epoch-by-sector
        self._output = np.zeros((n_epochs, n_sectors), dtype=np.float32)

        # Expected result for validation
        self._expected = None

    def run(self) -> None:
        """Execute streaming accumulation."""
        n_trials, n_epochs, n_sectors = self._input.shape

        # Reset output
        self._output[:] = 0.0

        # Nested loop accumulation (Python + NumPy broadcasting)
        # This models what Cython would do: iterate trials, accumulate epochs
        for trial_idx in range(n_trials):
            self._output += self._input[trial_idx]

    def validate(self) -> bool:
        """Check against NumPy's vectorized sum."""
        # Compute expected result using vectorized NumPy
        if self._expected is None:
            self._expected = np.sum(self._input, axis=0)

        # Check relative error
        diff = np.abs(self._output - self._expected)
        max_val = np.abs(self._expected).max()
        relative_error = (diff / (max_val + 1e-10)).max()

        return relative_error < 1e-5
