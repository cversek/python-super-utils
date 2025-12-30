"""
Branch prediction benchmark - data-dependent conditionals.

Branch-heavy workload: nested loops with unpredictable conditional logic.
Tests CPU branch predictor efficiency and pipeline stalls.
"""

import numpy as np
from .base import BenchmarkBase


class BranchPredictionHeavy(BenchmarkBase):
    """
    Data-dependent branching benchmark.

    Processes 2D data with nested conditionals based on runtime values.
    This stresses the CPU's branch predictor because:
    - Conditions depend on data values (not predictable at compile time)
    - Multiple nested if/else branches
    - Mix of simple and complex operations per branch

    Branch mispredictions cause pipeline stalls, making this sensitive
    to compiler optimizations that can eliminate or reorder branches.
    """

    name = "branch"
    description = "Data-dependent conditionals with nested loops"
    workload_type = "branch-heavy"

    def setup(self) -> None:
        """Initialize test data with varied values."""
        # Size selection
        size_map = {
            "small": (1000, 500),     # 1k x 500
            "medium": (5000, 1000),   # 5k x 1k
            "large": (10000, 2000),   # 10k x 2k
        }
        rows, cols = size_map.get(self.size, size_map["medium"])

        # Random data with varied distribution to make branches unpredictable
        self._data = np.random.randn(rows, cols).astype(np.float32)

        # Output arrays
        self._output = np.zeros_like(self._data)
        self._counters = {
            'positive': 0,
            'negative': 0,
            'small': 0,
            'large': 0,
        }

        # Expected results for validation
        self._expected_output = None
        self._expected_counters = None

    def run(self) -> None:
        """Execute branch-heavy processing."""
        rows, cols = self._data.shape

        # Reset state
        self._output[:] = 0.0
        for key in self._counters:
            self._counters[key] = 0

        # Nested loops with data-dependent branches
        for i in range(rows):
            for j in range(cols):
                val = self._data[i, j]

                # First-level branch: sign
                if val > 0:
                    self._counters['positive'] += 1

                    # Second-level branch: magnitude
                    if val > 1.0:
                        self._counters['large'] += 1
                        self._output[i, j] = val ** 2
                    else:
                        self._counters['small'] += 1
                        self._output[i, j] = val * 0.5
                else:
                    self._counters['negative'] += 1

                    # Second-level branch: magnitude
                    if val < -1.0:
                        self._counters['large'] += 1
                        self._output[i, j] = -(val ** 2)
                    else:
                        self._counters['small'] += 1
                        self._output[i, j] = val * 0.5

    def validate(self) -> bool:
        """Validate against vectorized NumPy implementation."""
        if self._expected_output is None:
            # Compute expected using NumPy (vectorized, no branches)
            self._expected_output = np.zeros_like(self._data)

            # Positive values
            pos_mask = self._data > 0
            large_pos_mask = self._data > 1.0
            small_pos_mask = pos_mask & ~large_pos_mask

            self._expected_output[large_pos_mask] = self._data[large_pos_mask] ** 2
            self._expected_output[small_pos_mask] = self._data[small_pos_mask] * 0.5

            # Negative values
            neg_mask = self._data <= 0
            large_neg_mask = self._data < -1.0
            small_neg_mask = neg_mask & ~large_neg_mask

            self._expected_output[large_neg_mask] = -(self._data[large_neg_mask] ** 2)
            self._expected_output[small_neg_mask] = self._data[small_neg_mask] * 0.5

            # Expected counters
            self._expected_counters = {
                'positive': np.sum(pos_mask),
                'negative': np.sum(neg_mask),
                'small': np.sum(small_pos_mask) + np.sum(small_neg_mask),
                'large': np.sum(large_pos_mask) + np.sum(large_neg_mask),
            }

        # Validate output array
        output_match = np.allclose(self._output, self._expected_output, rtol=1e-5)

        # Validate counters
        counters_match = all(
            self._counters[k] == self._expected_counters[k]
            for k in self._counters
        )

        return output_match and counters_match
