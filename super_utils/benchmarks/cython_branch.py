"""
Cython branch prediction benchmark - branch-heavy workload.

Wrapper for the Cython-compiled branch kernel.
Falls back gracefully if Cython module is not compiled.
"""

import numpy as np
from .base import BenchmarkBase


class CythonBranchBenchmark(BenchmarkBase):
    """
    Cython-compiled data-dependent branching benchmark.

    This is the Cython version of BranchPredictionHeavy, testing how
    compiler optimizations (branch elimination, if-conversion) affect
    data-dependent conditional code.

    Branch-heavy because conditions depend on runtime data values,
    making branch prediction difficult.
    """

    name = "cython_branch"
    description = "Cython data-dependent conditionals (branch-heavy)"
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
        self._counters = np.zeros(4, dtype=np.int64)  # [positive, negative, small, large]

        # Expected results for validation
        self._expected_output = None
        self._expected_counters = None

        # Import Cython kernel (raises ImportError if not compiled)
        from ._cython.branch import branch_process
        self._kernel = branch_process

    def run(self) -> None:
        """Execute Cython branch-heavy processing."""
        self._kernel(self._data, self._output, self._counters)

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
            self._expected_counters = np.array([
                np.sum(pos_mask),           # positive
                np.sum(neg_mask),           # negative
                np.sum(small_pos_mask) + np.sum(small_neg_mask),  # small
                np.sum(large_pos_mask) + np.sum(large_neg_mask),  # large
            ], dtype=np.int64)

        # Validate output array
        output_match = np.allclose(self._output, self._expected_output, rtol=1e-5)

        # Validate counters
        counters_match = np.array_equal(self._counters, self._expected_counters)

        return output_match and counters_match
