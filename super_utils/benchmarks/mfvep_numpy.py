"""
mfVEP NumPy Benchmark - Pure Python/NumPy Reference Implementation

This is the pure Python/NumPy version of the mfVEP kernel benchmark,
paired with mfvep_realistic (Cython) for direct comparison.

Uses sector-by-sector accumulation pattern (memory-safe NumPy approach)
rather than full 3D vectorization which would cause OOM on large datasets.

Reference: Cycle 7/14 - mfVEP kernel optimization lessons
"""

import numpy as np
from .base import BenchmarkBase


class MFVEPNumpyBenchmark(BenchmarkBase):
    """
    Pure NumPy mfVEP kernel benchmark using sector-by-sector accumulation.

    This is the memory-safe NumPy approach that avoids full 3D materialization
    but is slower than Cython streaming due to Python loop overhead.
    """

    name = "mfvep"
    description = "NumPy mfVEP kernel (sector-by-sector accumulation)"
    workload_type = "memory-bound"

    def setup(self):
        """Initialize with realistic mfVEP dimensions."""
        size_map = {
            "small": (4000, 300, 60, 2_000_000),
            "medium": (8000, 600, 60, 5_000_000),
            "large": (16000, 600, 60, 10_000_000),
        }

        n_trials, epoch_length, n_sectors, signal_length = size_map[self.size]

        self._n_trials = n_trials
        self._epoch_length = epoch_length
        self._n_sectors = n_sectors

        # Generate synthetic data
        self._signal = np.random.randn(signal_length).astype(np.float32)
        max_start = signal_length - epoch_length - 1
        self._trial_starts = np.random.randint(0, max_start, size=n_trials).astype(np.int32)
        self._weights = ((np.random.rand(n_sectors, n_trials) > 0.5).astype(np.float32) * 2 - 1)
        self._output = np.zeros((n_sectors, epoch_length), dtype=np.float32)

        # Compute reference for validation
        self._expected = self._compute_reference()

    def _compute_reference(self):
        """Compute reference using nested loops (gold standard)."""
        result = np.zeros((self._n_sectors, self._epoch_length), dtype=np.float32)
        for s in range(self._n_sectors):
            for t in range(self._n_trials):
                start = self._trial_starts[t]
                w = self._weights[s, t]
                result[s] += w * self._signal[start:start + self._epoch_length]
        return result

    def run(self):
        """Execute NumPy sector-by-sector accumulation."""
        self._output.fill(0)

        # Sector-by-sector accumulation (avoids full 3D materialization)
        for s in range(self._n_sectors):
            for t in range(self._n_trials):
                start = self._trial_starts[t]
                w = self._weights[s, t]
                self._output[s] += w * self._signal[start:start + self._epoch_length]

    def validate(self):
        """Verify result matches reference."""
        return np.allclose(self._output, self._expected, rtol=1e-4, atol=1e-6)
