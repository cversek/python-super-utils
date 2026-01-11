"""
mfVEP Realistic Benchmark - Memory Hierarchy Stress Test

This benchmark replicates the actual mfVEP kernel optimization challenge
that caused OOM failures on 8GB containers. It demonstrates why the
streaming pattern was essential - not just faster, but executable.

Dimensions based on real clinical datasets:
- small:  4k trials × 300 samples × 60 sectors  (~288 MB vectorized peak)
- medium: 8k trials × 600 samples × 60 sectors  (~1.1 GB vectorized peak)
- large:  16k trials × 600 samples × 60 sectors (~4.4 GB vectorized peak)

The "large" size approaches the real mfVEP problem that motivated
Cycle 7's Cython streaming solution.

Reference: Cycle 7 JOTEWR "Streaming Through Memory"
Reference: Cycle 14 JOTEWR "Memory Hierarchy Truth"
"""

import numpy as np
from .base import BenchmarkBase


class CythonMFVEPBenchmark(BenchmarkBase):
    """
    Cython mfVEP kernel benchmark that demonstrates memory hierarchy effects.

    Uses Cython streaming kernel which keeps working set in L1 cache.
    Compare against mfvep_numpy to see why streaming was essential.
    """

    name = "cython_mfvep"
    description = "Cython mfVEP kernel (streaming, cache-friendly)"
    workload_type = "memory-bound"

    def setup(self):
        """Initialize with realistic mfVEP dimensions."""
        # Size configurations based on actual clinical data processing
        size_map = {
            # (n_trials, epoch_length, n_sectors, signal_length)
            "small": (4000, 300, 60, 2_000_000),      # ~288 MB vectorized peak
            "medium": (8000, 600, 60, 5_000_000),     # ~1.1 GB vectorized peak
            "large": (16000, 600, 60, 10_000_000),    # ~4.4 GB vectorized peak
        }

        n_trials, epoch_length, n_sectors, signal_length = size_map[self.size]

        # Store dimensions for reporting
        self._n_trials = n_trials
        self._epoch_length = epoch_length
        self._n_sectors = n_sectors

        # Calculate expected memory for vectorized approach
        vectorized_bytes = n_trials * epoch_length * n_sectors * 4 * 2  # I_3d + S_3d
        self._vectorized_peak_mb = vectorized_bytes / (1024 * 1024)

        # Generate synthetic signal (1D, like real EEG data)
        self._signal = np.random.randn(signal_length).astype(np.float32)

        # Generate trial start indices (random positions in signal)
        max_start = signal_length - epoch_length - 1
        self._trial_starts = np.random.randint(
            0, max_start, size=n_trials
        ).astype(np.int32)

        # Generate kernel weights (M-sequence-like: -1 or +1)
        self._weights = (
            (np.random.rand(n_sectors, n_trials) > 0.5).astype(np.float32) * 2 - 1
        )

        # Pre-allocate output (this is what stays in cache)
        self._output = np.zeros((n_sectors, epoch_length), dtype=np.float32)

        # Compute reference using sector-by-sector accumulation (memory-safe)
        self._expected = self._compute_reference()

    def _compute_reference(self):
        """Compute reference result using memory-safe sector-by-sector approach."""
        result = np.zeros((self._n_sectors, self._epoch_length), dtype=np.float32)

        for s in range(self._n_sectors):
            for t in range(self._n_trials):
                start = self._trial_starts[t]
                w = self._weights[s, t]
                result[s] += w * self._signal[start:start + self._epoch_length]

        return result

    def run(self):
        """Execute Cython streaming kernel."""
        from ._cython.mfvep_kernel import mfvep_streaming_kernel

        mfvep_streaming_kernel(
            self._signal,
            self._trial_starts,
            self._weights,
            self._epoch_length,
            self._output,
        )

    def validate(self):
        """Verify streaming result matches reference."""
        return np.allclose(self._output, self._expected, rtol=1e-4, atol=1e-6)

    def get_stats(self):
        """Return additional statistics about this benchmark."""
        return {
            "n_trials": self._n_trials,
            "epoch_length": self._epoch_length,
            "n_sectors": self._n_sectors,
            "vectorized_peak_mb": self._vectorized_peak_mb,
            "streaming_peak_kb": (self._n_sectors * self._epoch_length * 4) / 1024,
            "memory_ratio": self._vectorized_peak_mb * 1024 / (
                self._n_sectors * self._epoch_length * 4 / 1024
            ),
        }
