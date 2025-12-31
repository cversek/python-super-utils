"""
Cython streaming accumulator benchmark - memory-bound workload.

Wrapper for the Cython-compiled streaming kernel.
Falls back gracefully if Cython module is not compiled.
"""

import numpy as np
from .base import BenchmarkBase


class CythonStreamingBenchmark(BenchmarkBase):
    """
    Cython-compiled nested loop accumulator benchmark.

    This is the Cython version of StreamingAccumulator, using typed
    memoryviews for efficient iteration through the nested loops.

    Memory-bound because the computation (simple addition) is trivial
    compared to memory access patterns.
    """

    name = "cython_streaming"
    description = "Cython nested loop accumulator (memory-bound)"
    workload_type = "memory-bound"

    def setup(self) -> None:
        """Initialize input arrays and output accumulator."""
        # Size selection (same as Python version)
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

        # Import Cython kernel (raises ImportError if not compiled)
        from ._cython.streaming import streaming_accumulate
        self._kernel = streaming_accumulate

    def run(self) -> None:
        """Execute Cython streaming accumulation."""
        self._kernel(self._input, self._output)

    def validate(self) -> bool:
        """Check against NumPy's vectorized sum."""
        # Compute expected result using vectorized NumPy
        if self._expected is None:
            self._expected = np.sum(self._input, axis=0)

        # Check relative error (allow slightly more tolerance for Cython)
        return np.allclose(self._output, self._expected, rtol=1e-5)
