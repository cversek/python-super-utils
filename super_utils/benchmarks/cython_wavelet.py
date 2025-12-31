"""
Cython wavelet-like convolution benchmark - compute-bound workload.

Wrapper for the Cython-compiled wavelet kernel.
Falls back gracefully if Cython module is not compiled.
"""

import numpy as np
from .base import BenchmarkBase


class CythonWaveletBenchmark(BenchmarkBase):
    """
    Cython-compiled wavelet decomposition/reconstruction benchmark.

    Unlike the Python version which uses pywavelets, this implements
    a simplified wavelet-like convolution directly in Cython to test
    compiler optimization of compute-intensive loops.

    Compute-bound because the convolution involves many arithmetic
    operations per memory access.
    """

    name = "cython_wavelet"
    description = "Cython wavelet-like convolution (compute-bound)"
    workload_type = "compute-bound"

    def setup(self) -> None:
        """Initialize test signals."""
        # Size selection
        size_map = {
            "small": (100, 1024),      # 100 signals, 1k samples
            "medium": (500, 4096),     # 500 signals, 4k samples
            "large": (1000, 8192),     # 1k signals, 8k samples
        }
        n_signals, signal_length = size_map.get(self.size, size_map["medium"])

        # Generate random signals
        self._signals = np.random.randn(n_signals, signal_length).astype(np.float64)

        # Decomposition levels
        self._levels = 5

        # Storage for results
        self._reconstructed = np.zeros_like(self._signals)
        self._expected = None

        # Import Cython kernel (raises ImportError if not compiled)
        from ._cython.wavelet import wavelet_convolve
        self._kernel = wavelet_convolve

    def run(self) -> None:
        """Execute wavelet-like convolution."""
        self._kernel(self._signals, self._reconstructed, self._levels)

    def validate(self) -> bool:
        """
        Check that output is finite and has reasonable values.

        Note: Unlike pywavelets SWT, this simplified implementation
        does not guarantee perfect reconstruction, so we validate
        that results are numerically reasonable rather than exact.
        """
        # Check for NaN/Inf
        if not np.all(np.isfinite(self._reconstructed)):
            return False

        # Check that values are in reasonable range
        # (should not explode due to filter instability)
        max_input = np.abs(self._signals).max()
        max_output = np.abs(self._reconstructed).max()

        # Output should not be orders of magnitude larger than input
        if max_output > max_input * 100:
            return False

        return True
