"""
Wavelet transform benchmark - compute-intensive signal processing.

Compute-bound workload: repeated wavelet decomposition and reconstruction.
Tests CPU throughput and vectorization efficiency.
"""

import numpy as np
from .base import BenchmarkBase

try:
    import pywt
    PYWAVELETS_AVAILABLE = True
except ImportError:
    PYWAVELETS_AVAILABLE = False


class WaveletTransform(BenchmarkBase):
    """
    Wavelet decomposition/reconstruction benchmark.

    Performs stationary wavelet transform (SWT) on multiple signals.
    This is compute-bound because wavelet filtering involves:
    - Convolution operations (FIR filters)
    - Downsampling/upsampling
    - Multiple decomposition levels

    Requires pywavelets package.
    """

    name = "wavelet"
    description = "Wavelet decomposition/reconstruction (uses pywavelets)"
    workload_type = "compute-bound"

    def setup(self) -> None:
        """Initialize test signals."""
        if not PYWAVELETS_AVAILABLE:
            raise ImportError("pywavelets required for wavelet benchmark")

        # Size selection
        size_map = {
            "small": (100, 1024),      # 100 signals, 1k samples
            "medium": (500, 4096),     # 500 signals, 4k samples
            "large": (1000, 8192),     # 1k signals, 8k samples
        }
        n_signals, signal_length = size_map.get(self.size, size_map["medium"])

        # Generate random signals
        self._signals = np.random.randn(n_signals, signal_length).astype(np.float64)

        # Wavelet parameters
        self._wavelet = 'db4'  # Daubechies 4
        self._level = 5        # Decomposition levels

        # Storage for results
        self._reconstructed = np.zeros_like(self._signals)
        self._expected = None

    def run(self) -> None:
        """Execute wavelet decomposition and reconstruction."""
        for i, signal in enumerate(self._signals):
            # Decompose
            coeffs = pywt.swt(signal, self._wavelet, level=self._level, trim_approx=True)

            # Reconstruct
            self._reconstructed[i] = pywt.iswt(coeffs, self._wavelet)

    def validate(self) -> bool:
        """Check perfect reconstruction property."""
        if not PYWAVELETS_AVAILABLE:
            return False

        # SWT should provide perfect reconstruction
        diff = np.abs(self._signals - self._reconstructed)
        max_error = diff.max()

        # Allow small numerical error
        return max_error < 1e-10
