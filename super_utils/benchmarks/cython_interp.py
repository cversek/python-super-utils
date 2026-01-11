"""
Cython interpolation and filtering benchmark - mixed workload.

Wrapper for the Cython-compiled interp kernel.
Falls back gracefully if Cython module is not compiled.
"""

import numpy as np
from .base import BenchmarkBase


class CythonInterpBenchmark(BenchmarkBase):
    """
    Cython-compiled interpolation + filtering pipeline benchmark.

    Unlike the Python version which uses scipy, this implements
    linear interpolation and moving average filtering directly
    in Cython to test mixed workload optimization.

    Mixed workload because it combines:
    - Memory-bound: Array access patterns
    - Compute-bound: Filter accumulation
    - Irregular access: Interpolation lookups
    """

    name = "cython_interp"
    description = "Cython interpolation + filter pipeline (mixed)"
    workload_type = "mixed"

    def setup(self) -> None:
        """Initialize test signals."""
        # Size selection
        size_map = {
            "small": (100, 1000),     # 100 signals, 1k samples
            "medium": (500, 4000),    # 500 signals, 4k samples
            "large": (1000, 8000),    # 1k signals, 8k samples
        }
        n_signals, signal_length = size_map.get(self.size, size_map["medium"])

        # Generate random signals with some structure
        t = np.linspace(0, 10, signal_length)
        self._signals = np.zeros((n_signals, signal_length), dtype=np.float64)
        for i in range(n_signals):
            # Mix of sine waves + noise
            freq = 0.5 + i * 0.01
            self._signals[i] = np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(signal_length)

        # Original time axis
        self._t_original = t.astype(np.float64)

        # Upsampling factor
        self._upsample_factor = 4

        # Upsampled time axis
        upsampled_length = signal_length * self._upsample_factor
        self._t_upsampled = np.linspace(t[0], t[-1], upsampled_length).astype(np.float64)

        # Filter width (odd number for symmetric moving average)
        self._filter_width = 7

        # Output storage (back to original length after decimation)
        self._output = np.zeros((n_signals, signal_length), dtype=np.float64)
        self._expected = None

        # Import Cython kernel (raises ImportError if not compiled)
        from ._cython.interp import interp_and_filter
        self._kernel = interp_and_filter

    def run(self) -> None:
        """Execute Cython interpolation and filtering pipeline."""
        self._kernel(
            self._signals,
            self._t_original,
            self._t_upsampled,
            self._output,
            self._filter_width
        )

    def validate(self) -> bool:
        """
        Validate that output is finite and has reasonable smoothing.

        Since the Cython version uses linear interpolation and simple
        moving average (vs. scipy's cubic + Butterworth), we validate
        that results are numerically reasonable rather than matching
        the Python version exactly.
        """
        # Check for NaN/Inf
        if not np.all(np.isfinite(self._output)):
            return False

        # Check that values are in reasonable range
        max_input = np.abs(self._signals).max()
        max_output = np.abs(self._output).max()

        # Output should be similar magnitude to input
        # (smoothing should not amplify signal)
        if max_output > max_input * 2:
            return False

        # Output should not be completely flat (some variation preserved)
        if np.std(self._output) < np.std(self._signals) * 0.1:
            return False

        return True
