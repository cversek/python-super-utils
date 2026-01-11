"""
Interpolation and filtering benchmark - mixed workload.

Mixed workload: combines interpolation, filtering, and array operations.
Tests a realistic pipeline mixing memory access, computation, and vectorization.
"""

import numpy as np
from scipy import signal, interpolate
from .base import BenchmarkBase


class InterpolationFilter(BenchmarkBase):
    """
    Interpolation + filtering pipeline benchmark.

    Processes signals through a realistic pipeline:
    1. Cubic interpolation (upsampling)
    2. Butterworth filtering
    3. Decimation (downsampling)

    This is a mixed workload because it combines:
    - Memory-bound: Array copying, reshaping
    - Compute-bound: Filter convolution
    - Irregular access: Interpolation lookups

    Common pattern in signal processing workflows.
    """

    name = "interp"
    description = "Interpolation + filtering pipeline"
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
        self._t_original = t

        # Upsampling factor
        self._upsample_factor = 4

        # Filter design (Butterworth lowpass)
        self._filter_order = 4
        self._filter_cutoff = 0.3  # Normalized frequency

        # Design filter once
        self._b, self._a = signal.butter(
            self._filter_order,
            self._filter_cutoff,
            btype='low',
            analog=False
        )

        # Output storage
        self._output = None
        self._expected = None

    def run(self) -> None:
        """Execute interpolation and filtering pipeline."""
        n_signals, signal_length = self._signals.shape
        upsampled_length = signal_length * self._upsample_factor

        # New time axis (upsampled)
        t_upsampled = np.linspace(
            self._t_original[0],
            self._t_original[-1],
            upsampled_length
        )

        # Process each signal
        results = []
        for signal_data in self._signals:
            # Step 1: Cubic interpolation (upsample)
            interp_func = interpolate.interp1d(
                self._t_original,
                signal_data,
                kind='cubic',
                bounds_error=False,
                fill_value='extrapolate'
            )
            upsampled = interp_func(t_upsampled)

            # Step 2: Apply Butterworth filter
            filtered = signal.filtfilt(self._b, self._a, upsampled)

            # Step 3: Decimate back to original length
            decimated = signal.resample(filtered, signal_length)

            results.append(decimated)

        self._output = np.array(results)

    def validate(self) -> bool:
        """Validate by comparing against cached expected result."""
        if self._expected is None:
            # Compute expected (same as run, but cached)
            self.run()
            self._expected = self._output.copy()
            return True

        # Check match
        if self._output is None:
            return False

        return np.allclose(self._output, self._expected, rtol=1e-10)
