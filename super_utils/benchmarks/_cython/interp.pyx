# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Interpolation and filtering kernel - mixed workload.

Implements linear interpolation and simple moving average filtering
in Cython. This creates a mixed workload with memory access patterns
(interpolation lookups) and computation (filtering).
"""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport floor


def interp_and_filter(
    np.ndarray[np.float64_t, ndim=2] signals,
    np.ndarray[np.float64_t, ndim=1] t_original,
    np.ndarray[np.float64_t, ndim=1] t_upsampled,
    np.ndarray[np.float64_t, ndim=2] output,
    int filter_width
):
    """
    Interpolate signals to new time axis and apply smoothing filter.

    Args:
        signals: 2D array of shape (n_signals, signal_length)
        t_original: 1D array of original time points
        t_upsampled: 1D array of upsampled time points
        output: 2D output array of shape (n_signals, len(t_original)) - decimated result
        filter_width: Width of moving average filter (odd number)

    Pipeline:
    1. Linear interpolation to t_upsampled
    2. Moving average filter
    3. Decimation back to original length
    """
    cdef Py_ssize_t n_signals = signals.shape[0]
    cdef Py_ssize_t original_length = signals.shape[1]
    cdef Py_ssize_t upsampled_length = t_upsampled.shape[0]

    cdef Py_ssize_t sig_idx, i, j, k, idx_low, idx_high
    cdef double t, t0, t1, v0, v1, frac, acc
    cdef double t_min, t_max, dt_original
    cdef int half_width

    # Precompute original time step
    t_min = t_original[0]
    t_max = t_original[original_length - 1]
    dt_original = (t_max - t_min) / (original_length - 1)

    half_width = filter_width // 2

    # Allocate working buffer for upsampled + filtered signal
    cdef np.ndarray[np.float64_t, ndim=1] upsampled = np.zeros(upsampled_length, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] filtered = np.zeros(upsampled_length, dtype=np.float64)

    for sig_idx in range(n_signals):
        # Step 1: Linear interpolation
        for i in range(upsampled_length):
            t = t_upsampled[i]

            # Clamp to range
            if t <= t_min:
                upsampled[i] = signals[sig_idx, 0]
            elif t >= t_max:
                upsampled[i] = signals[sig_idx, original_length - 1]
            else:
                # Find bracketing indices
                frac = (t - t_min) / dt_original
                idx_low = <Py_ssize_t>floor(frac)

                if idx_low >= original_length - 1:
                    idx_low = original_length - 2
                idx_high = idx_low + 1

                # Linear interpolation
                t0 = t_original[idx_low]
                t1 = t_original[idx_high]
                v0 = signals[sig_idx, idx_low]
                v1 = signals[sig_idx, idx_high]

                if t1 > t0:
                    frac = (t - t0) / (t1 - t0)
                else:
                    frac = 0.0

                upsampled[i] = v0 + frac * (v1 - v0)

        # Step 2: Moving average filter
        for i in range(upsampled_length):
            acc = 0.0
            k = 0
            for j in range(i - half_width, i + half_width + 1):
                if 0 <= j < upsampled_length:
                    acc += upsampled[j]
                    k += 1
            if k > 0:
                filtered[i] = acc / k
            else:
                filtered[i] = upsampled[i]

        # Step 3: Decimation - pick evenly spaced samples
        for i in range(original_length):
            idx_low = (i * upsampled_length) // original_length
            if idx_low >= upsampled_length:
                idx_low = upsampled_length - 1
            output[sig_idx, i] = filtered[idx_low]
