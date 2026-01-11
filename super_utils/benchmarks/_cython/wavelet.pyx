# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Wavelet-like convolution kernel - compute-bound workload.

Implements a simplified wavelet decomposition/reconstruction using
direct convolution. This tests compute throughput rather than relying
on pywavelets library.
"""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, cos, sin


# Daubechies-4 wavelet coefficients
cdef double H0 = (1.0 + sqrt(3.0)) / (4.0 * sqrt(2.0))
cdef double H1 = (3.0 + sqrt(3.0)) / (4.0 * sqrt(2.0))
cdef double H2 = (3.0 - sqrt(3.0)) / (4.0 * sqrt(2.0))
cdef double H3 = (1.0 - sqrt(3.0)) / (4.0 * sqrt(2.0))

# High-pass filter (derived from low-pass)
cdef double G0 = H3
cdef double G1 = -H2
cdef double G2 = H1
cdef double G3 = -H0


def wavelet_convolve(
    np.ndarray[np.float64_t, ndim=2] signals,
    np.ndarray[np.float64_t, ndim=2] output,
    int levels
):
    """
    Apply wavelet-like multi-level convolution to signals.

    Args:
        signals: 2D array of shape (n_signals, signal_length)
        output: 2D array of same shape - modified in place with reconstructed signals
        levels: Number of decomposition/reconstruction levels

    This performs a simplified wavelet transform using direct convolution
    to stress the CPU's compute capabilities.
    """
    cdef Py_ssize_t n_signals = signals.shape[0]
    cdef Py_ssize_t signal_length = signals.shape[1]

    cdef Py_ssize_t sig_idx, i, j, level
    cdef double low_sum, high_sum, recon_sum
    cdef Py_ssize_t idx0, idx1, idx2, idx3

    # Working buffers
    cdef np.ndarray[np.float64_t, ndim=1] low = np.zeros(signal_length, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] high = np.zeros(signal_length, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] temp = np.zeros(signal_length, dtype=np.float64)

    for sig_idx in range(n_signals):
        # Copy input to temp
        for i in range(signal_length):
            temp[i] = signals[sig_idx, i]

        # Multi-level decomposition and reconstruction
        for level in range(levels):
            # Forward pass: decompose into low and high frequency
            for i in range(signal_length):
                # Circular convolution indices
                idx0 = i
                idx1 = (i + 1) % signal_length
                idx2 = (i + 2) % signal_length
                idx3 = (i + 3) % signal_length

                # Low-pass filter (approximation coefficients)
                low[i] = H0 * temp[idx0] + H1 * temp[idx1] + H2 * temp[idx2] + H3 * temp[idx3]

                # High-pass filter (detail coefficients)
                high[i] = G0 * temp[idx0] + G1 * temp[idx1] + G2 * temp[idx2] + G3 * temp[idx3]

            # Inverse pass: reconstruct from low and high
            for i in range(signal_length):
                # Circular convolution indices for reconstruction
                idx0 = (i - 3 + signal_length) % signal_length
                idx1 = (i - 2 + signal_length) % signal_length
                idx2 = (i - 1 + signal_length) % signal_length
                idx3 = i

                # Reconstruction using transposed filters
                recon_sum = (
                    H3 * low[idx0] + H2 * low[idx1] + H1 * low[idx2] + H0 * low[idx3] +
                    G3 * high[idx0] + G2 * high[idx1] + G1 * high[idx2] + G0 * high[idx3]
                )
                temp[i] = recon_sum

        # Copy result to output
        for i in range(signal_length):
            output[sig_idx, i] = temp[i]
