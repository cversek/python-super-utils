# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Streaming accumulator kernel - memory-bound workload.

Ports the nested loop accumulation pattern from streaming.py to Cython
with typed memoryviews for efficient iteration.
"""

import numpy as np
cimport numpy as np
cimport cython


def streaming_accumulate(
    np.ndarray[np.float32_t, ndim=3] input_data,
    np.ndarray[np.float32_t, ndim=2] output
):
    """
    Accumulate 3D input array along first axis into 2D output.

    Args:
        input_data: 3D array of shape (n_trials, n_epochs, n_sectors)
        output: 2D array of shape (n_epochs, n_sectors) - modified in place

    This models the MFVEP kernel averaging pattern where trials are
    accumulated into epoch-by-sector results.
    """
    cdef Py_ssize_t n_trials = input_data.shape[0]
    cdef Py_ssize_t n_epochs = input_data.shape[1]
    cdef Py_ssize_t n_sectors = input_data.shape[2]

    cdef Py_ssize_t trial_idx, epoch_idx, sector_idx
    cdef float acc

    # Zero output array
    for epoch_idx in range(n_epochs):
        for sector_idx in range(n_sectors):
            output[epoch_idx, sector_idx] = 0.0

    # Streaming accumulation - iterate through all elements
    for trial_idx in range(n_trials):
        for epoch_idx in range(n_epochs):
            for sector_idx in range(n_sectors):
                output[epoch_idx, sector_idx] += input_data[trial_idx, epoch_idx, sector_idx]
