# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Branch prediction heavy kernel - branch-heavy workload.

Implements data-dependent conditional processing to stress the CPU's
branch predictor. Compiler optimizations may eliminate or reorder branches.
"""

import numpy as np
cimport numpy as np
cimport cython


def branch_process(
    np.ndarray[np.float32_t, ndim=2] data,
    np.ndarray[np.float32_t, ndim=2] output,
    np.ndarray[np.int64_t, ndim=1] counters
):
    """
    Process data with nested data-dependent conditionals.

    Args:
        data: 2D input array of shape (rows, cols)
        output: 2D output array of same shape - modified in place
        counters: 1D array of 4 counters [positive, negative, small, large]

    The nested conditionals based on runtime values stress branch prediction:
    - First level: sign (positive/negative)
    - Second level: magnitude (small/large based on threshold 1.0)
    """
    cdef Py_ssize_t rows = data.shape[0]
    cdef Py_ssize_t cols = data.shape[1]

    cdef Py_ssize_t i, j
    cdef float val

    # Counter indices
    cdef Py_ssize_t IDX_POSITIVE = 0
    cdef Py_ssize_t IDX_NEGATIVE = 1
    cdef Py_ssize_t IDX_SMALL = 2
    cdef Py_ssize_t IDX_LARGE = 3

    # Reset counters
    counters[IDX_POSITIVE] = 0
    counters[IDX_NEGATIVE] = 0
    counters[IDX_SMALL] = 0
    counters[IDX_LARGE] = 0

    # Nested loops with data-dependent branches
    for i in range(rows):
        for j in range(cols):
            val = data[i, j]

            # First-level branch: sign
            if val > 0:
                counters[IDX_POSITIVE] += 1

                # Second-level branch: magnitude
                if val > 1.0:
                    counters[IDX_LARGE] += 1
                    output[i, j] = val * val  # val ** 2
                else:
                    counters[IDX_SMALL] += 1
                    output[i, j] = val * 0.5
            else:
                counters[IDX_NEGATIVE] += 1

                # Second-level branch: magnitude
                if val < -1.0:
                    counters[IDX_LARGE] += 1
                    output[i, j] = -(val * val)  # -(val ** 2)
                else:
                    counters[IDX_SMALL] += 1
                    output[i, j] = val * 0.5
