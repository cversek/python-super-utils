# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
mfVEP Kernel Benchmark - Realistic Memory Hierarchy Test

This benchmark replicates the actual mfVEP kernel optimization challenge:
- 16,384 trials × 600 samples × 60 sectors
- Vectorized approach: 8.8 GB peak (OOM on 8GB systems)
- Streaming approach: ~65 KB peak (fits in L1 cache)

The streaming pattern processes one trial at a time, accumulating weighted
sums without materializing the full 3D arrays. This keeps the working set
small enough to fit in CPU cache, avoiding memory thrashing.

Reference: Cycle 7 JOTEWR "Streaming Through Memory"
Reference: Cycle 14 JOTEWR "Memory Hierarchy Truth"
"""

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

# Type definitions
ctypedef np.float32_t DTYPE_t
ctypedef np.int32_t INDEX_t


def mfvep_streaming_kernel(
    np.ndarray[DTYPE_t, ndim=1] signal,          # (total_samples,) - 1D signal
    np.ndarray[INDEX_t, ndim=1] trial_starts,    # (n_trials,) - start indices
    np.ndarray[DTYPE_t, ndim=2] weights,         # (n_sectors, n_trials) - kernel weights
    int epoch_length,                             # samples per epoch
    np.ndarray[DTYPE_t, ndim=2] output,          # (n_sectors, epoch_length) - pre-allocated
):
    """
    Streaming mfVEP kernel - processes one trial at a time.

    Memory pattern:
    - Never materializes full 3D arrays
    - Accumulator fits in L1 cache (~240 KB for 60 sectors × 1000 samples)
    - Each trial is processed and forgotten immediately

    This is the pattern that solved the 8GB OOM problem.
    """
    cdef int n_trials = trial_starts.shape[0]
    cdef int n_sectors = weights.shape[0]
    cdef int t, s, i
    cdef int start_idx
    cdef DTYPE_t w

    # Reset output accumulator
    output[:, :] = 0.0

    # Streaming accumulation: one trial at a time
    for t in range(n_trials):
        start_idx = trial_starts[t]
        for s in range(n_sectors):
            w = weights[s, t]
            for i in range(epoch_length):
                output[s, i] += w * signal[start_idx + i]


def mfvep_vectorized_reference(
    np.ndarray[DTYPE_t, ndim=1] signal,
    np.ndarray[INDEX_t, ndim=1] trial_starts,
    np.ndarray[DTYPE_t, ndim=2] weights,
    int epoch_length,
):
    """
    Vectorized reference implementation - allocates full 3D arrays.

    Memory pattern:
    - I_3d: (n_trials, epoch_length, n_sectors) int32 - indices
    - S_3d: (n_trials, epoch_length, n_sectors) float32 - signal data
    - Peak: n_trials * epoch_length * n_sectors * 8 bytes

    For 16k trials × 600 samples × 60 sectors = 8.8 GB peak

    This is the pattern that caused OOM on 8GB containers.
    WARNING: Will allocate large arrays. Do not run with n_trials > 8000
    on systems with less than 16GB RAM.
    """
    cdef int n_trials = trial_starts.shape[0]
    cdef int n_sectors = weights.shape[0]

    # Build 3D index array (this is the first big allocation)
    # Shape: (n_trials, epoch_length)
    I_2d = trial_starts[:, np.newaxis] + np.arange(epoch_length, dtype=np.int32)

    # Extract signal epochs (second big allocation)
    # Shape: (n_trials, epoch_length)
    S_2d = signal[I_2d]

    # Weighted sum across trials for each sector
    # weights: (n_sectors, n_trials), S_2d: (n_trials, epoch_length)
    # Result: (n_sectors, epoch_length)
    result = np.dot(weights, S_2d)

    return result.astype(np.float32)
