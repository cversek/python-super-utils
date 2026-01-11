"""
Cython-compiled benchmark kernels.

These modules provide optimized implementations of the benchmark algorithms
using Cython with typed memoryviews for performance testing.

Modules:
- streaming: Memory-bound nested loop accumulator
- wavelet: Compute-bound wavelet-like convolution
- branch: Branch-heavy data-dependent conditionals
- linalg: BLAS-intensive small matrix operations
- interp: Mixed workload interpolation pipeline
"""

# Kernels are imported dynamically by their wrapper benchmarks
# to handle cases where Cython is not compiled
