# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Small dense linear algebra kernel - BLAS-intensive workload.

Implements small matrix multiply operations directly in Cython.
For matrix inversion, we still use NumPy since hand-implementing
stable matrix inversion is complex and error-prone.
"""

import numpy as np
cimport numpy as np
cimport cython


def small_matmul(
    np.ndarray[np.float64_t, ndim=3] matrices_a,
    np.ndarray[np.float64_t, ndim=3] matrices_b,
    np.ndarray[np.float64_t, ndim=3] products
):
    """
    Multiply many small matrices: C[i] = A[i] @ B[i].

    Args:
        matrices_a: 3D array of shape (n_matrices, size, size)
        matrices_b: 3D array of shape (n_matrices, size, size)
        products: 3D output array of same shape - modified in place

    Implements direct matrix multiplication for small matrices
    to test Cython loop performance vs. BLAS.
    """
    cdef Py_ssize_t n_matrices = matrices_a.shape[0]
    cdef Py_ssize_t size = matrices_a.shape[1]

    cdef Py_ssize_t m, i, j, k
    cdef double acc

    for m in range(n_matrices):
        for i in range(size):
            for j in range(size):
                acc = 0.0
                for k in range(size):
                    acc += matrices_a[m, i, k] * matrices_b[m, k, j]
                products[m, i, j] = acc


def small_matmul_and_inv(
    np.ndarray[np.float64_t, ndim=3] matrices_a,
    np.ndarray[np.float64_t, ndim=3] matrices_b,
    np.ndarray[np.float64_t, ndim=3] products,
    np.ndarray[np.float64_t, ndim=3] inverses
):
    """
    Multiply and invert many small matrices.

    Args:
        matrices_a: 3D array of shape (n_matrices, size, size) - positive definite
        matrices_b: 3D array of shape (n_matrices, size, size)
        products: 3D output for C = A @ B
        inverses: 3D output for A^-1

    Matrix multiplication is done in Cython; inversion uses NumPy
    for numerical stability.
    """
    cdef Py_ssize_t n_matrices = matrices_a.shape[0]
    cdef Py_ssize_t size = matrices_a.shape[1]

    cdef Py_ssize_t m, i, j, k
    cdef double acc

    for m in range(n_matrices):
        # Cython matrix multiply
        for i in range(size):
            for j in range(size):
                acc = 0.0
                for k in range(size):
                    acc += matrices_a[m, i, k] * matrices_b[m, k, j]
                products[m, i, j] = acc

        # NumPy inversion (for stability)
        inverses[m] = np.linalg.inv(matrices_a[m])
