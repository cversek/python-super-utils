"""
Cython small dense linear algebra benchmark - BLAS-intensive workload.

Wrapper for the Cython-compiled linalg kernel.
Falls back gracefully if Cython module is not compiled.
"""

import numpy as np
from .base import BenchmarkBase


class CythonLinalgBenchmark(BenchmarkBase):
    """
    Cython-compiled small dense matrix operations benchmark.

    This version implements direct matrix multiplication in Cython
    while using NumPy for matrix inversion (for numerical stability).

    BLAS-intensive because it performs many small matrix operations,
    testing the overhead of Cython loops vs. BLAS library calls.
    """

    name = "cython_linalg"
    description = "Cython small matrix multiply + NumPy invert (BLAS-intensive)"
    workload_type = "BLAS-intensive"

    def setup(self) -> None:
        """Initialize matrix arrays."""
        # Size selection
        size_map = {
            "small": (5000, 4),      # 5k matrices, 4x4
            "medium": (20000, 8),    # 20k matrices, 8x8
            "large": (50000, 8),     # 50k matrices, 8x8
        }
        n_matrices, matrix_size = size_map.get(self.size, size_map["medium"])

        # Generate random positive definite matrices
        # (use A @ A.T to ensure positive definite for inversion)
        random_matrices = np.random.randn(n_matrices, matrix_size, matrix_size).astype(np.float64)
        self._matrices_a = np.zeros((n_matrices, matrix_size, matrix_size), dtype=np.float64)
        for i in range(n_matrices):
            A = random_matrices[i]
            self._matrices_a[i] = A @ A.T + np.eye(matrix_size) * 0.1  # Add small diagonal for stability

        # Second set for multiplication
        self._matrices_b = np.random.randn(n_matrices, matrix_size, matrix_size).astype(np.float64)

        # Output storage
        self._products = np.zeros_like(self._matrices_a)
        self._inverses = np.zeros_like(self._matrices_a)

        # Expected results
        self._expected_products = None

        # Import Cython kernel (raises ImportError if not compiled)
        from ._cython.linalg import small_matmul_and_inv
        self._kernel = small_matmul_and_inv

    def run(self) -> None:
        """Execute Cython matrix operations."""
        self._kernel(self._matrices_a, self._matrices_b, self._products, self._inverses)

    def validate(self) -> bool:
        """Validate against NumPy operations."""
        if self._expected_products is None:
            # Compute expected products using NumPy
            self._expected_products = np.zeros_like(self._products)
            for i in range(self._matrices_a.shape[0]):
                self._expected_products[i] = self._matrices_a[i] @ self._matrices_b[i]

        # Check products
        products_match = np.allclose(self._products, self._expected_products, rtol=1e-10)

        # Check inverses (verify A @ A^-1 = I)
        identity_check = True
        for i in range(min(100, self._matrices_a.shape[0])):  # Sample check
            product = self._matrices_a[i] @ self._inverses[i]
            identity = np.eye(self._matrices_a.shape[1])
            if not np.allclose(product, identity, rtol=1e-8):
                identity_check = False
                break

        return products_match and identity_check
