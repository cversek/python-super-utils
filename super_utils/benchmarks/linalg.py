"""
Small dense linear algebra benchmark - BLAS-intensive operations.

BLAS-intensive workload: many small matrix operations (4x4, 8x8).
Tests BLAS library efficiency and cache utilization.
"""

import numpy as np
from .base import BenchmarkBase


class LinearAlgebraKernel(BenchmarkBase):
    """
    Small dense matrix operations benchmark.

    Performs many small matrix multiply/invert operations.
    This is BLAS-intensive because:
    - Matrix operations use BLAS routines (dgemm, etc.)
    - Small matrices (4x4, 8x8) test BLAS overhead vs. raw computation
    - Many operations test cache efficiency

    Small matrices are common in VEP processing (covariance matrices,
    spatial filters, etc.) and benefit from different optimizations
    than large matrix ops.
    """

    name = "linalg"
    description = "Small dense matrix operations (4x4, 8x8 multiply/invert)"
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
        self._expected_inverses = None

    def run(self) -> None:
        """Execute matrix operations."""
        n_matrices = self._matrices_a.shape[0]

        # Loop over matrices
        for i in range(n_matrices):
            # Matrix multiply: C = A @ B
            self._products[i] = self._matrices_a[i] @ self._matrices_b[i]

            # Matrix inverse: A^-1
            self._inverses[i] = np.linalg.inv(self._matrices_a[i])

    def validate(self) -> bool:
        """Validate against saved expected results."""
        if self._expected_products is None:
            # Compute expected results (same operations, cached)
            self._expected_products = np.zeros_like(self._products)
            self._expected_inverses = np.zeros_like(self._inverses)

            for i in range(self._matrices_a.shape[0]):
                self._expected_products[i] = self._matrices_a[i] @ self._matrices_b[i]
                self._expected_inverses[i] = np.linalg.inv(self._matrices_a[i])

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
