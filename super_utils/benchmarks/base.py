"""
Base class for Cython optimization benchmarks.

All benchmark classes inherit from BenchmarkBase and implement:
- setup(): Initialize test data
- run(): Execute the benchmark kernel
- validate(): Check numerical correctness
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import time


class BenchmarkBase(ABC):
    """
    Abstract base class for Cython benchmarks.

    Attributes:
        name: Short identifier for the benchmark
        description: Human-readable description
        workload_type: One of "memory-bound", "compute-bound", "branch-heavy", "mixed"
    """

    name: str = ""
    description: str = ""
    workload_type: str = ""

    def __init__(self, size: str = "medium"):
        """
        Initialize benchmark.

        Args:
            size: Problem size - "small", "medium", or "large"
        """
        self.size = size
        self._data = None

    @abstractmethod
    def setup(self) -> None:
        """Initialize test data. Called once before iterations."""
        pass

    @abstractmethod
    def run(self) -> None:
        """Execute the benchmark kernel. Called multiple times."""
        pass

    @abstractmethod
    def validate(self) -> bool:
        """Check numerical correctness. Return True if valid."""
        pass

    def benchmark(self, iterations: int = 5, warmup: int = 3) -> Dict[str, Any]:
        """
        Run benchmark with warmup, return timing stats.

        Args:
            iterations: Number of timed runs
            warmup: Number of warmup runs (not timed)

        Returns:
            Dict with keys: mean_ms, std_ms, min_ms, max_ms, valid, workload_type
        """
        self.setup()

        # Warmup runs
        for _ in range(warmup):
            self.run()

        # Timed runs
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            self.run()
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # Convert to ms

        # Compute statistics
        mean = sum(times) / len(times)
        variance = sum((t - mean) ** 2 for t in times) / len(times)
        std = variance ** 0.5

        return {
            "mean_ms": mean,
            "std_ms": std,
            "min_ms": min(times),
            "max_ms": max(times),
            "valid": self.validate(),
            "workload_type": self.workload_type,
        }
