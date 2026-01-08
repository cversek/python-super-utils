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

from ..memory import MEMORY_SNAPSHOT, MEMORY_RESET, MEMORY_PEAK_START, MEMORY_PEAK_STOP


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

    def benchmark(self, iterations: int = 5, warmup: int = 3,
                  track_memory: bool = True, sample_interval_ms: int = 10) -> Dict[str, Any]:
        """
        Run benchmark with warmup, return timing and memory stats.

        Args:
            iterations: Number of timed runs
            warmup: Number of warmup runs (not timed)
            track_memory: Enable memory tracking via super_utils.memory
            sample_interval_ms: Interval for peak memory sampling (default: 10ms)

        Returns:
            Dict with keys:
                - mean_ms, std_ms, min_ms, max_ms: timing stats
                - valid, workload_type: validation and classification
                - setup_memory_mb: Memory allocated during setup()
                - run_peak_mb: Peak memory delta during benchmark iterations
                - total_peak_mb: Total peak memory from start to finish
        """
        # Memory tracking setup
        if track_memory:
            MEMORY_RESET()
            snap_start = MEMORY_SNAPSHOT("benchmark_start")
            baseline_rss = snap_start['rss'] if snap_start else 0

        self.setup()

        if track_memory:
            snap_setup = MEMORY_SNAPSHOT("after_setup")
            setup_rss = snap_setup['rss'] if snap_setup else baseline_rss

        # Warmup runs
        for _ in range(warmup):
            self.run()

        # Timed runs with threaded peak memory tracking
        times = []
        run_peak_mb = 0.0

        if track_memory:
            # Start background thread to capture peak memory during iterations
            MEMORY_PEAK_START("benchmark_iterations", sample_interval_ms=sample_interval_ms)

        for i in range(iterations):
            start = time.perf_counter()
            self.run()
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # Convert to ms

        if track_memory:
            # Stop tracker and get peak
            peak_result = MEMORY_PEAK_STOP("benchmark_iterations")
            run_peak_mb = peak_result['peak_mb']

        # Compute timing statistics
        mean = sum(times) / len(times)
        variance = sum((t - mean) ** 2 for t in times) / len(times)
        std = variance ** 0.5

        result = {
            "mean_ms": mean,
            "std_ms": std,
            "min_ms": min(times),
            "max_ms": max(times),
            "valid": self.validate(),
            "workload_type": self.workload_type,
        }

        # Add memory metrics
        if track_memory:
            bytes_to_mb = 1 / (1024 * 1024)
            setup_memory_mb = (setup_rss - baseline_rss) * bytes_to_mb
            result["setup_memory_mb"] = setup_memory_mb
            result["run_peak_mb"] = run_peak_mb  # From threaded tracker
            result["total_peak_mb"] = setup_memory_mb + run_peak_mb

        return result
