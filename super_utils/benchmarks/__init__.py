"""
Cython optimization benchmarks.

Provides 5 pure Python algorithm classes and 5 Cython-compiled versions
for workload-specific optimization testing:

Pure Python:
1. streaming - Memory-bound nested loop accumulator
2. wavelet - Compute-bound wavelet decomposition/reconstruction
3. branch - Branch-heavy data-dependent conditionals
4. linalg - BLAS-intensive small dense matrix operations
5. interp - Mixed workload interpolation + filtering pipeline

Cython-compiled (require build):
6. cython_streaming - Cython memory-bound accumulator
7. cython_wavelet - Cython compute-bound wavelet-like convolution
8. cython_branch - Cython branch-heavy conditionals
9. cython_linalg - Cython BLAS-intensive matrix operations
10. cython_interp - Cython mixed workload pipeline

Usage:
    from super_utils.benchmarks import BENCHMARKS, run_benchmarks

    # Run all benchmarks
    results = run_benchmarks()

    # Run specific benchmark
    results = run_benchmarks(classes=['streaming'])

    # Run Cython benchmarks (requires compiled extensions)
    results = run_benchmarks(classes=['cython_streaming'])

    # Custom settings
    results = run_benchmarks(
        classes=['streaming', 'wavelet'],
        iterations=10,
        warmup=5,
        size='large'
    )
"""

from typing import Dict, Any, List, Optional
import datetime
import json

from .base import BenchmarkBase
from .streaming import StreamingAccumulator
from .wavelet import WaveletTransform
from .branch import BranchPredictionHeavy
from .linalg import LinearAlgebraKernel
from .interp import InterpolationFilter
from .mfvep_numpy import MFVEPNumpyBenchmark


# Registry of all available benchmarks
BENCHMARKS: Dict[str, type] = {
    'streaming': StreamingAccumulator,
    'wavelet': WaveletTransform,
    'branch': BranchPredictionHeavy,
    'linalg': LinearAlgebraKernel,
    'interp': InterpolationFilter,
    'mfvep': MFVEPNumpyBenchmark,
}

# Try to import Cython benchmarks (may not be compiled)
try:
    from .cython_streaming import CythonStreamingBenchmark
    BENCHMARKS['cython_streaming'] = CythonStreamingBenchmark
except ImportError:
    pass  # Cython not compiled

try:
    from .cython_wavelet import CythonWaveletBenchmark
    BENCHMARKS['cython_wavelet'] = CythonWaveletBenchmark
except ImportError:
    pass  # Cython not compiled

try:
    from .cython_branch import CythonBranchBenchmark
    BENCHMARKS['cython_branch'] = CythonBranchBenchmark
except ImportError:
    pass  # Cython not compiled

try:
    from .cython_linalg import CythonLinalgBenchmark
    BENCHMARKS['cython_linalg'] = CythonLinalgBenchmark
except ImportError:
    pass  # Cython not compiled

try:
    from .cython_interp import CythonInterpBenchmark
    BENCHMARKS['cython_interp'] = CythonInterpBenchmark
except ImportError:
    pass  # Cython not compiled

try:
    from .cython_mfvep import CythonMFVEPBenchmark
    BENCHMARKS['cython_mfvep'] = CythonMFVEPBenchmark
except ImportError:
    pass  # Cython mfvep_kernel not compiled


def list_benchmarks() -> Dict[str, Dict[str, str]]:
    """
    Get metadata for all available benchmarks.

    Returns:
        Dict mapping benchmark name to metadata dict with keys:
        - name: Short identifier
        - description: Human-readable description
        - workload_type: Workload classification
    """
    return {
        name: {
            'name': cls.name,
            'description': cls.description,
            'workload_type': cls.workload_type,
        }
        for name, cls in BENCHMARKS.items()
    }


def run_benchmarks(
    classes: Optional[List[str]] = None,
    iterations: int = 5,
    warmup: int = 3,
    size: str = "medium",
    profile: str = "conservative",
    include_system_info: bool = True,
) -> Dict[str, Any]:
    """
    Run benchmarks and return results.

    Args:
        classes: List of benchmark names to run (default: all)
        iterations: Number of timed iterations per benchmark
        warmup: Number of warmup iterations
        size: Problem size ("small", "medium", "large")
        profile: Optimization profile for metadata ("conservative" or "aggressive")
        include_system_info: Include system spec and compiler flags in results

    Returns:
        Dict with structure:
        {
            'timestamp': ISO timestamp,
            'system': {...} if include_system_info else None,
            'profile': profile name,
            'flags': [...] if include_system_info else None,
            'settings': {'iterations': N, 'warmup': M, 'size': S},
            'results': {
                'benchmark_name': {
                    'mean_ms': float,
                    'std_ms': float,
                    'min_ms': float,
                    'max_ms': float,
                    'valid': bool,
                    'workload_type': str
                },
                ...
            },
            'recommendation': str or None
        }

    Raises:
        ValueError: If invalid benchmark name in classes list
        ImportError: If benchmark has missing dependencies (e.g., pywavelets)
    """
    # Default to all benchmarks
    if classes is None:
        classes = list(BENCHMARKS.keys())

    # Validate benchmark names
    invalid = set(classes) - set(BENCHMARKS.keys())
    if invalid:
        raise ValueError(f"Unknown benchmark(s): {', '.join(invalid)}")

    # System info (if requested)
    system_info = None
    compiler_flags = None
    if include_system_info:
        from ..system_spec import get_system_spec
        from ..cython_optimizer import get_optimal_compile_args

        spec = get_system_spec()
        opts = get_optimal_compile_args(spec=spec, profile=profile)

        system_info = {
            'cpu': spec.get('hardware', {}).get('cpu_model', 'unknown'),
            'cores': spec.get('hardware', {}).get('cpu_cores_logical', '?'),
            'arch': spec.get('hardware', {}).get('architecture', 'unknown'),
        }
        compiler_flags = opts.get('extra_compile_args', [])

    # Run benchmarks
    results = {}
    errors = {}

    for name in classes:
        try:
            benchmark_cls = BENCHMARKS[name]
            benchmark = benchmark_cls(size=size)
            result = benchmark.benchmark(iterations=iterations, warmup=warmup)
            results[name] = result
        except ImportError as e:
            errors[name] = f"Missing dependency: {e}"
        except Exception as e:
            errors[name] = f"Error: {e}"

    # Generate recommendation
    recommendation = _generate_recommendation(results)

    # Build output
    output = {
        'timestamp': datetime.datetime.now().isoformat(),
        'system': system_info,
        'profile': profile,
        'flags': compiler_flags,
        'settings': {
            'iterations': iterations,
            'warmup': warmup,
            'size': size,
        },
        'results': results,
        'recommendation': recommendation,
    }

    if errors:
        output['errors'] = errors

    return output


def _generate_recommendation(results: Dict[str, Dict[str, Any]]) -> Optional[str]:
    """
    Generate optimization recommendation based on benchmark results.

    Args:
        results: Dict of benchmark results

    Returns:
        Recommendation string or None if insufficient data
    """
    if not results:
        return None

    # Categorize by workload type
    workload_times = {}
    for name, result in results.items():
        wtype = result.get('workload_type', 'unknown')
        time_ms = result.get('mean_ms', 0)
        if wtype not in workload_times:
            workload_times[wtype] = []
        workload_times[wtype].append((name, time_ms))

    # Find dominant workload
    workload_totals = {
        wtype: sum(t for _, t in times)
        for wtype, times in workload_times.items()
    }

    if not workload_totals:
        return None

    dominant_type = max(workload_totals.keys(), key=lambda k: workload_totals[k])
    dominant_total = workload_totals[dominant_type]
    total_time = sum(workload_totals.values())
    dominant_pct = (dominant_total / total_time) * 100 if total_time > 0 else 0

    # Generate recommendation
    recommendations = {
        'memory-bound': (
            "Your workload appears memory-bound. Consider:\n"
            "  • Streaming/accumulator patterns (minimize intermediate allocations)\n"
            "  • Cache-friendly data layouts (structure of arrays)\n"
            "  • Profile may have limited impact vs. algorithmic changes"
        ),
        'compute-bound': (
            "Your workload appears compute-bound. The 'aggressive' profile may yield +15-20% speedup.\n"
            "  • -ffast-math can significantly accelerate FP operations\n"
            "  • Verify numerical correctness with aggressive optimizations\n"
            "  • Consider Cython with typed memoryviews for hot loops"
        ),
        'branch-heavy': (
            "Your workload has heavy branching. Compiler optimizations can help:\n"
            "  • -O3 enables if-conversion and branch elimination\n"
            "  • Consider restructuring as vectorized operations (branchless)\n"
            "  • Profile-guided optimization (PGO) may help with branch prediction"
        ),
        'BLAS-intensive': (
            "Your workload is BLAS-intensive. NumPy optimizations matter most:\n"
            "  • Ensure NumPy linked against optimized BLAS (MKL, OpenBLAS, Accelerate)\n"
            "  • Compiler flags have limited impact (BLAS is already optimized)\n"
            "  • Check BLAS threading: export OMP_NUM_THREADS=<cores>"
        ),
        'mixed': (
            "Your workload is mixed (computation + memory + I/O).\n"
            "  • Conservative profile is recommended (balanced)\n"
            "  • Profile individual stages to identify bottlenecks\n"
            "  • Consider pipeline parallelism if stages are independent"
        ),
    }

    base_rec = recommendations.get(dominant_type, "Unable to determine dominant workload.")
    return f"Dominant workload: {dominant_type} ({dominant_pct:.1f}% of total time)\n{base_rec}"


def save_results(results: Dict[str, Any], filepath: str) -> None:
    """
    Save benchmark results to JSON file.

    Args:
        results: Results dict from run_benchmarks()
        filepath: Output path
    """
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
