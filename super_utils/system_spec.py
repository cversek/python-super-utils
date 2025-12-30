"""
system_spec.py
==============

System specification utilities for capturing hardware, OS, compiler, and
numerical backend information. Enables reproducible benchmarking by recording
the exact system configuration under which performance measurements were taken.

Functions:
----------
- get_system_spec: Capture comprehensive system specification.
- get_hardware_spec: Hardware info (CPU, memory, architecture).
- get_compiler_spec: Compiler and Cython configuration.
- get_numpy_spec: NumPy and BLAS backend info.
- print_system_report: Rich-formatted console display.
- export_system_spec: Save specification to JSON file.
"""
# standard imports
import os
import sys
import platform
import sysconfig
import subprocess
import json
import re
from datetime import datetime
from typing import Optional, Dict, Any, List

# third party imports
from rich.console import Console
from rich.table import Table
from rich.panel import Panel


# Module-level cache for expensive operations
_cached_spec: Optional[Dict[str, Any]] = None
_spec_version = "1.0"


def _parse_proc_cpuinfo() -> Dict[str, Any]:
    """Parse /proc/cpuinfo for CPU details (Linux only)."""
    info = {
        "model": None,
        "vendor": None,
        "cores_physical": None,
        "cores_logical": None,
        "frequency_mhz": None,
        "features": [],
        "cache_size_kb": None,
    }

    try:
        with open('/proc/cpuinfo', 'r') as f:
            content = f.read()

        # Parse first processor block for model info
        lines = content.split('\n')
        physical_ids = set()
        core_count = 0

        for line in lines:
            if ':' not in line:
                continue
            key, value = line.split(':', 1)
            key = key.strip().lower()
            value = value.strip()

            if key == 'model name' and info["model"] is None:
                info["model"] = value
            elif key == 'vendor_id' and info["vendor"] is None:
                info["vendor"] = value
            elif key == 'cpu mhz' and info["frequency_mhz"] is None:
                try:
                    info["frequency_mhz"] = float(value)
                except ValueError:
                    pass
            elif key == 'cache size' and info["cache_size_kb"] is None:
                # Usually "6144 KB"
                match = re.match(r'(\d+)', value)
                if match:
                    info["cache_size_kb"] = int(match.group(1))
            elif key == 'flags' and not info["features"]:
                info["features"] = value.split()
            elif key == 'physical id':
                physical_ids.add(value)
            elif key == 'processor':
                core_count += 1

        info["cores_logical"] = core_count
        info["cores_physical"] = len(physical_ids) if physical_ids else None

    except (FileNotFoundError, PermissionError):
        pass

    return info


def _get_memory_info() -> Dict[str, Any]:
    """Get system memory information."""
    info = {
        "total_bytes": None,
        "available_bytes": None,
        "total_gb": None,
        "available_gb": None,
    }

    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    kb = int(line.split()[1])
                    info["total_bytes"] = kb * 1024
                    info["total_gb"] = round(kb / (1024 * 1024), 2)
                elif line.startswith('MemAvailable:'):
                    kb = int(line.split()[1])
                    info["available_bytes"] = kb * 1024
                    info["available_gb"] = round(kb / (1024 * 1024), 2)
    except (FileNotFoundError, PermissionError):
        pass

    return info


def _detect_container() -> Optional[str]:
    """Detect if running in a container."""
    # Check for Docker
    if os.path.exists('/.dockerenv'):
        return "docker"

    # Check cgroup for docker/podman
    try:
        with open('/proc/1/cgroup', 'r') as f:
            content = f.read()
            if 'docker' in content:
                return "docker"
            if 'podman' in content:
                return "podman"
    except (FileNotFoundError, PermissionError):
        pass

    # Check for container environment variable
    if os.environ.get('container'):
        return os.environ.get('container')

    return None


def _get_linux_distribution() -> Optional[str]:
    """Get Linux distribution info."""
    try:
        # Try /etc/os-release (modern)
        if os.path.exists('/etc/os-release'):
            with open('/etc/os-release', 'r') as f:
                for line in f:
                    if line.startswith('PRETTY_NAME='):
                        return line.split('=', 1)[1].strip().strip('"')

        # Fallback to platform
        return platform.platform()
    except Exception:
        return None


def _resolve_march_native(compiler: str) -> Optional[str]:
    """
    Resolve -march=native to actual architecture by running compiler.

    Args:
        compiler: Compiler command (e.g., 'gcc', 'clang')

    Returns:
        Architecture name (e.g., 'skylake', 'znver2') or None
    """
    if not compiler:
        return None

    # Extract just the compiler name (gcc, clang, etc.)
    cc_name = compiler.split()[0] if compiler else None
    if not cc_name:
        return None

    try:
        # Run: gcc -march=native -Q --help=target | grep march
        result = subprocess.run(
            [cc_name, '-march=native', '-Q', '--help=target'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if '-march=' in line and 'native' not in line:
                    # Line like "  -march=                    	skylake"
                    parts = line.split()
                    if len(parts) >= 2:
                        return parts[-1]

    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass

    return None


def _get_compiler_version(compiler: str) -> Optional[str]:
    """Get compiler version string."""
    if not compiler:
        return None

    cc_name = compiler.split()[0] if compiler else None
    if not cc_name:
        return None

    try:
        result = subprocess.run(
            [cc_name, '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # First line usually contains version
            first_line = result.stdout.split('\n')[0]
            return first_line
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass

    return None


def _get_numpy_blas_info() -> Dict[str, Any]:
    """Get NumPy and BLAS backend information."""
    info = {
        "numpy_version": None,
        "blas_library": None,
        "blas_info": None,
        "lapack_info": None,
    }

    try:
        import numpy as np
        info["numpy_version"] = np.__version__

        # Try to get BLAS info
        # NumPy 1.x uses __config__, NumPy 2.x may differ
        if hasattr(np, '__config__'):
            config = np.__config__

            # Try blas_opt_info or blas_info
            for attr in ['blas_opt_info', 'blas_info', 'blas_mkl_info', 'openblas_info']:
                if hasattr(config, attr):
                    blas = getattr(config, attr)
                    if blas:
                        info["blas_info"] = str(blas)
                        # Try to identify library
                        blas_str = str(blas).lower()
                        if 'mkl' in blas_str:
                            info["blas_library"] = "MKL"
                        elif 'openblas' in blas_str:
                            info["blas_library"] = "OpenBLAS"
                        elif 'atlas' in blas_str:
                            info["blas_library"] = "ATLAS"
                        break

            # Try lapack info
            for attr in ['lapack_opt_info', 'lapack_info']:
                if hasattr(config, attr):
                    lapack = getattr(config, attr)
                    if lapack:
                        info["lapack_info"] = str(lapack)
                        break

        # Alternative: try show_config (captures stdout)
        if info["blas_library"] is None:
            import io
            import contextlib

            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                np.show_config()
            config_str = f.getvalue().lower()

            if 'mkl' in config_str:
                info["blas_library"] = "MKL"
            elif 'openblas' in config_str:
                info["blas_library"] = "OpenBLAS"
            elif 'atlas' in config_str:
                info["blas_library"] = "ATLAS"
            elif 'blas' in config_str:
                info["blas_library"] = "Generic BLAS"

    except ImportError:
        pass
    except Exception:
        pass

    return info


def get_hardware_spec() -> Dict[str, Any]:
    """
    Get hardware specification.

    Returns:
        Dict with CPU, memory, and architecture info.

    Example:
        >>> spec = get_hardware_spec()
        >>> print(spec['cpu_model'])
        'Intel(R) Core(TM) i7-10700K CPU @ 3.80GHz'
    """
    cpu_info = _parse_proc_cpuinfo()
    mem_info = _get_memory_info()

    # Filter to performance-relevant CPU features
    perf_features = []
    all_features = cpu_info.get("features", [])
    relevant = ['sse', 'sse2', 'sse3', 'ssse3', 'sse4_1', 'sse4_2',
                'avx', 'avx2', 'avx512f', 'avx512vl', 'fma', 'neon']
    for feat in all_features:
        if feat.lower() in relevant:
            perf_features.append(feat)

    return {
        "cpu_model": cpu_info.get("model") or platform.processor() or "unknown",
        "cpu_vendor": cpu_info.get("vendor"),
        "cpu_cores_physical": cpu_info.get("cores_physical"),
        "cpu_cores_logical": cpu_info.get("cores_logical") or os.cpu_count(),
        "cpu_frequency_mhz": cpu_info.get("frequency_mhz"),
        "cpu_features": perf_features,
        "cpu_cache_kb": cpu_info.get("cache_size_kb"),
        "memory_total_gb": mem_info.get("total_gb"),
        "memory_available_gb": mem_info.get("available_gb"),
        "architecture": platform.machine(),
    }


def get_compiler_spec() -> Dict[str, Any]:
    """
    Get compiler and Cython specification.

    Returns:
        Dict with compiler name, version, flags, and Cython version.

    Example:
        >>> spec = get_compiler_spec()
        >>> print(spec['cc'])
        'gcc'
    """
    cc = sysconfig.get_config_var('CC')
    cflags = sysconfig.get_config_var('CFLAGS')
    opt = sysconfig.get_config_var('OPT')

    # Get Cython version
    cython_version = None
    try:
        import Cython
        cython_version = Cython.__version__
    except ImportError:
        pass

    # Resolve -march=native
    march_resolved = None
    if cflags and '-march=native' in cflags:
        march_resolved = _resolve_march_native(cc)

    return {
        "cc": cc,
        "cc_version": _get_compiler_version(cc),
        "cflags": cflags,
        "opt": opt,
        "march_resolved": march_resolved,
        "cython_version": cython_version,
    }


def get_numpy_spec() -> Dict[str, Any]:
    """
    Get NumPy and BLAS backend specification.

    Returns:
        Dict with NumPy version, BLAS library, and threading config.

    Example:
        >>> spec = get_numpy_spec()
        >>> print(spec['blas_library'])
        'OpenBLAS'
    """
    numpy_info = _get_numpy_blas_info()

    return {
        "numpy_version": numpy_info.get("numpy_version"),
        "blas_library": numpy_info.get("blas_library"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
    }


def get_system_spec(use_cache: bool = True) -> Dict[str, Any]:
    """
    Get comprehensive system specification.

    Combines hardware, OS, Python, compiler, and NumPy specs into
    a single dictionary suitable for JSON export and profiling reports.

    Parameters:
        use_cache: If True, return cached spec if available.

    Returns:
        Dict with all system specification categories.

    Example:
        >>> spec = get_system_spec()
        >>> print(json.dumps(spec, indent=2))
    """
    global _cached_spec

    if use_cache and _cached_spec is not None:
        return _cached_spec

    spec = {
        "hardware": get_hardware_spec(),
        "os": {
            "platform": platform.system(),
            "kernel": platform.release(),
            "distribution": _get_linux_distribution(),
            "container": _detect_container(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "build": platform.python_build()[0],
            "executable": sys.executable,
            "venv": os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_PREFIX"),
        },
        "compiler": get_compiler_spec(),
        "numpy": get_numpy_spec(),
        "timestamp": datetime.now().isoformat(),
        "spec_version": _spec_version,
    }

    _cached_spec = spec
    return spec


def print_system_report(spec: Optional[Dict[str, Any]] = None):
    """
    Print Rich-formatted system specification report.

    Parameters:
        spec: System spec dict (if None, calls get_system_spec()).

    Example:
        >>> print_system_report()
    """
    if spec is None:
        spec = get_system_spec()

    console = Console()

    # Hardware table
    hw = spec.get("hardware", {})
    hw_table = Table(title="Hardware", show_header=True, header_style="bold cyan")
    hw_table.add_column("Property", style="cyan", width=20)
    hw_table.add_column("Value", style="white")

    hw_table.add_row("CPU Model", str(hw.get("cpu_model", "unknown")))
    hw_table.add_row("Cores (logical)", str(hw.get("cpu_cores_logical", "?")))
    hw_table.add_row("Frequency (MHz)", str(hw.get("cpu_frequency_mhz", "?")))
    hw_table.add_row("Architecture", str(hw.get("architecture", "?")))
    hw_table.add_row("Memory (GB)", str(hw.get("memory_total_gb", "?")))

    features = hw.get("cpu_features", [])
    if features:
        # Highlight AVX/AVX2 in green
        feat_str = ", ".join(features)
        hw_table.add_row("CPU Features", feat_str)

    console.print(hw_table)
    console.print()

    # OS table
    os_info = spec.get("os", {})
    os_table = Table(title="Operating System", show_header=True, header_style="bold cyan")
    os_table.add_column("Property", style="cyan", width=20)
    os_table.add_column("Value", style="white")

    os_table.add_row("Platform", str(os_info.get("platform", "?")))
    os_table.add_row("Kernel", str(os_info.get("kernel", "?")))
    os_table.add_row("Distribution", str(os_info.get("distribution", "?")))
    if os_info.get("container"):
        os_table.add_row("Container", f"[yellow]{os_info.get('container')}[/yellow]")

    console.print(os_table)
    console.print()

    # Compiler table
    comp = spec.get("compiler", {})
    comp_table = Table(title="Compiler / Cython", show_header=True, header_style="bold cyan")
    comp_table.add_column("Property", style="cyan", width=20)
    comp_table.add_column("Value", style="white")

    comp_table.add_row("Compiler", str(comp.get("cc", "?")))
    if comp.get("march_resolved"):
        comp_table.add_row("Architecture Target", f"[green]{comp.get('march_resolved')}[/green]")
    if comp.get("cython_version"):
        comp_table.add_row("Cython Version", str(comp.get("cython_version")))

    console.print(comp_table)
    console.print()

    # NumPy table
    np_info = spec.get("numpy", {})
    np_table = Table(title="NumPy / BLAS", show_header=True, header_style="bold cyan")
    np_table.add_column("Property", style="cyan", width=20)
    np_table.add_column("Value", style="white")

    np_table.add_row("NumPy Version", str(np_info.get("numpy_version", "?")))
    blas = np_info.get("blas_library")
    if blas:
        color = "green" if blas in ["MKL", "OpenBLAS"] else "yellow"
        np_table.add_row("BLAS Library", f"[{color}]{blas}[/{color}]")
    if np_info.get("omp_num_threads"):
        np_table.add_row("OMP_NUM_THREADS", str(np_info.get("omp_num_threads")))

    console.print(np_table)
    console.print()


def export_system_spec(filepath: str, spec: Optional[Dict[str, Any]] = None):
    """
    Export system specification to JSON file.

    Parameters:
        filepath: Output file path.
        spec: System spec dict (if None, calls get_system_spec()).

    Example:
        >>> export_system_spec("system_spec.json")
    """
    if spec is None:
        spec = get_system_spec()

    with open(filepath, 'w') as f:
        json.dump(spec, f, indent=2)


def clear_cache():
    """Clear the cached system specification."""
    global _cached_spec
    _cached_spec = None
