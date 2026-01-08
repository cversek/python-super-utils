"""
system_spec.py
==============

Cross-platform system specification utilities for capturing hardware, OS,
compiler, and numerical backend information. Enables reproducible benchmarking
by recording the exact system configuration under which performance
measurements were taken.

Supports Linux (x86_64, aarch64) and macOS (Intel, Apple Silicon).

Functions:
----------
Core:
- get_system_spec: Capture comprehensive system specification.
- get_hardware_spec: Hardware info (CPU, memory, architecture) - cross-platform.
- get_compiler_spec: Compiler and Cython configuration.
- get_numpy_spec: NumPy and BLAS backend info.

Platform-specific (detailed topology):
- get_apple_silicon_info: Apple M-series details (P/E cores, cache hierarchy).
- get_linux_cpu_info: Linux CPU topology (sockets, threads, cache).

Output:
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
_spec_version = "1.1"  # Added macOS Apple Silicon + Linux enhanced detection


# =============================================================================
# Platform-specific helpers
# =============================================================================

def _run_sysctl(key: str) -> Optional[str]:
    """
    Run sysctl -n <key> and return the value (macOS/BSD).

    Args:
        key: sysctl key (e.g., 'machdep.cpu.brand_string')

    Returns:
        Value string or None if unavailable
    """
    try:
        result = subprocess.run(
            ['sysctl', '-n', key],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    return None


def _get_macos_hardware_spec() -> Dict[str, Any]:
    """
    Get hardware specification on macOS using sysctl.

    Returns:
        Dict with CPU, memory, and architecture info.
    """
    info = {
        "cpu_model": None,
        "cpu_vendor": "Apple" if platform.machine() == "arm64" else None,
        "cpu_cores_physical": None,
        "cpu_cores_logical": None,
        "cpu_frequency_mhz": None,
        "cpu_features": [],
        "cpu_cache_kb": None,
        "memory_total_gb": None,
        "memory_available_gb": None,
        "architecture": platform.machine(),
    }

    # CPU model
    brand = _run_sysctl('machdep.cpu.brand_string')
    if brand:
        info["cpu_model"] = brand

    # Core counts
    logical = _run_sysctl('hw.ncpu')
    if logical:
        info["cpu_cores_logical"] = int(logical)

    physical = _run_sysctl('hw.physicalcpu')
    if physical:
        info["cpu_cores_physical"] = int(physical)

    # Memory (hw.memsize returns bytes)
    memsize = _run_sysctl('hw.memsize')
    if memsize:
        mem_bytes = int(memsize)
        info["memory_total_gb"] = round(mem_bytes / (1024**3), 2)

    # CPU frequency (not available on Apple Silicon, but try)
    freq = _run_sysctl('hw.cpufrequency')
    if freq:
        info["cpu_frequency_mhz"] = round(int(freq) / 1_000_000, 1)

    # Cache size (L2 or L3 depending on availability)
    # Try L3 first, then L2
    for cache_key in ['hw.l3cachesize', 'hw.l2cachesize']:
        cache = _run_sysctl(cache_key)
        if cache and int(cache) > 0:
            info["cpu_cache_kb"] = int(cache) // 1024
            break

    # CPU features on macOS/ARM64
    if platform.machine() == "arm64":
        info["cpu_features"] = ["neon", "fp16", "dotprod"]  # Standard ARM64 features
    else:
        # Intel Mac - try to get features
        features = _run_sysctl('machdep.cpu.features')
        if features:
            all_features = features.lower().split()
            relevant = ['sse', 'sse2', 'sse3', 'ssse3', 'sse4_1', 'sse4_2',
                       'avx', 'avx2', 'avx512f', 'fma']
            info["cpu_features"] = [f for f in all_features if f in relevant]

    return info


def get_apple_silicon_info() -> Optional[Dict[str, Any]]:
    """
    Get Apple Silicon specific information.

    Returns None on non-Apple Silicon platforms.
    Returns dict with M-series specific details on Apple Silicon:
        - chip_name: Full chip name (e.g., "Apple M3 Max")
        - generation: M-series generation (e.g., "M3")
        - variant: Chip variant (e.g., "Max", "Pro", "Ultra", or None for base)
        - performance_cores: Number of performance (P) cores
        - efficiency_cores: Number of efficiency (E) cores
        - cache_performance: Dict with L1I, L1D, L2 sizes for P-cores (bytes)
        - cache_efficiency: Dict with L1I, L1D, L2 sizes for E-cores (bytes)
        - cache_line_bytes: Cache line size
        - is_rosetta: True if running under Rosetta translation

    Example:
        >>> info = get_apple_silicon_info()
        >>> if info:
        ...     print(f"{info['chip_name']}: {info['performance_cores']}P + {info['efficiency_cores']}E")
        Apple M3 Max: 12P + 4E
    """
    # Check if macOS and ARM64
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return None

    info = {
        "chip_name": None,
        "generation": None,
        "variant": None,
        "performance_cores": None,
        "efficiency_cores": None,
        "cache_performance": {},
        "cache_efficiency": {},
        "cache_line_bytes": None,
        "is_rosetta": False,
    }

    # Chip name and generation
    brand = _run_sysctl('machdep.cpu.brand_string')
    if brand:
        info["chip_name"] = brand

        # Parse generation (M1, M2, M3, M4, etc.)
        for gen in ['M4', 'M3', 'M2', 'M1']:
            if gen in brand:
                info["generation"] = gen
                break

        # Parse variant
        brand_upper = brand.upper()
        if 'ULTRA' in brand_upper:
            info["variant"] = "Ultra"
        elif 'MAX' in brand_upper:
            info["variant"] = "Max"
        elif 'PRO' in brand_upper:
            info["variant"] = "Pro"
        # else: base model, variant = None

    # Performance cores (perflevel0)
    perf = _run_sysctl('hw.perflevel0.physicalcpu')
    if perf:
        info["performance_cores"] = int(perf)

    # Efficiency cores (perflevel1)
    eff = _run_sysctl('hw.perflevel1.physicalcpu')
    if eff:
        info["efficiency_cores"] = int(eff)

    # Cache line size
    cache_line = _run_sysctl('hw.cachelinesize')
    if cache_line:
        info["cache_line_bytes"] = int(cache_line)

    # Performance core caches
    for key, sysctl_key in [
        ('l1i', 'hw.perflevel0.l1icachesize'),
        ('l1d', 'hw.perflevel0.l1dcachesize'),
        ('l2', 'hw.perflevel0.l2cachesize'),
    ]:
        val = _run_sysctl(sysctl_key)
        if val:
            info["cache_performance"][key] = int(val)

    # Efficiency core caches
    for key, sysctl_key in [
        ('l1i', 'hw.perflevel1.l1icachesize'),
        ('l1d', 'hw.perflevel1.l1dcachesize'),
        ('l2', 'hw.perflevel1.l2cachesize'),
    ]:
        val = _run_sysctl(sysctl_key)
        if val:
            info["cache_efficiency"][key] = int(val)

    # Rosetta detection
    rosetta = _run_sysctl('sysctl.proc_translated')
    if rosetta:
        info["is_rosetta"] = rosetta == '1'

    return info


def detect_performance_cores() -> Dict[str, Any]:
    """
    Detect Performance (P) and Efficiency (E) core counts on heterogeneous CPUs.

    Supports:
    - Apple Silicon (M1/M2/M3/M4) via sysctl
    - Intel hybrid (Alder Lake+) via frequency grouping
    - Linux ARM big.LITTLE via frequency grouping

    Environment variable override:
        NEUROVEP_P_CORES=12  # Force specific P-core count (useful in containers)

    Returns:
        Dict with keys:
            - performance_cores: Number of P-cores (or total if homogeneous)
            - efficiency_cores: Number of E-cores (0 if homogeneous)
            - is_hybrid: True if heterogeneous architecture detected
            - detection_method: How cores were detected

    Example:
        >>> info = detect_performance_cores()
        >>> print(f"{info['performance_cores']}P + {info['efficiency_cores']}E")
        12P + 4E  # Apple M3 Max
    """
    result = {
        "performance_cores": None,
        "efficiency_cores": 0,
        "is_hybrid": False,
        "detection_method": "fallback",
    }

    # Check environment variable override first
    env_p_cores = os.environ.get('NEUROVEP_P_CORES')
    if env_p_cores:
        try:
            p_cores = int(env_p_cores)
            if p_cores > 0:
                result["performance_cores"] = p_cores
                result["efficiency_cores"] = max(0, (os.cpu_count() or p_cores) - p_cores)
                result["is_hybrid"] = result["efficiency_cores"] > 0
                result["detection_method"] = "env_override"
                return result
        except ValueError:
            pass  # Invalid value, continue with auto-detection

    system = platform.system()

    if system == "Darwin":
        # macOS: Use sysctl for Apple Silicon P/E cores
        perf = _run_sysctl('hw.perflevel0.physicalcpu')
        eff = _run_sysctl('hw.perflevel1.physicalcpu')

        if perf:
            result["performance_cores"] = int(perf)
            result["efficiency_cores"] = int(eff) if eff else 0
            result["is_hybrid"] = result["efficiency_cores"] > 0
            result["detection_method"] = "macos_sysctl"
            return result

    elif system == "Linux":
        # Linux: Detect heterogeneous cores by frequency grouping
        try:
            freqs = []
            cpu_id = 0
            while True:
                freq_path = f"/sys/devices/system/cpu/cpu{cpu_id}/cpufreq/scaling_max_freq"
                if not os.path.exists(freq_path):
                    break
                with open(freq_path) as f:
                    freqs.append(int(f.read().strip()))
                cpu_id += 1

            if freqs and len(set(freqs)) > 1:
                # Heterogeneous: P-cores have highest frequency
                max_freq = max(freqs)
                p_cores = sum(1 for f in freqs if f == max_freq)
                e_cores = len(freqs) - p_cores

                result["performance_cores"] = p_cores
                result["efficiency_cores"] = e_cores
                result["is_hybrid"] = True
                result["detection_method"] = "linux_freq_grouping"
                return result
            elif freqs:
                # Homogeneous: all cores are "performance"
                result["performance_cores"] = len(freqs)
                result["detection_method"] = "linux_homogeneous"
                return result
        except (IOError, ValueError):
            pass

    # Fallback: use total CPU count as P-cores
    result["performance_cores"] = os.cpu_count() or 1
    result["detection_method"] = "fallback"
    return result


def _parse_lscpu() -> Dict[str, Any]:
    """
    Parse lscpu output for detailed CPU topology (Linux).

    Returns richer info than /proc/cpuinfo on modern systems.
    """
    info = {}
    try:
        result = subprocess.run(
            ['lscpu'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if ':' not in line:
                    continue
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()

                if key == 'Vendor ID':
                    info['vendor'] = value
                elif key == 'Model name':
                    info['model_name'] = value
                elif key == 'CPU(s)':
                    info['cpus_total'] = int(value)
                elif key == 'Thread(s) per core':
                    info['threads_per_core'] = int(value)
                elif key == 'Core(s) per socket':
                    info['cores_per_socket'] = int(value)
                elif key == 'Socket(s)':
                    try:
                        info['sockets'] = int(value)
                    except ValueError:
                        pass
                elif key == 'CPU max MHz':
                    info['max_mhz'] = float(value)
                elif key == 'CPU min MHz':
                    info['min_mhz'] = float(value)
                elif key == 'L1d cache':
                    info['l1d_cache'] = value
                elif key == 'L1i cache':
                    info['l1i_cache'] = value
                elif key == 'L2 cache':
                    info['l2_cache'] = value
                elif key == 'L3 cache':
                    info['l3_cache'] = value
                elif key == 'Flags':
                    info['flags'] = value.split()
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    return info


def _get_linux_cache_from_sysfs() -> Dict[str, Any]:
    """
    Get cache information from /sys/devices/system/cpu (Linux).

    Returns detailed cache hierarchy when available.
    """
    cache_info = {}
    try:
        # Check cpu0's cache directory
        cache_base = '/sys/devices/system/cpu/cpu0/cache'
        if os.path.exists(cache_base):
            import glob
            for cache_dir in sorted(glob.glob(f'{cache_base}/index*')):
                try:
                    with open(f'{cache_dir}/level', 'r') as f:
                        level = f.read().strip()
                    with open(f'{cache_dir}/type', 'r') as f:
                        cache_type = f.read().strip()
                    with open(f'{cache_dir}/size', 'r') as f:
                        size = f.read().strip()

                    key = f'L{level}_{cache_type[0].lower()}'  # e.g., L1_d, L2_u
                    cache_info[key] = size
                except (FileNotFoundError, IOError):
                    pass
    except Exception:
        pass
    return cache_info


def get_linux_cpu_info() -> Optional[Dict[str, Any]]:
    """
    Get detailed Linux CPU information (topology, cache, features).

    Returns None on non-Linux platforms.
    Returns dict with Linux-specific CPU details:
        - vendor: CPU vendor (Intel, AMD, ARM, etc.)
        - model_name: Full model name
        - sockets: Number of physical sockets
        - cores_per_socket: Cores per socket
        - threads_per_core: Hyperthreading/SMT threads per core
        - total_cores: Physical core count
        - total_threads: Logical CPU count
        - cache: Dict with L1d, L1i, L2, L3 sizes
        - features: Performance-relevant CPU flags
        - is_hybrid: True if Intel hybrid (P/E cores) detected
        - max_mhz: Maximum frequency if available
        - architecture: x86_64, aarch64, etc.

    Example:
        >>> info = get_linux_cpu_info()
        >>> if info:
        ...     print(f"{info['model_name']}: {info['total_cores']}C/{info['total_threads']}T")
        Intel(R) Core(TM) i7-10700K: 8C/16T
    """
    if platform.system() != "Linux":
        return None

    info = {
        "vendor": None,
        "model_name": None,
        "sockets": None,
        "cores_per_socket": None,
        "threads_per_core": None,
        "total_cores": None,
        "total_threads": None,
        "cache": {},
        "features": [],
        "is_hybrid": False,
        "max_mhz": None,
        "architecture": platform.machine(),
    }

    # Try lscpu first (richer data)
    lscpu = _parse_lscpu()
    if lscpu:
        info["vendor"] = lscpu.get('vendor')
        info["model_name"] = lscpu.get('model_name')
        info["sockets"] = lscpu.get('sockets')
        info["cores_per_socket"] = lscpu.get('cores_per_socket')
        info["threads_per_core"] = lscpu.get('threads_per_core')
        info["total_threads"] = lscpu.get('cpus_total')
        info["max_mhz"] = lscpu.get('max_mhz')

        # Calculate total physical cores
        if info["sockets"] and info["cores_per_socket"]:
            info["total_cores"] = info["sockets"] * info["cores_per_socket"]
        elif info["total_threads"] and info["threads_per_core"]:
            info["total_cores"] = info["total_threads"] // info["threads_per_core"]

        # Cache from lscpu
        for key in ['l1d_cache', 'l1i_cache', 'l2_cache', 'l3_cache']:
            if key in lscpu:
                cache_key = key.replace('_cache', '').upper()
                info["cache"][cache_key] = lscpu[key]

        # Features - filter to performance-relevant
        if 'flags' in lscpu:
            relevant = {'sse', 'sse2', 'sse3', 'ssse3', 'sse4_1', 'sse4_2',
                       'avx', 'avx2', 'avx512f', 'avx512vl', 'fma', 'neon',
                       'asimd', 'aes', 'sha1', 'sha2'}
            info["features"] = [f for f in lscpu['flags'] if f in relevant]

    # Try sysfs for cache if lscpu didn't provide it
    if not info["cache"]:
        sysfs_cache = _get_linux_cache_from_sysfs()
        if sysfs_cache:
            info["cache"] = sysfs_cache

    # Fallback to /proc/cpuinfo for missing data
    if not info["model_name"]:
        proc_info = _parse_proc_cpuinfo()
        info["model_name"] = proc_info.get("model")
        info["vendor"] = proc_info.get("vendor")
        if not info["features"] and proc_info.get("features"):
            relevant = {'sse', 'sse2', 'sse3', 'ssse3', 'sse4_1', 'sse4_2',
                       'avx', 'avx2', 'avx512f', 'avx512vl', 'fma', 'neon',
                       'asimd', 'aes', 'sha1', 'sha2'}
            info["features"] = [f for f in proc_info["features"] if f.lower() in relevant]

    # Detect Intel hybrid architecture (Alder Lake+)
    if info["model_name"] and "Intel" in str(info["vendor"] or ""):
        # Hybrid CPUs have "P-core" or "E-core" indicators in some tools
        # or we can check for specific model numbers (12th gen+)
        model_lower = info["model_name"].lower()
        if any(x in model_lower for x in ['12th gen', '13th gen', '14th gen', 'ultra']):
            info["is_hybrid"] = True

    return info


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
    Get hardware specification (cross-platform: Linux + macOS).

    Returns:
        Dict with CPU, memory, and architecture info.

    Example:
        >>> spec = get_hardware_spec()
        >>> print(spec['cpu_model'])
        'Intel(R) Core(TM) i7-10700K CPU @ 3.80GHz'  # Linux
        'Apple M3 Max'  # macOS Apple Silicon
    """
    # Platform dispatch
    if platform.system() == "Darwin":
        # macOS path
        return _get_macos_hardware_spec()

    # Linux path (original implementation)
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
