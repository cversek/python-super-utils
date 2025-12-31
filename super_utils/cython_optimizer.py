"""
cython_optimizer.py
===================

Cython compiler optimization utilities for detecting hardware and recommending
optimal compilation flags. Supports Apple Silicon (M1/M2/M3/M4) and Linux
(x86_64/ARM) platforms.

Functions:
----------
- get_optimal_compile_args: Recommend optimized compiler flags for current hardware
- get_mcpu_for_chip: Map Apple Silicon generation to -mcpu value
- explain_flags: Get human-readable explanations for compiler flags

Usage:
------
    from super_utils.cython_optimizer import get_optimal_compile_args

    # Auto-detect hardware and get conservative flags
    result = get_optimal_compile_args()
    print(result['extra_compile_args'])  # ['-O3', '-mcpu=apple-m3', ...]

    # Use aggressive optimization
    result = get_optimal_compile_args(profile="aggressive")
"""

import platform
from typing import Dict, Any, List, Optional

from .system_spec import get_system_spec, get_apple_silicon_info, get_linux_cpu_info


def get_mcpu_for_chip(chip_info: Dict) -> Optional[str]:
    """
    Map Apple Silicon generation to appropriate -mcpu value.

    Args:
        chip_info: Dict from get_apple_silicon_info() containing 'generation' key

    Returns:
        -mcpu value string (e.g., 'apple-m3') or None if not Apple Silicon

    Example:
        >>> chip = get_apple_silicon_info()
        >>> if chip:
        ...     mcpu = get_mcpu_for_chip(chip)
        ...     print(f"Use -mcpu={mcpu}")
        Use -mcpu=apple-m3
    """
    if not chip_info:
        return None

    generation = chip_info.get('generation')
    if not generation:
        return None

    # Map generation to -mcpu flag
    # Apple Silicon generations: M1, M2, M3, M4
    generation_map = {
        'M1': 'apple-m1',
        'M2': 'apple-m2',
        'M3': 'apple-m3',
        'M4': 'apple-m4',
    }

    return generation_map.get(generation)


def _get_apple_silicon_flags(chip_info: Dict, profile: str) -> Dict[str, Any]:
    """Get optimized flags for Apple Silicon."""
    mcpu = get_mcpu_for_chip(chip_info)

    # Conservative profile (default)
    flags = [
        '-O3',                    # Full optimization
        '-fno-math-errno',        # Don't set errno on math calls (faster)
        '-ftree-vectorize',       # Enable auto-vectorization
    ]

    reasoning = {
        '-O3': 'Full optimization - balances compilation time and runtime performance',
        '-fno-math-errno': 'Skip errno updates on math functions (safe for most numerical code)',
        '-ftree-vectorize': 'Enable SIMD vectorization for loops (NEON on ARM64)',
    }

    # Add -mcpu if detected
    if mcpu:
        flags.append(f'-mcpu={mcpu}')
        reasoning[f'-mcpu={mcpu}'] = f'Target {chip_info.get("generation")} {chip_info.get("variant", "").strip() or "base"} architecture explicitly'
    else:
        # Fallback to generic ARM64
        flags.append('-mcpu=native')
        reasoning['-mcpu=native'] = 'Auto-detect ARM64 target (compiler will choose best match)'

    # Aggressive profile additions
    if profile == "aggressive":
        flags.append('-ffast-math')
        reasoning['-ffast-math'] = 'AGGRESSIVE: Relaxed IEEE 754 compliance for speed (may affect NaN/Inf handling)'

    return {
        'extra_compile_args': flags,
        'extra_link_args': [],
        'profile': profile,
        'reasoning': reasoning,
        'platform': 'apple_silicon',
        'chip_generation': chip_info.get('generation'),
        'chip_variant': chip_info.get('variant'),
    }


def _get_linux_x86_flags(cpu_info: Dict, profile: str) -> Dict[str, Any]:
    """Get optimized flags for Linux x86_64."""
    flags = [
        '-O3',
        '-march=native',          # Use all available CPU features
        '-fno-math-errno',
        '-ftree-vectorize',
    ]

    reasoning = {
        '-O3': 'Full optimization - balances compilation time and runtime performance',
        '-march=native': 'Use all CPU features (AVX/AVX2/AVX512 if available)',
        '-fno-math-errno': 'Skip errno updates on math functions (safe for most numerical code)',
        '-ftree-vectorize': 'Enable SIMD vectorization for loops',
    }

    # Check for specific features
    features = cpu_info.get('features', [])
    feature_notes = []
    if 'avx512f' in features:
        feature_notes.append('AVX-512')
    elif 'avx2' in features:
        feature_notes.append('AVX2')
    elif 'avx' in features:
        feature_notes.append('AVX')

    if 'fma' in features:
        feature_notes.append('FMA')

    if feature_notes:
        reasoning['-march=native'] += f' (detected: {", ".join(feature_notes)})'

    # Aggressive profile additions
    if profile == "aggressive":
        flags.append('-ffast-math')
        reasoning['-ffast-math'] = 'AGGRESSIVE: Relaxed IEEE 754 compliance for speed (may affect NaN/Inf handling)'

    return {
        'extra_compile_args': flags,
        'extra_link_args': [],
        'profile': profile,
        'reasoning': reasoning,
        'platform': 'linux_x86_64',
        'cpu_model': cpu_info.get('model_name'),
        'cpu_features': features,
    }


def _get_linux_arm_flags(cpu_info: Dict, profile: str) -> Dict[str, Any]:
    """Get optimized flags for Linux ARM/aarch64."""
    flags = [
        '-O3',
        '-march=native',          # Use all available CPU features
        '-fno-math-errno',
        '-ftree-vectorize',
    ]

    reasoning = {
        '-O3': 'Full optimization - balances compilation time and runtime performance',
        '-march=native': 'Use all ARM64 features (NEON, SVE if available)',
        '-fno-math-errno': 'Skip errno updates on math functions (safe for most numerical code)',
        '-ftree-vectorize': 'Enable SIMD vectorization for loops (NEON/SVE)',
    }

    # Check for NEON/SVE
    features = cpu_info.get('features', [])
    feature_notes = []
    if 'neon' in [f.lower() for f in features]:
        feature_notes.append('NEON')
    if 'sve' in [f.lower() for f in features]:
        feature_notes.append('SVE')

    if feature_notes:
        reasoning['-march=native'] += f' (detected: {", ".join(feature_notes)})'

    # Aggressive profile additions
    if profile == "aggressive":
        flags.append('-ffast-math')
        reasoning['-ffast-math'] = 'AGGRESSIVE: Relaxed IEEE 754 compliance for speed (may affect NaN/Inf handling)'

    return {
        'extra_compile_args': flags,
        'extra_link_args': [],
        'profile': profile,
        'reasoning': reasoning,
        'platform': 'linux_arm64',
        'cpu_model': cpu_info.get('model_name'),
        'cpu_features': features,
    }


def _get_generic_flags(profile: str) -> Dict[str, Any]:
    """Fallback flags for unknown/unsupported platforms."""
    flags = [
        '-O3',
        '-fno-math-errno',
    ]

    reasoning = {
        '-O3': 'Full optimization - balances compilation time and runtime performance',
        '-fno-math-errno': 'Skip errno updates on math functions (safe for most numerical code)',
    }

    if profile == "aggressive":
        flags.append('-ffast-math')
        reasoning['-ffast-math'] = 'AGGRESSIVE: Relaxed IEEE 754 compliance for speed (may affect NaN/Inf handling)'

    return {
        'extra_compile_args': flags,
        'extra_link_args': [],
        'profile': profile,
        'reasoning': reasoning,
        'platform': 'generic',
        'warning': 'Platform not recognized - using conservative generic flags',
    }


def get_optimal_compile_args(
    spec: Optional[Dict] = None,
    profile: str = "conservative"
) -> Dict[str, Any]:
    """
    Get optimal Cython compiler flags for current hardware.

    Detects platform (Apple Silicon, Linux x86_64, Linux ARM) and returns
    appropriate compiler flags for setup.py extra_compile_args.

    Args:
        spec: System spec dict from get_system_spec() (auto-detected if None)
        profile: Optimization profile - "baseline", "conservative" (default), or "aggressive"
                 Baseline: -O2 only (system defaults for comparison baseline)
                 Conservative: -O3 -march=native/mcpu -ftree-vectorize -fno-math-errno
                 Aggressive: adds -ffast-math (WARNING: may violate IEEE 754)

    Returns:
        Dict containing:
            - extra_compile_args: List of compiler flags
            - extra_link_args: List of linker flags (usually empty)
            - profile: The profile used
            - reasoning: Dict mapping each flag to explanation
            - platform: Platform identifier string
            - Additional platform-specific metadata

    Example:
        >>> result = get_optimal_compile_args()
        >>> print(result['extra_compile_args'])
        ['-O3', '-mcpu=apple-m3', '-fno-math-errno', '-ftree-vectorize']
        >>>
        >>> # Use in setup.py:
        >>> from super_utils.cython_optimizer import get_optimal_compile_args
        >>> opts = get_optimal_compile_args(profile="aggressive")
        >>> ext_modules = [
        ...     Extension("mymodule", ["mymodule.pyx"],
        ...               extra_compile_args=opts['extra_compile_args'])
        ... ]
    """
    if profile not in ("baseline", "conservative", "aggressive"):
        raise ValueError(f"profile must be 'baseline', 'conservative', or 'aggressive', got: {profile}")

    # Baseline profile: minimal flags for comparison
    if profile == "baseline":
        return {
            'extra_compile_args': ['-O2'],
            'extra_link_args': [],
            'profile': 'baseline',
            'reasoning': {
                '-O2': 'Moderate optimization (system default baseline for comparison)',
            },
            'platform': 'baseline',
        }

    # Auto-detect system if not provided
    if spec is None:
        spec = get_system_spec()

    # Determine platform and dispatch
    os_platform = spec.get('os', {}).get('platform')
    arch = spec.get('hardware', {}).get('architecture')

    # macOS Apple Silicon
    if os_platform == "Darwin" and arch == "arm64":
        chip_info = get_apple_silicon_info()
        if chip_info:
            return _get_apple_silicon_flags(chip_info, profile)

    # Linux x86_64
    if os_platform == "Linux" and arch == "x86_64":
        cpu_info = get_linux_cpu_info()
        if cpu_info:
            return _get_linux_x86_flags(cpu_info, profile)

    # Linux ARM/aarch64
    if os_platform == "Linux" and arch in ("aarch64", "arm64"):
        cpu_info = get_linux_cpu_info()
        if cpu_info:
            return _get_linux_arm_flags(cpu_info, profile)

    # Fallback
    return _get_generic_flags(profile)


def explain_flags(flags: List[str]) -> Dict[str, str]:
    """
    Get human-readable explanations for compiler flags.

    Args:
        flags: List of compiler flags

    Returns:
        Dict mapping each flag to its explanation

    Example:
        >>> explanations = explain_flags(['-O3', '-march=native', '-ffast-math'])
        >>> for flag, desc in explanations.items():
        ...     print(f"{flag}: {desc}")
    """
    explanations = {
        '-O3': 'Full optimization level 3 (aggressive inlining, loop unrolling, vectorization)',
        '-O2': 'Moderate optimization level 2 (balanced speed/size)',
        '-Os': 'Optimize for size',
        '-march=native': 'Use all CPU instruction set features available on this machine',
        '-mcpu=native': 'Target current CPU architecture',
        '-ffast-math': 'Allow aggressive math optimizations that violate IEEE 754 (DANGEROUS for some code)',
        '-fno-math-errno': 'Do not set errno for math functions (faster, safe for most code)',
        '-ftree-vectorize': 'Enable automatic SIMD vectorization of loops',
        '-funroll-loops': 'Unroll loops for better performance (increases code size)',
        '-fomit-frame-pointer': 'Omit frame pointer for extra register (makes debugging harder)',
    }

    result = {}
    for flag in flags:
        # Check exact match first
        if flag in explanations:
            result[flag] = explanations[flag]
        # Check for -mcpu=apple-m* variants
        elif flag.startswith('-mcpu=apple-m'):
            generation = flag.replace('-mcpu=apple-', '').upper()
            result[flag] = f'Target Apple Silicon {generation} architecture specifically'
        # Check for other -mcpu variants
        elif flag.startswith('-mcpu='):
            target = flag.replace('-mcpu=', '')
            result[flag] = f'Target {target} CPU architecture'
        # Check for -march variants
        elif flag.startswith('-march='):
            target = flag.replace('-march=', '')
            if target == 'native':
                result[flag] = explanations['-march=native']
            else:
                result[flag] = f'Target {target} instruction set architecture'
        else:
            result[flag] = 'No explanation available'

    return result
