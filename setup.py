from setuptools import setup, find_packages, Extension
import os

# Check for Cython availability
try:
    from Cython.Build import cythonize
    import numpy as np
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
    cythonize = None
    np = None


def get_extensions():
    """
    Get Cython extension modules if Cython is available.

    Returns empty list if Cython or NumPy is not installed,
    allowing pure-Python installation to proceed.
    """
    if not CYTHON_AVAILABLE:
        return []

    # List of Cython modules to compile
    pyx_files = [
        ("super_utils.benchmarks._cython.streaming", "super_utils/benchmarks/_cython/streaming.pyx"),
        ("super_utils.benchmarks._cython.wavelet", "super_utils/benchmarks/_cython/wavelet.pyx"),
        ("super_utils.benchmarks._cython.branch", "super_utils/benchmarks/_cython/branch.pyx"),
        ("super_utils.benchmarks._cython.linalg", "super_utils/benchmarks/_cython/linalg.pyx"),
        ("super_utils.benchmarks._cython.interp", "super_utils/benchmarks/_cython/interp.pyx"),
        ("super_utils.benchmarks._cython.mfvep_kernel", "super_utils/benchmarks/_cython/mfvep_kernel.pyx"),
    ]

    # Filter to only existing files
    existing_pyx = [
        (name, path) for name, path in pyx_files
        if os.path.exists(path)
    ]

    if not existing_pyx:
        return []

    # Create Extension objects
    extensions = [
        Extension(
            name,
            [path],
            include_dirs=[np.get_include()],
        )
        for name, path in existing_pyx
    ]

    # Cythonize with language_level=3
    return cythonize(
        extensions,
        compiler_directives={'language_level': "3"},
        quiet=True
    )


setup(
    name="python-super-utils",
    version="0.2.0",
    packages=find_packages(),
    ext_modules=get_extensions(),
    install_requires=[
        "IPython>=7.0",
        "nest_asyncio>=1.5",
        "matplotlib>=3.0",
        "numpy>=1.20",
        "rich>=10.0",
    ],
    extras_require={
        "cython": ["Cython>=0.29", "numpy>=1.20"],
    },
    entry_points={
        "console_scripts": [
            "superutils=super_utils.cli.main:main",
        ],
    },
    python_requires=">=3.8",
    author="NeuroFieldz",
    description="Advanced debugging, logging, and profiling utilities",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Cython",
    ],
)
