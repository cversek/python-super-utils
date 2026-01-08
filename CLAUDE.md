# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`python-super-utils` is a Python package providing advanced debugging tools, customizable tagged logging, profiling utilities, and Cython optimization infrastructure. The package consists of these modules:

- `super_utils.debug`: Debugging utilities (traceback printing, IPython embedding, matplotlib non-blocking plots)
- `super_utils.logging`: Tagged logging with automatic function/module name inclusion
- `super_utils.memory`: Memory profiling with tracemalloc-based timeline tracking and Rich-formatted reports
- `super_utils.timing`: Execution timing with timeline tracking and JSON export
- `super_utils.system_spec`: System specification capture for reproducible benchmarks (CPU, memory, BLAS, compiler info)
- `super_utils.cython_optimizer`: Hardware-aware Cython compiler flag recommendations
- `super_utils.benchmarks`: Optimization benchmark suite with 12 benchmarks in 6 paired classes
- `super_utils.cli`: Command-line interface (`superutils` command)

## Development Commands

```bash
# Install in editable mode
pip install -e .

# Run examples
python examples/example_debug.py
python examples/example_logging.py
python examples/example_combined.py

# Build distribution
python -m build
```

## Architecture

### Debug Module (`debug.py`)
All `DEBUG_*` functions print output in a consistent asterisk-bordered format. Key design decisions:
- `DEBUG_TAG()` and `DEBUG_EMBED()` require passing `currentframe()` explicitly due to Python's frame inspection limitations
- `DEBUG_BREAKPOINT()` is the preferred entry point - it internally calls `currentframe().f_back` to get the caller's frame
- Uses `nest_asyncio` to enable IPython embedding in async contexts
- Uses Rich library for enhanced exception formatting in `DEBUG_PRINT_EXCEPTION()`

### Logging Module (`logging.py`)
- `TaggedFormatter` traverses the call stack to find the outermost user code frame (skipping logging internals)
- Logs include timestamps and `[module.function]` tags automatically
- `configure_global_logging()` provides single-point control for all loggers/handlers via a module-level `_global_logging_level` variable

### Profiling Modules (`memory.py`, `timing.py`, `system_spec.py`)
These three modules follow consistent patterns:
- Module-level state (snapshots/records lists, configuration flags)
- `*_CONFIGURE()` for global settings
- `*_RESET()` to clear accumulated data
- `*_REPORT()` for Rich-formatted console output
- `*_TIMELINE()` for chronological display
- All use Rich tables/panels for output formatting

**Memory**: Two tracking modes available:
- *Point-in-time*: `MEMORY_SNAPSHOT()` captures RSS at discrete moments with caller frame context.
- *Threaded peak*: `MEMORY_PEAK_START/STOP` or `MEMORY_PEAK_CONTEXT` sample RSS in background thread (default 10ms) to catch transient allocations freed before next snapshot. Use for NumPy vectorized ops; use snapshots for streaming/accumulator patterns.

**Timing**: Uses `time.perf_counter()` for precision. `PROFILE_SECTION` context manager combines timing + memory tracking. `TIMING_EXPORT_JSON()` includes system spec for reproducibility.

**System Spec**: Captures hardware (via `/proc/cpuinfo`, `/proc/meminfo` on Linux; `sysctl` on macOS), compiler (via `sysconfig`), and NumPy/BLAS info. Can resolve `-march=native` to actual architecture. Results are cached at module level.

### Cython Optimization (`cython_optimizer.py`, `benchmarks/`)

**Cython Optimizer**: `get_optimal_compile_args(profile="conservative"|"aggressive")` returns hardware-aware compiler flags. Detects CPU features (AVX, NEON) and generates appropriate `-march`, `-O3`, `-ftree-vectorize` flags.

**Benchmarks**: 12 benchmarks in 6 paired classes test different workload characteristics:
- `streaming` / `cython_streaming`: Memory-bound nested loop accumulation
- `wavelet` / `cython_wavelet`: Compute-bound convolution (14x speedup)
- `branch` / `cython_branch`: Branch-heavy conditionals (66x speedup)
- `linalg` / `cython_linalg`: BLAS-intensive matrix operations
- `interp` / `cython_interp`: Mixed workload interpolation pipeline
- `mfvep` / `cython_mfvep`: Canonical clinical mfVEP kernel (12x speedup, 16k×600×60 dimensions)

Cython kernels live in `benchmarks/_cython/*.pyx` with directives: `boundscheck=False, wraparound=False, cdivision=True`.

**CLI** (`cli/cython_cmd.py`): `superutils cython benchmark` runs paired benchmarks with speedup display. `--rebuild` recompiles extensions to test compiler flag effectiveness.

## Key Dependencies

- `rich>=14` - Console formatting for all output
- `IPython>=7.0` - Interactive debugging via `DEBUG_EMBED`/`DEBUG_BREAKPOINT`
- `nest_asyncio>=1.5` - Enable IPython in async contexts
- `matplotlib>=3.0` - Non-blocking plot support
