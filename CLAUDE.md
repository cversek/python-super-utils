# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`python-super-utils` is a Python package providing advanced debugging tools, customizable tagged logging, and profiling utilities. The package consists of five modules:

- `super_utils.debug`: Debugging utilities (traceback printing, IPython embedding, matplotlib non-blocking plots)
- `super_utils.logging`: Tagged logging with automatic function/module name inclusion
- `super_utils.memory`: Memory profiling with tracemalloc-based timeline tracking and Rich-formatted reports
- `super_utils.timing`: Execution timing with timeline tracking and JSON export
- `super_utils.system_spec`: System specification capture for reproducible benchmarks (CPU, memory, BLAS, compiler info)

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

**Memory**: Wraps `tracemalloc` with RSS tracking via `/proc/self/status`. `MEMORY_SNAPSHOT()` captures caller frame context automatically.

**Timing**: Uses `time.perf_counter()` for precision. `PROFILE_SECTION` context manager combines timing + memory tracking. `TIMING_EXPORT_JSON()` includes system spec for reproducibility.

**System Spec**: Captures hardware (via `/proc/cpuinfo`, `/proc/meminfo`), compiler (via `sysconfig`), and NumPy/BLAS info. Can resolve `-march=native` to actual architecture. Results are cached at module level.

## Key Dependencies

- `rich>=14` - Console formatting for all output
- `IPython>=7.0` - Interactive debugging via `DEBUG_EMBED`/`DEBUG_BREAKPOINT`
- `nest_asyncio>=1.5` - Enable IPython in async contexts
- `matplotlib>=3.0` - Non-blocking plot support
