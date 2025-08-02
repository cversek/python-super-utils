# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`python-super-utils` is a Python package providing advanced debugging tools and customizable tagged logging. The package consists of two main modules:
- `super_utils.debug`: Debugging utilities including traceback printing, IPython embedding, and matplotlib non-blocking plots
- `super_utils.logging`: Tagged logging with automatic function/module name inclusion

## Development Commands

### Installation
```bash
# Install in editable mode for development
pip install -e .

# Install dependencies only
pip install IPython>=7.0 nest_asyncio>=1.5 matplotlib>=3.0 rich>=14
```

### Running Examples
```bash
cd examples
python example_debug.py
python example_logging.py
python example_combined.py
```

### Building
```bash
# Build distribution packages
python -m build

# Build wheel only
python -m build --wheel
```

## Architecture

The package has a simple structure with two main modules:

1. **debug.py**: Provides debugging utilities
   - `DEBUG_TAG()`: Prints file location and optional message
   - `DEBUG_PRINT_EXCEPTION()`: Formats and prints exception tracebacks using rich
   - `DEBUG_EMBED()`/`DEBUG_BREAKPOINT()`: Drops into IPython shell with current namespace
   - `DEBUG_PLOTS_NONBLOCKING()`: Enables non-blocking matplotlib plots

2. **logging.py**: Enhanced logging with automatic tagging
   - `TaggedFormatter`: Custom formatter that adds module.function tags to log messages
   - `setup_tagged_logger()`: Creates a logger with the tagged formatter
   - Uses timestamp formatting and includes automatic function/module detection

## Key Implementation Details

- Debug functions require passing `currentframe()` explicitly due to Python's frame inspection limitations
- The logging module traverses the call stack to find the outermost user code frame
- Rich library is used for enhanced exception formatting
- nest_asyncio is used to handle IPython embedding in async contexts
- All debug output uses a consistent asterisk-bordered format

## Recent Changes

The logging module was recently updated to:
- Add timestamps to every log message
- Support writing exceptions to a log file
- Use rich for better exception formatting