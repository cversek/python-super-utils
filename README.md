# python-super-utils

**`python-super-utils`** is a Python package providing advanced debugging tools, profiling utilities, and Cython compiler optimization for developers.

- **Debugging utilities**: Print detailed traceback, embed IPython shells, and handle Matplotlib plots in non-blocking mode.
- **Tagged logging**: Automatically include module and function names in log outputs for better traceability.
- **Profiling**: Memory and timing instrumentation with Rich-formatted reports.
- **System specification**: Cross-platform hardware detection for reproducible benchmarks.
- **Cython optimization**: Hardware-aware compiler flag recommendations and build tools.

---

## **CLI: superutils**

The package provides a command-line interface for Cython optimization and system inspection.

### Cython Commands

```bash
# Detect hardware and show recommended compiler flags
superutils cython detect

# Run optimization benchmarks to identify workload characteristics
superutils cython benchmark --list              # List available benchmark classes
superutils cython benchmark                     # Run all benchmarks
superutils cython benchmark --class streaming   # Run specific class

# Build Cython extensions with optimized flags
superutils cython compile --dry-run             # Preview build plan
superutils cython compile                       # Build with optimized flags
superutils cython compile --profile aggressive  # Use aggressive optimization
```

### System Specification Commands

```bash
# Display system specification
superutils sysspec show

# Export system specification to JSON
superutils sysspec export spec.json
```

### Optimization Workflow

1. **Detect hardware**: `superutils cython detect` shows your CPU capabilities and recommended flags
2. **Run benchmarks**: `superutils cython benchmark` identifies your workload type (memory-bound, compute-bound, etc.)
3. **Preview build**: `superutils cython compile --dry-run` shows the build plan
4. **Build**: `superutils cython compile` builds with optimized flags

---

## **Features**

### Debugging Tools
- `DEBUG_TAG`: Print the file name, line number, and optional messages for debugging checkpoints.
- `DEBUG_PRINT_EXCEPTION`: Display formatted exception tracebacks.
- `DEBUG_EMBED`: Drop into an IPython shell with the current namespace for live debugging.
- `DEBUG_PLOTS_NONBLOCKING`: Show Matplotlib plots in a non-blocking way.

### Tagged Logging
- Logs include module and function names as tags for easier traceability.
- Integrates seamlessly with the Python `logging` module.

### Profiling Tools
- `TIMING_START` / `TIMING_END`: Measure execution time of code sections.
- `MEMORY_SNAPSHOT`: Capture memory usage at checkpoints.
- `PROFILE_SECTION`: Context manager combining timing and memory tracking.
- Rich-formatted reports and JSON export for reproducibility.

### System Specification
- `get_system_spec()`: Cross-platform hardware, OS, and compiler detection.
- `get_apple_silicon_info()`: Apple M1/M2/M3/M4 specific details.
- `get_linux_cpu_info()`: Linux CPU topology and features.

### Cython Optimization
- `get_optimal_compile_args()`: Hardware-aware compiler flag recommendations.
- Benchmark suite with 5 algorithm classes (streaming, wavelet, branch, linalg, interp).
- Build orchestration with automatic flag injection.

---

## **Installation**

### Standard Installation (End-Users)
Install the latest version directly from PyPI:
```bash
pip install python-super-utils
```

If the package is not yet published on PyPI, you can install it directly from the GitHub repository:
```bash
pip install git+https://github.com/cversek/python-super-utils.git
```

### Developer Installation (Editable Mode)
For developers contributing to or testing the project, install in editable mode:

```bash
git clone https://github.com/cversek/python-super-utils.git
cd python-super-utils
pip install -e .
```

Editable installation allows changes to the source code to reflect immediately without reinstalling the package.

---

## **Usage**

### Debugging Tools
Example: DEBUG_TAG
Print the file name, line number, and optional message.
```python
from super_utils.debug import DEBUG_TAG, currentframe

def example():
    DEBUG_TAG(currentframe(), "This is a debug checkpoint")

example()
```
Output:

```
****************************************
*** DEBUG ***
***     L#6 in 'example.py': This is a debug checkpoint
****************************************
```
Note: The line number and file name are automatically extracted from the current frame. Unfortunately, this cannot be done inside the library, so you need to pass the current frame to the function.

Example: DEBUG_PRINT_EXCEPTION
Print a formatted exception traceback.
```python
from super_utils.debug import DEBUG_PRINT_EXCEPTION

try:
    1 / 0
except:
    DEBUG_PRINT_EXCEPTION()
```
Output:
```
****************************************
*** EXCEPTION ***
*** 
    Traceback (most recent call last):
      File "example.py", line 4, in <module>
        1 / 0
    ZeroDivisionError: division by zero
****************************************
```

Example: DEBUG_BREAKPOINT
Drop into an IPython shell with the current namespace for live debugging.
```python
from super_utils.debug import DEBUG_BREAKPOINT

global_variable = "Hello, Global Debug!"

def interactive_debug():
    local_variable = "Hello, Local Debug!"
    DEBUG_BREAKPOINT()

interactive_debug()
```
Note: both global and local variables are available in the IPython shell.


### Tagged Logging
Example: Tagged Logs
```python
from super_utils.logging import setup_tagged_logger

logger = setup_tagged_logger()

def example_function():
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")

example_function()

```
Output:
```
[example_function] DEBUG: This is a debug message.
[example_function] INFO: This is an info message.
```

### Cython Optimization
Example: Using optimized flags in setup.py
```python
from super_utils.cython_optimizer import get_optimal_compile_args
from setuptools import setup, Extension
from Cython.Build import cythonize

# Get hardware-optimized compiler flags
opts = get_optimal_compile_args(profile="conservative")

extensions = [
    Extension(
        "mymodule._fast",
        ["mymodule/_fast.pyx"],
        extra_compile_args=opts['extra_compile_args'],
    )
]

setup(
    name="mymodule",
    ext_modules=cythonize(extensions),
)
```

The `get_optimal_compile_args()` function returns flags like `-O3`, `-march=native`, and `-ftree-vectorize` based on detected hardware. Use `profile="aggressive"` for maximum performance (includes `-ffast-math`).

---
## **Examples**
The examples/ directory contains scripts demonstrating all functionalities:
- `example_debug.py`: Debugging tools examples.
- `example_logging.py`: Tagged logging examples.
- `example_combined.py`: Combined usage of debugging and logging utilities.
To run the examples, navigate to the examples directory and run:
```bash
cd examples
python example_debug.py
python example_logging.py
python example_combined.py
```
---

## **Contributing**
Contributions are welcome! Follow these steps:
1. Fork the repository on GitHub
2. Clone the forked repository to your local machine
```bash
git clone https://github.com/your-username/python-super-utils.git
```
3. Install the package in editable mode
```bash
pip install -e .
```
4. Create a new branch, make your changes, and submit a pull request.


## **License**
This project is open-sourced under the MIT License - see the LICENSE file for details.

## **Links**
- [GitHub Repository](https://github.com/cversek/python-super-utils)
- [Issues](https://github.com/cversek/python-super-utils/issues)