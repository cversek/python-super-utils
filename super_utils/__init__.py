"""
super_utils
===========

A Python package providing advanced debugging utilities, tagged logging, and memory profiling.

Modules:
--------
- debug: Debugging tools for traceback, embedding IPython shells, and more.
- logging: Custom logging utilities with module and function tagging.
- memory: Memory profiling with timeline tracking and Rich-formatted reports.
"""

from .debug import DEBUG_TAG, DEBUG_EMBED, DEBUG_PRINT_EXCEPTION, DEBUG_PLOTS_NONBLOCKING, DEBUG_BREAKPOINT
from .logging import setup_tagged_logger
from .memory import (
    MEMORY_SNAPSHOT,
    MEMORY_DELTA,
    MEMORY_REPORT,
    MEMORY_TIMELINE,
    MEMORY_RESET,
    MEMORY_CONFIGURE,
)
from .timing import (
    TIMING_START,
    TIMING_END,
    TIMING_REPORT,
    TIMING_TIMELINE,
    TIMING_RESET,
    TIMING_CONFIGURE,
    TIMING_EXPORT_JSON,
    PROFILE_SECTION,
)
from .system_spec import (
    get_system_spec,
    get_hardware_spec,
    get_compiler_spec,
    get_numpy_spec,
    print_system_report,
    export_system_spec,
)
