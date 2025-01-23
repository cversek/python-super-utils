"""
super_utils
===========

A Python package providing advanced debugging utilities and tagged logging.

Modules:
--------
- debug: Debugging tools for traceback, embedding IPython shells, and more.
- logging: Custom logging utilities with module and function tagging.
"""

from .debug import DEBUG_TAG, DEBUG_EMBED, DEBUG_PRINT_EXCEPTION, DEBUG_PLOTS_NONBLOCKING, DEBUG_BREAKPOINT
from .logging import setup_tagged_logger
