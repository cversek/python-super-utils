"""
example_debug.py
================

Examples of debugging functions from the super_utils.debug module.
"""

from super_utils.debug import (
    DEBUG_TAG,
    DEBUG_PRINT_EXCEPTION,
    DEBUG_EMBED,
    DEBUG_BREAKPOINT,
    DEBUG_PLOTS_NONBLOCKING,
)
from inspect import currentframe
import matplotlib.pyplot as plt


def demo_debug_tag():
    """
    Demonstrate DEBUG_TAG functionality.
    """
    print("\n--- Example: DEBUG_TAG ---")
    DEBUG_TAG(currentframe(), "This is a tagged debug message.")


def demo_debug_print_exception():
    """
    Demonstrate DEBUG_PRINT_EXCEPTION functionality.
    """
    print("\n--- Example: DEBUG_PRINT_EXCEPTION ---")
    try:
        1 / 0  # This will raise a ZeroDivisionError
    except:
        DEBUG_PRINT_EXCEPTION()

global_var = "I'm a global variable in the IPython shell!"

def demo_debug_embed():
    """
    Demonstrate DEBUG_EMBED functionality.
    """
    print("\n--- Example: DEBUG_EMBED ---")
    local_var = "I'm a local variable in the IPython shell!"
    DEBUG_EMBED(local_ns=locals(), global_ns=globals(), exit=False)


def demo_debug_breakpoint():
    """
    Demonstrate DEBUG_BREAKPOINT functionality.
    """
    print("\n--- Example: DEBUG_BREAKPOINT ---")
    x = 42
    y = "hello world"
    DEBUG_BREAKPOINT("Check the state of x,y,global_var")
    print("After the breakpoint")


def demo_debug_plots():
    """
    Demonstrate DEBUG_PLOTS_NONBLOCKING functionality.
    """
    print("\n--- Example: DEBUG_PLOTS_NONBLOCKING ---")
    plt.plot([1, 2, 3], [4, 5, 6])
    plt.title("Non-blocking Plot Example")
    DEBUG_PLOTS_NONBLOCKING(do_prompt=True)


if __name__ == "__main__":
    demo_debug_tag()
    demo_debug_print_exception()
    demo_debug_embed()
    demo_debug_breakpoint()
    demo_debug_plots()
