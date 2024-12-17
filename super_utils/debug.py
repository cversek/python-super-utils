"""
debug.py
========

Debugging tools for printing detailed traceback information, embedding IPython 
interactive shells, and controlling Matplotlib plots non-blocking behavior.

Functions:
----------
- DEBUG_TAG: Print a debug tag with file name and line number.
- DEBUG_PRINT_EXCEPTION: Print formatted exception tracebacks.
- DEBUG_EMBED: Embed an IPython shell for debugging.
- DEBUG_PLOTS_NONBLOCKING: Show Matplotlib plots in a non-blocking way.
"""

import sys
import logging
import traceback
import inspect
import IPython
import nest_asyncio
from inspect import currentframe

def DEBUG_TAG(frame, msg=None):
    """
    Print a debug tag including file name, line number, and an optional message.

    Parameters:
    -----------
    frame : frame
        Current frame object from `inspect.currentframe()`.
    msg : str, optional
        An optional debug message.

    Example:
    --------
    >>> DEBUG_TAG(currentframe(), "Debugging start point")
    """
    info = inspect.getframeinfo(frame)
    tag = f"*** DEBUG ***\n*** \tL#{info.lineno} in '{info.filename}'"
    if msg is not None:
        tag += f": {msg}"
    print('*' * 40)
    print(tag)
    print('*' * 40)


def DEBUG_PRINT_EXCEPTION():
    """
    Print the formatted traceback for the most recent exception.

    Example:
    --------
    try:
        1 / 0
    except:
        DEBUG_PRINT_EXCEPTION()
    """
    exc = traceback.format_exc()
    tag = f"*** EXCEPTION ***\n*** \n\t{exc}"
    print('*' * 40)
    print(tag)
    print('*' * 40)


def DEBUG_EMBED(local_ns, global_ns=None, exit=False, force_tty=False):
    """
    Embed an interactive IPython shell for debugging.

    Parameters:
    -----------
    local_ns : dict
        Local namespace to make variables available in the IPython shell.
    global_ns : dict, optional
        Global namespace.
    exit : bool, optional
        Exit the program after the shell (default: False).
    force_tty : bool, optional
        Force TTY mode for interactive shell (default: False).

    Notes:
    ------
    Useful for dropping into an interactive session mid-execution.
    """
    nest_asyncio.apply()  # Enable nested event loops
    if global_ns is None:
        global_ns = {}
    user_ns = global_ns.copy()
    user_ns.update(local_ns)

    # Check for TTY input environment
    from sys import stdin
    if not force_tty and stdin.isatty():
        IPython.embed(user_ns=user_ns)
    elif force_tty:
        IPython.embed(user_ns=user_ns)
    else:
        print('*' * 40)
        print("*** NOTICE: cannot use DEBUG_EMBED without an input terminal")
        print('*' * 40)

    if exit:
        sys.exit()


def DEBUG_PLOTS_NONBLOCKING(do_prompt=False):
    """
    Display Matplotlib plots in non-blocking mode and optionally close them.

    Parameters:
    -----------
    do_prompt : bool, optional
        Prompt the user to close plots manually (default: False).

    Example:
    --------
    >>> DEBUG_PLOTS_NONBLOCKING(True)
    """
    from matplotlib import pyplot as plt
    plt.show(block=False)
    if do_prompt:
        try:
            resp = input("*** PRESS ENTER TO CLOSE PLOTS AND CONTINUE (enter 'Keep' to keep open) ***")
            if resp.lower() != 'keep':
                plt.close('all')
        except EOFError:
            plt.close('all')
