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
# standard imports
import sys, logging, traceback, inspect, datetime
from inspect import currentframe

# third party imports
import IPython
import nest_asyncio
import rich
from rich.console import Console
from rich.traceback import Traceback



def DEBUG_TAG(frame, msg=None, tag_name = "DEBUG", include_fileinfo=True):
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
    tag = f"*** {tag_name} ***"
    if include_fileinfo:
        tag += f"\n*** \tL#{info.lineno} in '{info.filename}'"
    if msg is not None:
        tag += f": {msg}"
    print('*' * 40)
    print(tag)
    print('*' * 40)


def DEBUG_PRINT_EXCEPTION(log_file=None):
    """
    Print a visually enhanced traceback for the most recent exception.

    Parameters:
    -----------
    log_file : str, optional
        If provided, the traceback is also saved to the specified log file.

    Example:
    --------
    try:
        1 / 0
    except:
        DEBUG_PRINT_EXCEPTION(log_file="errors.log")
    """
    # Create a rich console
    console = Console()

    # Capture the exception details
    exc_type, exc_value, exc_traceback = sys.exc_info()

    # Render the traceback using rich
    console.print("*"*40)
    console.print(Traceback.from_exception(exc_type, exc_value, exc_traceback))
    console.print("*"*40)

    # Optionally log to a file
    if log_file:
        with open(log_file, "a") as f:
            f.write(f"--- Exception Logged on {datetime.datetime.now()} ---\n")
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)
            f.write("\n")



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

def DEBUG_BREAKPOINT(message=None, force_tty=False, exit=False):
    """
    Drop into an interactive IPython shell with contextual debugging information.

    Parameters:
    -----------
    message : str, optional
        An optional message to display alongside the debug tag.
    force_tty : bool, optional
        Force TTY mode for IPython shell (default: False).
    exit : bool, optional
        Exit the program after the IPython shell (default: False).
    """
    # Retrieve the current frame (caller of DEBUG_BREAKPOINT)
    frame = inspect.currentframe().f_back

    # Print the debug tag (file name, line number, optional message)
    DEBUG_TAG(frame, msg=message, tag_name="DEBUG_BREAKPOINT")

    # Launch IPython shell with the local and global namespaces
    DEBUG_EMBED(local_ns=frame.f_locals, global_ns=frame.f_globals, exit=exit, force_tty=force_tty)



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
