"""
logging.py
==========

Custom logging utilities with automatic tagging of module and function names.

Classes:
--------
- TaggedFormatter: A logging formatter that adds module and function name tags.

Functions:
----------
- setup_tagged_logger: Set up a logger with the TaggedFormatter.
"""

import logging
import inspect

class TaggedFormatter(logging.Formatter):
    """
    Custom formatter to include the module and function name tags 
    from the *outermost caller* in user code.
    """
    def format(self, record):
        # Traverse the call stack and skip over internal logging frames
        stack = inspect.stack()
        caller_frame = None

        for frame_info in stack:
            module_name = frame_info.frame.f_globals.get("__name__", "__main__")
            # Skip frames from the logging system and this custom formatter
            if not (module_name.startswith("logging") or module_name.startswith("super_utils.logging")):
                caller_frame = frame_info
                break

        # If a caller frame is found, extract module and function
        if caller_frame:
            module_name = caller_frame.frame.f_globals.get("__name__", "__main__")
            function_name = caller_frame.function
        else:
            module_name = "__unknown__"
            function_name = "__unknown__"

        # Add the tag to the log record
        record.tag = f"[{module_name}.{function_name}]"
        return super().format(record)
    

def setup_tagged_logger(name=None, level=logging.DEBUG):
    """
    Set up a logger that includes module and function tags in log messages.

    Parameters:
    -----------
    name : str, optional
        Logger name (default: None).
    level : int, optional
        Logging level (default: DEBUG).

    Returns:
    --------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name or __name__)
    logger.setLevel(level)

    if not logger.handlers:  # Avoid duplicate handlers
        handler = logging.StreamHandler()
        formatter = TaggedFormatter("%(tag)s %(levelname)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
