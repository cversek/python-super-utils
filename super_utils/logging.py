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
    Custom formatter to include module and function name tags in log records.

    Example Output:
    ---------------
    [module_name.function_name] DEBUG: This is a debug message.
    """
    def format(self, record):
        # Find the caller frame (3 frames up)
        caller_frame = inspect.getouterframes(inspect.currentframe())[3]
        module_name = caller_frame.frame.f_globals.get("__name__", "__main__")
        function_name = caller_frame.function

        # Add the tag to the record
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

    Example:
    --------
    >>> logger = setup_tagged_logger()
    >>> logger.debug("This is a tagged debug message")
    """
    logger = logging.getLogger(name or __name__)
    logger.setLevel(level)

    if not logger.handlers:  # Avoid duplicate handlers
        handler = logging.StreamHandler()
        formatter = TaggedFormatter("%(tag)s %(levelname)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
