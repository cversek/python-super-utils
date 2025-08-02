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
- configure_global_logging: Configure global logging level for all tagged loggers.
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
    

def setup_tagged_logger(name=None, level=None):
    """
    Set up a logger that includes module and function tags in log messages.

    Parameters:
    -----------
    name : str, optional
        Logger name (default: None).
    level : int, optional
        Logging level (default: None, which inherits from root logger or defaults to DEBUG).

    Returns:
    --------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name or __name__)
    
    # If no level specified, use global configured level, root logger level, or DEBUG as fallback
    if level is None:
        global _global_logging_level
        if _global_logging_level is not None:
            level = _global_logging_level
        else:
            root_logger = logging.getLogger()
            if root_logger.level != logging.NOTSET:
                level = root_logger.level
            else:
                level = logging.DEBUG
    
    logger.setLevel(level)

    if not logger.handlers:  # Avoid duplicate handlers
        handler = logging.StreamHandler()
        handler.setLevel(level)  # Set handler level to match logger level
        formatter = TaggedFormatter("%(asctime)s %(tag)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        # Update existing handler levels to match new logger level
        for handler in logger.handlers:
            handler.setLevel(level)

    return logger


def configure_global_logging(level=logging.INFO):
    """
    Configure global logging level for all loggers and handlers.
    
    This function sets the logging level for:
    - The root logger
    - All existing loggers and their handlers
    - Future loggers created by setup_tagged_logger
    
    This hides the complexity of Python's logging system and provides
    a single point of control for logging levels.
    
    Parameters:
    -----------
    level : int
        Logging level (e.g., logging.DEBUG, logging.INFO, logging.WARNING)
    """
    # Set root logger level
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Update all existing loggers and their handlers
    for name in list(logging.Logger.manager.loggerDict.keys()):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)
    
    # Store the global level for future loggers
    global _global_logging_level
    _global_logging_level = level


# Global variable to store the configured logging level
_global_logging_level = None
