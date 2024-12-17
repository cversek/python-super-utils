"""
example_logging.py
==================

Examples of tagged logging from the super_utils.logging module.
"""

from super_utils.logging import setup_tagged_logger

# Initialize logger
logger = setup_tagged_logger()


def demo_info_logging():
    """
    Demonstrate an info log with module and function tagging.
    """
    logger.info("This is an info message.")


def demo_debug_logging():
    """
    Demonstrate a debug log with module and function tagging.
    """
    logger.debug("This is a debug message.")


def demo_exception_logging():
    """
    Demonstrate logging an exception with traceback.
    """
    try:
        1 / 0
    except ZeroDivisionError as e:
        logger.exception("An exception occurred!")


if __name__ == "__main__":
    demo_info_logging()
    demo_debug_logging()
    demo_exception_logging()
