"""
example_combined.py
===================

Demonstrates the combined use of debugging utilities and tagged logging.
"""

from super_utils.debug import DEBUG_TAG, DEBUG_PRINT_EXCEPTION
from super_utils.logging import setup_tagged_logger
from inspect import currentframe

# Initialize logger
logger = setup_tagged_logger()

def combined_demo():
    """
    Combined demonstration of DEBUG_TAG and tagged logging.
    """
    try:
        DEBUG_TAG(currentframe(), "Starting combined demo...")
        logger.info("This is a tagged info message.")
        1 / 0  # Trigger exception
    except:
        DEBUG_PRINT_EXCEPTION()
        logger.exception("An exception was logged with tagged details!")

if __name__ == "__main__":
    combined_demo()
