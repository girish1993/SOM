import logging
import time
from functools import wraps

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def timer(func):
    """A decorator that prints the execution time of a function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        logger.info(f"Initiating function {func.__name__!r}")
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        logger.info(f"Function {func.__name__!r} took: {execution_time:.4f} seconds")
        return result

    return wrapper
