import logging
import time
from functools import wraps

logger = logging.getLogger(__name__)


def timer(func):
    """A decorator that prints the execution time of a function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"Function {func.__name__!r} took: {execution_time:.4f} seconds")
        return result

    return wrapper
