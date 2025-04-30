#!/usr/bin/env python3
"""
Logging utilities for the gamma-ray streaming simulation.
"""
import logging
import sys
import time
from functools import wraps
from pathlib import Path
from contextlib import contextmanager

# Configure the logger
def setup_logger(name="gamma_streaming", log_file="simulation.log", level=logging.INFO):
    """Set up a logger with both file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear handlers if they exist
    if logger.handlers:
        logger.handlers.clear()
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    return logger

# Create a global logger instance
logger = setup_logger()

@contextmanager
def LogSection(section_name):
    """
    Context manager for logging sections with timing.
    
    Parameters:
    -----------
    section_name : str
        Name of the section to log
        
    Examples:
    ---------
    >>> with LogSection("Processing data"):
    >>>     process_data()
    """
    start_time = time.time()
    logger.info(f"⏱️  Starting: {section_name}")
    try:
        yield
    except Exception as e:
        logger.error(f"❌ Error in {section_name}: {str(e)}")
        raise
    finally:
        elapsed = time.time() - start_time
        logger.info(f"✅ Completed: {section_name} (in {elapsed:.2f} seconds)")

def timeit(func):
    """
    Decorator to measure and log the execution time of a function.
    
    Parameters:
    -----------
    func : callable
        Function to be decorated
        
    Returns:
    --------
    callable
        Decorated function that logs execution time
        
    Examples:
    ---------
    >>> @timeit
    >>> def long_running_function():
    >>>     # function body
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"⏱️  Starting function: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            logger.error(f"❌ Error in {func.__name__}: {str(e)}")
            raise
        finally:
            elapsed = time.time() - start_time
            logger.info(f"✅ Completed function: {func.__name__} (in {elapsed:.2f} seconds)")
    return wrapper
