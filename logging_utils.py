#!/usr/bin/env python3
"""
Logging utilities for the gamma-ray streaming simulation.
"""
import logging
import sys
import time
import functools
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
    log_path = Path(log_file)
    log_path.parent.mkdir(exist_ok=True, parents=True)
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

# Create a default logger
logger = setup_logger()

class LogSection:
    """Context manager for logging a section of code with timing."""
    def __init__(self, section_name):
        self.section_name = section_name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"Starting: {self.section_name}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        if exc_type:
            logger.error(f"Error in {self.section_name}: {exc_val}")
        else:
            logger.info(f"Completed: {self.section_name} ({elapsed_time:.2f} seconds)")
        return False  # Don't suppress exceptions

def timeit(func):
    """Decorator for timing function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"Starting: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            logger.info(f"Completed: {func.__name__} ({elapsed_time:.2f} seconds)")
            return result
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"Error in {func.__name__} after {elapsed_time:.2f} seconds: {str(e)}")
            raise
    return wrapper

@contextmanager
def log_errors(context_name="Operation", reraise=True):
    """Context manager for catching and logging errors."""
    try:
        yield
    except Exception as e:
        logger.error(f"Error in {context_name}: {str(e)}", exc_info=True)
        if reraise:
            raise

def set_log_level(level):
    """Set the log level for all handlers."""
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)
