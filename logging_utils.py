#!/usr/bin/env python3
"""
Logging utilities for the gamma-ray streaming simulation.
"""
import logging
import sys
import time
from functools import wraps
from pathlib import Path

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

# Create the main logger
logger = setup_logger()

# Decorator for timing functions
def timeit(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"Starting {func.__name__}...")
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        if duration < 60:
            logger.info(f"Completed {func.__name__} in {duration:.2f} seconds")
        elif duration < 3600:
            logger.info(f"Completed {func.__name__} in {duration/60:.2f} minutes")
        else:
            logger.info(f"Completed {func.__name__} in {duration/3600:.2f} hours")
        return result
    return wrapper

# Progress bar
def progress_bar(current, total, bar_length=50):
    """Display a text progress bar."""
    percent = float(current) / total
    arrow = '-' * int(round(percent * bar_length))
    spaces = ' ' * (bar_length - len(arrow))
    
    sys.stdout.write(f"\r[{arrow}{spaces}] {int(percent*100)}% ({current}/{total})")
    sys.stdout.flush()
    
    if current == total:
        sys.stdout.write('\n')

# Create a context manager for sections of code
class LogSection:
    """Context manager for logging sections of code."""
    def __init__(self, section_name, level=logging.INFO):
        self.section_name = section_name
        self.level = level
        
    def __enter__(self):
        logger.log(self.level, f"=== Starting section: {self.section_name} ===")
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        if exc_type is not None:
            logger.error(f"Error in section {self.section_name}: {exc_val}")
        elif duration < 60:
            logger.log(self.level, f"=== Completed section: {self.section_name} in {duration:.2f} seconds ===")
        elif duration < 3600:
            logger.log(self.level, f"=== Completed section: {self.section_name} in {duration/60:.2f} minutes ===")
        else:
            logger.log(self.level, f"=== Completed section: {self.section_name} in {duration/3600:.2f} hours ===")
