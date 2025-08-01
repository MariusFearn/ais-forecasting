"""Centralized logging configuration for maritime discovery project."""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

def setup_logging(
    level: str = "INFO", 
    log_file: Optional[str] = None,
    include_timestamp: bool = True,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup consistent logging configuration across the project.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to
        include_timestamp: Whether to include timestamps in log messages
        format_string: Custom format string for log messages
        
    Returns:
        Configured logger instance
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    # Create formatter
    if format_string is None:
        if include_timestamp:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        else:
            format_string = '%(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Create project-specific logger
    logger = logging.getLogger('maritime_discovery')
    logger.info(f"Logging configured at {level} level")
    
    if log_file:
        logger.info(f"Log file: {log_file}")
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f'maritime_discovery.{name}')

def log_performance_metrics(
    logger: logging.Logger,
    operation: str,
    duration_seconds: float,
    data_size: int,
    memory_mb: Optional[float] = None
) -> None:
    """
    Log standardized performance metrics.
    
    Args:
        logger: Logger instance
        operation: Name of the operation
        duration_seconds: Time taken in seconds
        data_size: Size of data processed (records, bytes, etc.)
        memory_mb: Peak memory usage in MB (optional)
    """
    metrics = [
        f"Operation: {operation}",
        f"Duration: {duration_seconds:.2f}s",
        f"Data size: {data_size:,}",
        f"Rate: {data_size/duration_seconds:,.0f} items/sec"
    ]
    
    if memory_mb:
        metrics.append(f"Memory: {memory_mb:.1f} MB")
    
    logger.info("PERFORMANCE - " + " | ".join(metrics))
