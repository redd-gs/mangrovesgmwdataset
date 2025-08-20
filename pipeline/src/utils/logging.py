import logging
import os

def setup_logging(log_file='pipeline.log', level=logging.INFO):
    """Set up logging configuration."""
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        filename=log_file,
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=level
    )

def log_info(message):
    """Log an info message."""
    logging.info(message)

def log_warning(message):
    """Log a warning message."""
    logging.warning(message)

def log_error(message):
    """Log an error message."""
    logging.error(message)

def log_debug(message):
    """Log a debug message."""
    logging.debug(message)