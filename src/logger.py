import logging
import sys
from typing import Optional


_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_LOG_LEVEL = logging.INFO
_LOGGERS = {}

def configure_logging(level: Optional[str] = None):
    """
    Configure the global logging settings.
    
    Args:
        level: Logging level (e.g., "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    """
    global _LOG_LEVEL
    
    if level:
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        _LOG_LEVEL = level_map.get(level.upper(), logging.INFO)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(_LOG_LEVEL)
    
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
   
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(_LOG_FORMAT)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name (typically __name__ of the module)
        
    Returns:
        Configured logger instance
    """
    if name not in _LOGGERS:
        logger = logging.getLogger(name)
        logger.setLevel(_LOG_LEVEL)
        _LOGGERS[name] = logger
    
    return _LOGGERS[name]