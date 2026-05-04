import sys

from loguru import logger as _logger

from app.config import PROJECT_ROOT

_print_level = "INFO"

def define_log_level(print_level="INFO", logfile_level="DEBUG", name: str = None):
    global _print_level
    _print_level = print_level

    _logger.remove()
    _logger.add(sys.stderr, level=print_level)
    _logger.add(PROJECT_ROOT / f"logs/{name or logfile_level}.log", level=logfile_level)
    return _logger

logger = define_log_level()

