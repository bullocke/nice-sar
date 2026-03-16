"""Package-level configuration and logging setup."""

import logging


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the nice_sar namespace.

    Args:
        name: Module name, typically ``__name__``.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
