import logging
import sys


def get_logger(name: str, level=logging.DEBUG, strategy="stream", log_file="log.out"):
    """
    Return a configured logger.
    - name (str) should be the module name.
    - level should be a logging level, e.g., LOGGING.INFO
    - strategy should be one of "stream", "file", or "combined"
    - log_file: path to a file to log to. 
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.hasHandlers():
        if strategy == 'file' or strategy == 'combined':
            handler = logging.FileHandler(log_file)
        if strategy == 'stream' or strategy == 'combined':
            handler = logging.StreamHandler(sys.stdout)
        else:
            handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s'
            '[in %(pathname)s:%(lineno)d]'
        ))
        handler.setLevel(level)
        logger.addHandler(handler)
    logger.debug("returning a logger set to level: {} for module: {}".format(level, name))
    return logger    