import logging
import os

from pythonjsonlogger.jsonlogger import JsonFormatter

APP_NAME = os.environ.get("APP_NAME", "MRC")

logger_level = logging.DEBUG if os.environ.get("LOG_LEVEL", "debug") == "debug" else logging.INFO


def get_logger(name: str):
    logger = logging.getLogger(f"{APP_NAME}_{name}_logger")
    logger.setLevel(logger_level)

    if not logger.hasHandlers():
        # Create a formatter to customize the log message format
        formatter = JsonFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger
