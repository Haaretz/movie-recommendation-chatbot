import logging
import os

from pythonjsonlogger import jsonlogger

APP_NAME = os.environ.get("APP_NAME", "UNKNOWN_APP_NAME")

logger_level = logging.DEBUG if os.environ.get("LOG_LEVEL", "debug") == "debug" else logging.INFO
logger = logging.getLogger(f"{APP_NAME}_logger")
logger.setLevel(logger_level)

formatter = jsonlogger.JsonFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")


# Create a stream handler (for stdout) and add it to the logger
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


if __name__ == "__main__":

    logger.debug("Debug message (לא יוצג כי רמה INFO)")
    logger.info("Info message from logger_config")
    logger.warning("Warning message from logger_config")
    logger.error("Error message from logger_config")
