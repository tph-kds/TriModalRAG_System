import os
import sys
import logging
from src.trim_rag.exception import MyException
from src.trim_rag.config import LoggerArgumentsConfig


class LoggerHandler(logging.Handler):
    def __init__(self):
        super(LoggerHandler, self).__init__()
        self.log = ""

    def reset(self):
        self.log = ""

    def emit(self, record):
        if record.name == "httpx":
            return

        log_entry = self.fotmat(record)
        self.log += log_entry
        self.log += "\n\n"


class MainLoggerHandler(LoggerHandler):
    def __init__(self, logger_config: LoggerArgumentsConfig):
        super(MainLoggerHandler, self).__init__()
        self.name = logger_config.name
        self.format_logging = logger_config.format_logging
        self.datefmt_logging = logger_config.datefmt_logging
        self.log_dir = logger_config.log_dir
        self.name_file_logs = logger_config.name_file_logs
        self.logger = self.get_logger()

    def reset_logging(self):
        r"""
        Removes basic config of root logger
        """
        root = logging.getLogger()
        list(map(root.removeHandler, root.handlers))
        list(map(root.removeFilter, root.filters))

    def get_logger(self) -> logging.Logger:
        log_filepath = os.path.join(self.log_dir, self.name_file_logs)
        os.makedirs(self.log_dir, exist_ok=True)

        formatter = logging.Formatter(
            fmt=self.format_logging, datefmt=self.datefmt_logging
        )
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(log_filepath)
        file_handler.setFormatter(formatter)

        logger = logging.getLogger(self.name)
        logger.setLevel(logging.INFO)

        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)

        return logger

    def log_message(self, level, message):
        if level == "debug":
            self.logger.debug(message)
        elif level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        elif level == "critical":
            self.logger.critical(message)
