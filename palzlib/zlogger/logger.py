import logging
import os
from logging.handlers import RotatingFileHandler


class Logger:
    def __init__(
        self,
        name: str,
        log_file: str = None,
        log_level: int = logging.INFO,
        file_size: int = 10**6,
        backup_count: int = 5,
    ):
        """
        Initializes the logger with a specified name, file, level, and output options.

        Parameters:
        - name (str): The name of the logger.
        - log_file (str, optional): Path to the log file. If None or empty, logs to console.
        - log_level (int): Logging level (default: logging.INFO).
        - file_size (int): Max file size before rotation (default: 1MB).
        - backup_count (int): Number of backup files to keep (default: 5).
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        self.logger.handlers.clear()  # Clear existing handlers to prevent duplicates

        # Define log format
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        if log_file and log_file.strip():
            # Ensure the directory exists
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

            # File handler
            file_handler = RotatingFileHandler(
                log_file, maxBytes=file_size, backupCount=backup_count
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            return  # Exit early to avoid adding a console handler

        # Console handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

    def get_logger(self):
        return self.logger
