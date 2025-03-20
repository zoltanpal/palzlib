import unittest
import logging
import os
from palzlib.zlogger.logger import Logger  # Update with the correct module name


class TestLogger(unittest.TestCase):
    def setUp(self):
        self.logger_name = "test_logger"
        self.log_file = "test_log.log"

    def tearDown(self):
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

    def test_logger_console_output(self):
        """Test if logger initializes correctly with console output."""
        logger_instance = Logger(name=self.logger_name)
        logger = logger_instance.get_logger()

        self.assertEqual(logger.name, self.logger_name)
        self.assertEqual(logger.level, logging.INFO)
        self.assertEqual(len(logger.handlers), 1)
        self.assertIsInstance(logger.handlers[0], logging.StreamHandler)

    def test_logger_file_output(self):
        """Test if logger writes to file when log_file is provided."""
        logger_instance = Logger(name=self.logger_name, log_file=self.log_file)
        logger = logger_instance.get_logger()

        self.assertEqual(logger.name, self.logger_name)
        self.assertEqual(logger.level, logging.INFO)
        self.assertEqual(len(logger.handlers), 1)
        self.assertIsInstance(logger.handlers[0], logging.handlers.RotatingFileHandler)

        logger.info("Test log entry")

        with open(self.log_file, "r") as f:
            log_contents = f.read()
            self.assertIn("Test log entry", log_contents)

    def test_logger_custom_log_level(self):
        """Test if logger respects custom log level."""
        logger_instance = Logger(name=self.logger_name, log_level=logging.DEBUG)
        logger = logger_instance.get_logger()

        self.assertEqual(logger.level, logging.DEBUG)

    def test_logger_directory_creation(self):
        """Test if logger creates the directory when the log file path includes a non-existent directory."""
        log_dir = "test_logs"
        log_path = os.path.join(log_dir, "logfile.log")

        logger_instance = Logger(name=self.logger_name, log_file=log_path)

        self.assertTrue(os.path.exists(log_dir))

        if os.path.exists(log_path):
            os.remove(log_path)
        os.rmdir(log_dir)


if __name__ == "__main__":
    unittest.main()
