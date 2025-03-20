import logging
import os
import unittest

from palzlib.zlogger.logger import Logger  # Update with the correct module name


class TestLogger(unittest.TestCase):
    def setUp(self):
        self.log_dir = os.path.dirname(os.path.abspath(__file__))
        self.logger_name = "test_logger"
        self.log_file = "test_log.log"

    def tearDown(self):
        log_file_path = os.path.join(self.log_dir, self.log_file)

        if os.path.exists(log_file_path):
            os.remove(log_file_path)

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
        log_file = os.path.join(self.log_dir, self.log_file)
        logger_instance = Logger(name=self.logger_name, log_file=log_file)
        logger = logger_instance.get_logger()

        self.assertEqual(logger.name, self.logger_name)
        self.assertEqual(logger.level, logging.INFO)
        self.assertEqual(len(logger.handlers), 1)
        self.assertIsInstance(logger.handlers[0], logging.handlers.RotatingFileHandler)

        logger.info("Test log entry")

        with open(log_file, "r") as f:
            log_contents = f.read()
            self.assertIn("Test log entry", log_contents)

    def test_logger_custom_log_level(self):
        """Test if logger respects custom log level."""
        logger_instance = Logger(name=self.logger_name, log_level=logging.DEBUG)
        logger = logger_instance.get_logger()

        self.assertEqual(logger.level, logging.DEBUG)


if __name__ == "__main__":
    unittest.main()
