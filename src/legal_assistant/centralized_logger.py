import logging
import os

class CentralizedLogger:
    _shared_logger = None

    def __init__(self, log_file_name="assistant.log", log_dir="~/log", log_level=logging.INFO):
        if not CentralizedLogger._shared_logger:
            # Expand user directory & create log directory
            log_dir = os.path.expanduser(log_dir)
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, log_file_name)

            # Create named logger
            logger = logging.getLogger("CentralizedLogger")
            logger.setLevel(log_level)

            # Create file handler and formatter
            file_handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)

            # Add handler to the logger
            logger.addHandler(file_handler)

            # Prevent propagation to root logger if needed
            logger.propagate = False  # Set to True if you want duplicate logs in console

            CentralizedLogger._shared_logger = logger

    def get_logger(self):
        return CentralizedLogger._shared_logger