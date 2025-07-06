import logging
from logging.handlers import RotatingFileHandler
import os

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "app.log")

os.makedirs(LOG_DIR, exist_ok=True)

# Logger setup
logger = logging.getLogger("FinGuardPro")
logger.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")

# File handler with rotation (1MB per file, keep 5 backups)
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=5)
file_handler.setFormatter(formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# Attach handlers if not already
if not logger.hasHandlers():
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
