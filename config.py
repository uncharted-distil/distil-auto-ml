import os

DB_LOCATION = os.getenv("DB_URI", "test.db")

# Debug flag to output more verbose logging
# - defaults to False
DEBUG = os.getenv("DEBUG", False)

# Configurable output directory for saving machine learning model pickles
# - defaults to ../output
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")

# Port to make worker service available on
# PORT = os.getenv("PORT", "45042")
PORT = os.getenv("PORT", "45042")

# Configurable filename for output logs
LOG_FILENAME = os.getenv("LOG_FILENAME", "distil-auto-ml.log")

# User agent to supply to TA3 Systems
SERVER_USER_AGENT='qntfy_ta2'

# Primitives static file directory
D3MSTATICDIR = os.getenv("D3MSTATICDIR", '/static')

# Enable GPU pipelines - "auto" will try to detect, "true" and "false" will force
GPU = os.getenv("GPU", "auto")

# Solution serach progress update message interval in seconds
PROGRESS_INTERVAL = float(os.getenv("PROGRESS_INTERVAL", 10.0))

# maximum number of augment columns to support
AUG_MAX_COLS = int(os.getenv("AUG_MAX_COLS", 50))

# maximum number of augment rows to support
AUG_MAX_ROWS = int(os.getenv("AUG_MAX_ROWS", 50000))

# maximum amount of time for hyperparam tuning in seconds
TIME_LIMIT = int(os.getenv("TIME_LIMIT", 600))

# use untuned/internally tuned pipelines (faster) or external tuning (better results)
HYPERPARAMETER_TUNING = os.getenv("HYPERPARAMETER_TUNING", True) == 'True'